//! Forward kinematics, thrust, sensing, and the full per-tick orchestration
//! -- the Rust-native replacement for pixel_world.py's `World.tick()`. Every
//! step that used to be a Python loop over `Individual` objects is now a
//! loop over SoA component arrays with no interpreter overhead.
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::combat;
use crate::individuals::{ACT_DIM, MEM_DIM, SENSE_DIM};
use crate::spatial::SpatialGrid;
use crate::terrain::TerrainKind;
use crate::{Corpse, ExperienceRow, Weather, World};

fn normal(rng: &mut rand_pcg::Pcg64, mean: f32, std: f32) -> f32 {
    if std <= 0.0 { return mean; }
    Normal::new(mean, std).unwrap().sample(rng)
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Real forward kinematics through the growth-order joint chain, with a
/// per-pixel-flex-scaled traveling-wave oscillation -- matches
/// pixel_world.py's `Individual.world_positions`.
pub fn world_positions(world: &World, slot: usize, t: f32) -> Vec<[f32; 2]> {
    world_positions_at(world, slot, t, world.individuals.root_pos[slot])
}

/// Same as `world_positions`, but for a hypothetical root position rather
/// than the individual's actual current one -- used to get the EXACT shape
/// a candidate move would produce (current heading/flex/phase/scale, just a
/// different root), for terrain collision. This is not an approximation:
/// since every other pixel's position is root + a root-independent function
/// of heading/flex/amp/phase/rest_angle, translating the root by any delta
/// is mathematically identical to recomputing the whole chain at the new
/// root -- AS LONG AS the two calls use the same t/heading/flex/etc, which
/// is exactly how the collision check below uses this.
pub fn world_positions_at(world: &World, slot: usize, t: f32, root: [f32; 2]) -> Vec<[f32; 2]> {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let heading = world.individuals.heading[slot];
    let amp = world.individuals.bend_amplitude[slot];
    let freq = world.individuals.bend_frequency[slot];
    let phase = world.individuals.bend_phase[slot];
    // Growth is pure inflation, never new anatomy: the body PLAN (this joint
    // chain, walked below) is fixed for life at birth: only this uniform
    // per-joint length multiplier grows. Everything downstream of position
    // -- thrust (drag scales with segment length), contact/collision
    // (point-to-point distance), and the frontend's rendering (it just
    // draws these positions) -- gets bigger automatically because the
    // geometry itself is bigger, with no separate size formula to keep in
    // sync anywhere else.
    let scale = world.individuals.size_scale[slot];

    let mut angles = vec![0f32; count];
    let mut positions = vec![[0f32; 2]; count];
    for k in 0..count {
        let flex = world.pixels.flex[offset + k];
        let raw_wave = flex * amp * (std::f32::consts::TAU * freq * t + phase + (k as f32) * 0.7).sin();
        // Joint angle limits: a real hinge constraint on how far THIS
        // joint's animated bend can deviate from its rest pose, heritable
        // per part (pixels.rs's min_angle/max_angle). Previously every
        // joint swung through the same unbounded range regardless of what
        // kind of part it was -- a stiff plate and a whip-like tail moved
        // identically except for amplitude. A narrow range reads as a
        // rigid/braced joint, a wide one as a loose/flexible one, and nothing
        // here decides which is good; it's just now possible to evolve.
        let wave = raw_wave.clamp(world.pixels.min_angle[offset + k], world.pixels.max_angle[offset + k]);
        let parent = world.pixels.parent_idx[offset + k];
        // A part's own evolved `size` stretches ITS segment specifically
        // (on top of the individual-wide size_scale inflation) -- a body
        // can evolve one big limb and several small ones, not just scale
        // uniformly everywhere.
        let seg_len = scale * world.pixels.size[offset + k];
        if parent < 0 {
            angles[k] = heading + wave;
            positions[k] = root;
        } else {
            let p = parent as usize;
            angles[k] = angles[p] + world.pixels.rest_angle[offset + k] + wave;
            positions[k] = [positions[p][0] + seg_len * angles[k].cos(), positions[p][1] + seg_len * angles[k].sin()];
        }
    }
    positions
}

pub fn pixel_velocities(world: &World, slot: usize, t: f32, eps: f32) -> Vec<[f32; 2]> {
    let plus = world_positions(world, slot, t + eps);
    let minus = world_positions(world, slot, t - eps);
    plus.iter().zip(minus.iter()).map(|(p, m)| [(p[0] - m[0]) / (2.0 * eps), (p[1] - m[1]) / (2.0 * eps)]).collect()
}

pub fn fluid_thrust_force(world: &World, slot: usize, pos: &[[f32; 2]], vel: &[[f32; 2]]) -> [f32; 2] {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = pos.len();
    let mut force = [0f32; 2];
    for k in 1..count {
        let parent = world.pixels.parent_idx[offset + k];
        if parent < 0 { continue; }
        let p = parent as usize;
        let sx = pos[k][0] - pos[p][0];
        let sy = pos[k][1] - pos[p][1];
        let seg_len = (sx * sx + sy * sy).sqrt();
        if seg_len < 1e-8 { continue; }
        let (tx, ty) = (sx / seg_len, sy / seg_len);
        let (vx, vy) = (vel[k][0], vel[k][1]);
        let v_par = vx * tx + vy * ty;
        let (vparx, vpary) = (v_par * tx, v_par * ty);
        let (vperpx, vperpy) = (vx - vparx, vy - vpary);
        force[0] += -(crate::DRAG_PARALLEL * vparx + crate::DRAG_PERPENDICULAR * vperpx) * seg_len;
        force[1] += -(crate::DRAG_PARALLEL * vpary + crate::DRAG_PERPENDICULAR * vperpy) * seg_len;
    }
    force
}

/// The real resting height for a non-digging body at world-x `x`: normally
/// just the nominal sand-band top, EXCEPT where an "on_floor" rock cluster
/// (see terrain.rs) pokes up above it -- resting a body at the nominal sand
/// line there would embed it in rock instead. Scans from the nominal line
/// upward for the first non-Rock cell (never scans the sand band itself,
/// since by construction it's never Empty there).
fn sand_surface_height(world: &World, x: u32, base_floor_height: f32) -> f32 {
    let start = base_floor_height as u32;
    let max_scan = start + 40; // generous vs. the largest possible rock cluster radius
    let mut y = start;
    while y < world.size && y < max_scan {
        if world.terrain.at(x, y) != TerrainKind::Rock {
            return y as f32;
        }
        y += 1;
    }
    base_floor_height
}

/// Sum of every pixel's own evolved `size` -- a real measure of how much
/// BODY this individual actually has, used for mass and (in combat.rs) for
/// scaling how hard it is to kill in one hit. pixel_count alone treated a
/// body of ten tiny parts and ten huge ones as identical; this doesn't.
pub fn body_size_sum(world: &World, slot: usize) -> f32 {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    world.pixels.size[offset..offset + count].iter().sum()
}

/// Is `pos` either literally inside solid ground, or resting right on top
/// of it? The latter matters because "resting on a surface" means
/// occupying the open cell directly above the solid one, not the solid
/// cell itself -- a naive "is THIS cell Sand/Rock" check misses every
/// non-digging individual sitting on the sand floor, since the sand-
/// surface correction elsewhere deliberately settles them into the empty
/// cell immediately above the sand band, not into the sand band itself.
fn resting_on_solid_ground(world: &World, pos: [f32; 2]) -> bool {
    let (gx, gy) = grid_xy(world, pos);
    if matches!(world.terrain.at(gx, gy), TerrainKind::Rock | TerrainKind::Sand) {
        return true;
    }
    gy > 0 && matches!(world.terrain.at(gx, gy - 1), TerrainKind::Rock | TerrainKind::Sand)
}

pub(crate) fn grid_xy(world: &World, pos: [f32; 2]) -> (u32, u32) {
    let x = pos[0].clamp(0.0, world.size as f32 - 1.0) as u32;
    let y = pos[1].clamp(0.0, world.size as f32 - 1.0) as u32;
    (x, y)
}

/// exp(-distance) similarity, in (0, 1], between `slot`'s kin_signature and
/// the CLOSEST spatially-nearby individual's -- "is there a close relative
/// right next to me", not "who in the whole world is my closest relative".
/// That framing is what makes it actually useful as a suppress-aggression
/// signal: it's only large exactly when a fight/crowd-pressure decision
/// about a nearby target is also being made.
/// Returns (similarity to the most-similar nearby individual, total
/// CONSPECIFIC density around this one). The second value is the sum of
/// kin-similarity over all neighbors, so a near-identical neighbor counts
/// ~1.0 and an unrelated one ~0 -- i.e. "how many of my own kind are
/// packed around me", which is deliberately NOT the same as the generic
/// crowding the quorum field already measures. It drives the
/// Janzen-Connell pathogen pressure in the tick loop; see
/// PATHOGEN_DAMAGE_RATE in lib.rs. Accumulated inside the scan that was
/// already happening for kin similarity, so it costs no extra traversal.
// --- Body-part derived stats. All read the cached per-individual tally (see
// individuals::recompute_part_counts), so these are a handful of arithmetic
// ops rather than a pixel scan, and stay cheap enough to call in the hot
// per-tick paths. Each is a plain multiplier with a ceiling: stacking twenty
// eyes should help less and less, or a single degenerate "all one organ"
// body plan would dominate and the interesting mixed anatomies would never
// get a look in.
fn part_bonus(world: &World, slot: usize, kind: u8, per_part: f32, cap: f32) -> f32 {
    let n = world.individuals.part_counts[slot][kind as usize] as f32;
    (1.0 + per_part * n).min(cap)
}

/// Sight range, or 0 for a body with no eyes at all.
///
/// Vision used to be a free universal sense that every creature had in full,
/// with eyes merely extending it -- so nothing was ever selected FOR having
/// eyes, and every animal was equally aware of the world regardless of its
/// anatomy. Senses are supplied by organs now: no eye, no visual input. The
/// brain always has the same input slots, but a slot only carries signal if
/// the body has the component that feeds it, which is what makes sensory
/// anatomy a real evolutionary decision rather than decoration.
pub(crate) fn vision_range_of(world: &World, slot: usize) -> f32 {
    let eyes = world.individuals.part_counts[slot][crate::pixels::PART_EYE as usize];
    if eyes == 0 {
        return 0.0;
    }
    crate::VISION_RANGE
        * part_bonus(world, slot, crate::pixels::PART_EYE, crate::EYE_VISION_BONUS, crate::EYE_VISION_MAX)
}

pub(crate) fn bite_multiplier(world: &World, slot: usize) -> f32 {
    part_bonus(world, slot, crate::pixels::PART_MOUTH, crate::MOUTH_BITE_BONUS, crate::MOUTH_BITE_MAX)
}

pub(crate) fn grip_multiplier(world: &World, slot: usize) -> f32 {
    part_bonus(world, slot, crate::pixels::PART_TENTACLE, crate::TENTACLE_GRIP_BONUS, crate::TENTACLE_GRIP_MAX)
}

pub(crate) fn armor_multiplier(world: &World, slot: usize) -> f32 {
    part_bonus(world, slot, crate::pixels::PART_ARMOR, crate::ARMOR_TOUGHNESS_BONUS, crate::ARMOR_TOUGHNESS_MAX)
}

/// A gut extracts more energy from the same meal. This is what makes a
/// grazing or scavenging life history viable next to simply killing things:
/// the same mouthful is worth more to a body that invested in digesting it.
pub(crate) fn digestion_multiplier(world: &World, slot: usize) -> f32 {
    let n = world.individuals.part_counts[slot][crate::pixels::PART_GUT as usize] as f32;
    (1.0 + crate::GUT_DIGESTION_BONUS * n).min(crate::GUT_DIGESTION_MAX)
}

pub(crate) fn thrust_multiplier(world: &World, slot: usize) -> f32 {
    part_bonus(world, slot, crate::pixels::PART_FLIPPER, crate::FLIPPER_THRUST_BONUS, crate::FLIPPER_THRUST_MAX)
}

/// Total metabolic upkeep multiplier for this body: every part costs, and
/// specialised organs cost more than plain structural tissue. Returned as an
/// effective part count so the caller's existing per-pixel formula is
/// unchanged in shape.
pub(crate) fn metabolic_part_load(world: &World, slot: usize) -> f32 {
    let counts = &world.individuals.part_counts[slot];
    let mut load = 0.0;
    for kind in 0..crate::pixels::PART_KIND_COUNT as usize {
        load += counts[kind] as f32 * crate::PART_METABOLISM[kind];
    }
    load
}

fn kin_similarity_nearest(world: &World, slot: usize, grid: &SpatialGrid) -> (f32, f32) {
    let pos = world.individuals.root_pos[slot];
    let my_sig = world.individuals.kin_signature[slot];
    let mut best = 0.0f32;
    let mut conspecific_density = 0.0f32;
    for other in grid.nearby(pos) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        let other_sig = world.individuals.kin_signature[other];
        let d2: f32 = (0..crate::KIN_DIM).map(|k| (my_sig[k] - other_sig[k]).powi(2)).sum();
        let sim = (-d2.sqrt()).exp();
        // Sharpened deliberately: raw `sim` decays too gently to represent
        // a HOST-SPECIFIC pathogen -- an unrelated neighbor still scored
        // ~0.3, so in any crowd a genuinely rare individual accumulated
        // almost as much "conspecific" density as a member of a
        // monoculture (measured: 9.83 vs 13.56, a useless 1.4x). Raising
        // it to a power collapses that contamination (a sibling at
        // sim~0.85 still counts ~0.5, an unrelated neighbor at ~0.3 counts
        // ~0.008) so this measures "my own kind specifically", which is
        // what makes it a diversity mechanism rather than a second, redundant
        // copy of the species-blind quorum field.
        conspecific_density += sim.powi(crate::CONSPECIFIC_KERNEL_EXPONENT);
        if sim > best { best = sim; }
    }
    (best, conspecific_density)
}

/// Real vision: direction + proximity to the nearest meaningfully-bigger
/// body (a plausible threat), the nearest meaningfully-smaller one (a
/// plausible meal), AND the nearest actually-eligible mate (opposite sex,
/// mature, not a female mid-cooldown) within VISION_RANGE. Uses the same
/// spatial grid as collision/crowd checks, so it's the same cost class as
/// those, not a new bottleneck. A "similar size" neighbor (within 15%)
/// counts as neither threat nor prey -- vision distinguishes them by
/// scale, not by picking a fight.
///
/// The mate channel exists specifically to make mate-competition
/// ("nuptial aggression") POSSIBLE to evolve: a brain that senses both a
/// nearby mate and a nearby same-scale rival now has the raw material to
/// learn "fight only when a mate is at stake", instead of aggression
/// having no relationship to reproduction at all. Nothing here rewards or
/// triggers that -- it's still entirely a possible behavior, not an
/// enforced one.
pub(crate) fn vision(world: &World, slot: usize, grid: &SpatialGrid) -> [f32; 9] {
    let pos = world.individuals.root_pos[slot];
    // Sight range is anatomical now: an eyeless body barely perceives past
    // its own skin, while one that invested in eyes sees far enough to
    // actually hunt or flee rather than blunder into things.
    let range = vision_range_of(world, slot);
    if range <= 0.0 {
        return [0f32; 9]; // blind: the visual input slots stay dead
    }
    let my_size = body_size_sum(world, slot) * world.individuals.size_scale[slot];
    let my_female = world.individuals.female[slot];
    let mut best_threat: Option<(f32, [f32; 2])> = None;
    let mut best_prey: Option<(f32, [f32; 2])> = None;
    let mut best_mate: Option<(f32, [f32; 2])> = None;
    for other in grid.nearby_radius(pos, range) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        let other_pos = world.individuals.root_pos[other];
        let d = dist(other_pos, pos);
        if d > range || d < 1e-6 { continue; }
        let other_size = body_size_sum(world, other) * world.individuals.size_scale[other];
        let delta = [other_pos[0] - pos[0], other_pos[1] - pos[1]];
        if other_size > my_size * 1.15 {
            if best_threat.map_or(true, |(bd, _)| d < bd) { best_threat = Some((d, delta)); }
        } else if other_size < my_size * 0.85 {
            if best_prey.map_or(true, |(bd, _)| d < bd) { best_prey = Some((d, delta)); }
        }
        let other_mature = world.individuals.age[other] as f32 >= crate::MATURITY_AGE * world.maturity_multiplier;
        let other_recovered = !world.individuals.female[other] || world.individuals.ticks_since_reproduced[other] >= crate::FEMALE_REPRODUCTION_COOLDOWN;
        if world.individuals.female[other] != my_female && other_mature && other_recovered {
            if best_mate.map_or(true, |(bd, _)| d < bd) { best_mate = Some((d, delta)); }
        }
    }
    let mut out = [0f32; 9];
    if let Some((d, delta)) = best_threat {
        out[0] = delta[0] / range;
        out[1] = delta[1] / range;
        out[2] = 1.0 - d / range;
    }
    if let Some((d, delta)) = best_prey {
        out[3] = delta[0] / range;
        out[4] = delta[1] / range;
        out[5] = 1.0 - d / range;
    }
    if let Some((d, delta)) = best_mate {
        out[6] = delta[0] / range;
        out[7] = delta[1] / range;
        out[8] = 1.0 - d / range;
    }
    out
}

/// Returns the sense vector plus the RAW (un-normalized) conspecific
/// density, which the pathogen drain needs at full precision: the sense
/// vector's copy is tanh-squashed (correct for a network input, bounded)
/// but saturates around a density of ~3-4, which is exactly the range the
/// Janzen-Connell response has to be able to tell apart -- reading damage
/// off the squashed value made a dense monoculture and an ordinary family
/// group pay almost the same, turning a diversity mechanism into a flat
/// population-wide tax (measured: it cut lineage counts and crashed
/// population in all 3 A/B seeds).
fn sense(world: &World, slot: usize, grid: &SpatialGrid) -> ([f32; SENSE_DIM], f32) {
    let pos = world.individuals.root_pos[slot];
    let (fgx, fgy) = crate::fields::Fields::gradient_at_range(&world.fields.food, world.size, pos, crate::FOOD_SMELL_RANGE);
    let (phgx, phgy) = crate::fields::Fields::gradient(&world.fields.pheromone, world.size, pos);
    let (blgx, blgy) = crate::fields::Fields::gradient(&world.fields.blood, world.size, pos);
    let (acgx, acgy) = crate::fields::Fields::gradient(&world.fields.acid, world.size, pos);
    let (ligx, ligy) = crate::fields::Fields::gradient(&world.fields.light, world.size, pos);
    let heading = world.individuals.heading[slot];
    let fwd = [pos[0] + crate::VISION_LOOKAHEAD * heading.cos(), pos[1] + crate::VISION_LOOKAHEAD * heading.sin()];
    let fwd_food = crate::fields::Fields::sample(&world.fields.food, world.size, fwd);
    let fwd_light = crate::fields::Fields::sample(&world.fields.light, world.size, fwd);
    let energy_norm = (world.individuals.energy[slot] / 20.0).tanh();
    let size_norm = (body_size_sum(world, slot) * world.individuals.size_scale[slot] / 10.0).tanh();
    let (kin_sim, conspecific_density) = kin_similarity_nearest(world, slot, grid);
    // Quorum sensing: local concentration of the passive "presence" signal
    // (see fields.rs) -- a real, continuous crowding read. Purely
    // sense-only: nothing here decides what a high reading should make an
    // individual do. If dispersing (or clustering, or breeding faster, or
    // getting more aggressive) when crowd pays off, evolution has the
    // sensory means to find it on its own.
    let quorum_local = crate::fields::Fields::sample(&world.fields.quorum, world.size, pos).tanh();
    let vis = vision(world, slot, grid);
    let mem = world.pixels.memory[world.individuals.pixel_offset[slot] as usize];

    // Territoriality: direction+distance back to this individual's own
    // home_pos (normalized/saturating so it's still informative far past
    // the nominal range, never blows up), and the local scent-mark
    // concentration -- see fields.rs's `territory` field doc comment for
    // why there's no per-owner distinction here.
    let home = world.individuals.home_pos[slot];
    let home_dx = ((home[0] - pos[0]) / crate::HOME_RANGE_NORM).tanh();
    let home_dy = ((home[1] - pos[1]) / crate::HOME_RANGE_NORM).tanh();
    let territory_local = crate::fields::Fields::sample(&world.fields.territory, world.size, pos).tanh();

    let mut out = [0f32; SENSE_DIM];
    let base = [fgx, fgy, phgx, phgy, blgx, blgy, acgx, acgy, ligx, ligy, fwd_food, fwd_light, energy_norm, size_norm, kin_sim, quorum_local];
    out[..16].copy_from_slice(&base);
    out[16..25].copy_from_slice(&vis);
    out[25] = world.day_light * 2.0 - 1.0; // rescaled to -1..1 to match the rest of the sense vector's rough range
    out[26] = home_dx;
    out[27] = home_dy;
    out[28] = territory_local;
    // Conspecific crowding -- the signal an individual would need in order
    // to evolve dispersal AWAY from its own kind (the behavioral half of
    // the Janzen-Connell story). Distinct from quorum_local above, which is
    // species-blind crowding: this one only counts individuals genetically
    // like this one, which is exactly what its specialist pathogens track.
    out[29] = (conspecific_density / crate::CONSPECIFIC_DENSITY_NORM).tanh();
    out[30..30 + MEM_DIM].copy_from_slice(&mem);
    (out, conspecific_density)
}

/// Removes pixel `local_idx` and every descendant, re-indexing survivors --
/// matches pixel_world.py's `remove_pixel`. Returns true if the individual
/// should die (root removed, or nothing left).
fn remove_pixel(world: &mut World, slot: usize, local_idx: u32) -> bool {
    let offset = world.individuals.pixel_offset[slot];
    let count = world.individuals.pixel_count[slot];
    let mut to_remove = std::collections::HashSet::new();
    to_remove.insert(local_idx);
    loop {
        let mut changed = false;
        for k in 0..count {
            let p = world.pixels.parent_idx[(offset + k) as usize];
            if p >= 0 && to_remove.contains(&(p as u32)) && !to_remove.contains(&k) {
                to_remove.insert(k);
                changed = true;
            }
        }
        if !changed { break; }
    }
    let root_parent = world.pixels.parent_idx[offset as usize];
    if (local_idx == 0 && root_parent < 0) || to_remove.len() as u32 >= count {
        return true;
    }

    let mut old_to_new = vec![-1i32; count as usize];
    let mut new_count = 0u32;
    for k in 0..count {
        if to_remove.contains(&k) { continue; }
        old_to_new[k as usize] = new_count as i32;
        new_count += 1;
    }
    let new_offset = world.pixels.allocate(new_count);
    let mut w = 0u32;
    for k in 0..count {
        if to_remove.contains(&k) { continue; }
        let p = world.pixels.parent_idx[(offset + k) as usize];
        let new_parent = if p < 0 { -1 } else { old_to_new[p as usize] };
        world.pixels.parent_idx[(new_offset + w) as usize] = new_parent;
        world.pixels.rest_angle[(new_offset + w) as usize] = world.pixels.rest_angle[(offset + k) as usize];
        world.pixels.flex[(new_offset + w) as usize] = world.pixels.flex[(offset + k) as usize];
        world.pixels.memory[(new_offset + w) as usize] = world.pixels.memory[(offset + k) as usize];
        // NOTE: this reindexing was previously missing storage entirely (a
        // pre-existing bug, not introduced here) -- every combat-driven
        // pixel loss silently reset survivors' storage/size/angle-limit/
        // health to whatever stale data happened to occupy the freshly
        // (re)allocated block, corrupting evolved anatomy on every fight.
        world.pixels.storage[(new_offset + w) as usize] = world.pixels.storage[(offset + k) as usize];
        world.pixels.part_type[(new_offset + w) as usize] = world.pixels.part_type[(offset + k) as usize];
        world.pixels.size[(new_offset + w) as usize] = world.pixels.size[(offset + k) as usize];
        world.pixels.min_angle[(new_offset + w) as usize] = world.pixels.min_angle[(offset + k) as usize];
        world.pixels.max_angle[(new_offset + w) as usize] = world.pixels.max_angle[(offset + k) as usize];
        world.pixels.health[(new_offset + w) as usize] = world.pixels.health[(offset + k) as usize];
        w += 1;
    }
    world.pixels.free(offset, count);
    world.individuals.pixel_offset[slot] = new_offset;
    world.individuals.pixel_count[slot] = new_count;
    crate::individuals::recompute_part_counts(&mut world.individuals, &world.pixels, slot);
    new_count == 0
}

/// The world-space position of the attacker's own pixel with the highest
/// evolved `storage` trait -- the closest thing to an anatomical "mouth" or
/// "belly" this engine has, and it's wherever evolution puts it, not a fixed
/// body part. Falls back to the root if every pixel has zero storage (the
/// common early case, before any lineage has evolved the trait at all).
/// `root` is an explicit override, not read live from `world.individuals.
/// root_pos[slot]` -- this matters when a chain of captures exists (an
/// attacker that is ITSELF someone else's captive). The attachment loop
/// processes every attacker in slot order each tick; without a fixed
/// snapshot, an attacker further along in a chain would compute its own
/// anchor using a root position some EARLIER individual in the SAME tick's
/// loop had already overwritten, letting a teleport cascade transitively
/// through the whole chain in one tick instead of propagating gradually,
/// tick by tick, like everything else in this engine. Symptom this fixed:
/// creatures suddenly "rushing" to some far-off place and dying.
fn storage_anchor_position(world: &World, slot: usize, root: [f32; 2]) -> [f32; 2] {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let mut best_idx = 0usize;
    let mut best_val = -1.0f32;
    for k in 0..count {
        let s = world.pixels.storage[offset + k];
        if s > best_val {
            best_val = s;
            best_idx = k;
        }
    }
    if best_val <= 0.0 {
        return root;
    }
    let positions = world_positions_at(world, slot, world.sim_time, root);
    positions[best_idx]
}

fn kill(world: &mut World, slot: usize) {
    if !world.individuals.alive[slot] { return; }
    let t = world.sim_time;
    let positions = world_positions(world, slot, t);
    let root = world.individuals.root_pos[slot];
    let local_shape: Vec<[f32; 2]> = positions.iter().map(|p| [p[0] - root[0], p[1] - root[1]]).collect();
    let count = world.individuals.pixel_count[slot];
    world.corpses.push(Corpse {
        root_pos: root,
        local_shape,
        color: world.individuals.color[slot],
        energy: crate::CORPSE_ENERGY_PER_PIXEL * count as f32,
    });
    world.individuals.free_slot(slot);
    world.deaths += 1;
}

fn update_weather(world: &mut World) {
    if world.weather == Weather::None {
        if world.rng.random::<f64>() < crate::WEATHER_TRIGGER_CHANCE {
            let choice = world.rng.random_range(0..3);
            world.weather = match choice { 0 => Weather::FoodBloom, 1 => Weather::ColdSnap, _ => Weather::FertileSurge };
            world.weather_ticks_left = world.rng.random_range(200..500);
            match world.weather {
                Weather::FoodBloom => { world.food_regrow_multiplier = 3.0; }
                Weather::ColdSnap => { world.metabolism_multiplier = 1.7; }
                Weather::FertileSurge => { world.maturity_multiplier = 0.35; }
                Weather::None => {}
            }
        }
    } else {
        world.weather_ticks_left -= 1;
        if world.weather_ticks_left <= 0 {
            world.weather = Weather::None;
            world.food_regrow_multiplier = 1.0;
            world.metabolism_multiplier = 1.0;
            world.maturity_multiplier = 1.0;
        }
    }
}

/// Was previously satisfied by ANY nearby individual at all -- not
/// checking sex, maturity, or eligibility. That's a real, mechanical bug,
/// not just an imprecise name: it meant simply sitting in a crowd (which
/// gravity + zero cost for staying still already produces for free) was
/// indistinguishable from actually finding a mate. Combined with active
/// swimming's real energy cost (MOVE_COST) and rarely putting a mobile
/// individual within MATE_RADIUS of anyone at all, this made passive
/// clustering strictly more reproductively successful than exploring --
/// exactly backwards from what mobility should buy an individual.
fn has_nearby_mate(world: &World, slot: usize, grid: &SpatialGrid) -> bool {
    let pos = world.individuals.root_pos[slot];
    let my_female = world.individuals.female[slot];
    // MUST be nearby_radius, not nearby: the latter only ever searches one
    // cell in each direction (~4 units), so the `dist < MATE_RADIUS` test
    // below could never actually bind and the effective mate-search radius
    // was about a fifth of the intended 20 -- a twenty-fivefold shortfall in
    // search AREA. This is the identical bug that was found and fixed in
    // vision earlier; has_nearby_mate was simply never updated with it. It
    // made mates far scarcer than designed, which is a large part of why
    // thinned-out populations slid into the Allee trap and went extinct
    // instead of recovering.
    for other in grid.nearby_radius(pos, crate::MATE_RADIUS) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        if world.individuals.female[other] == my_female { continue; }
        let other_mature = world.individuals.age[other] as f32 >= crate::MATURITY_AGE * world.maturity_multiplier;
        if !other_mature { continue; }
        let other_recovered = !world.individuals.female[other] || world.individuals.ticks_since_reproduced[other] >= crate::FEMALE_REPRODUCTION_COOLDOWN;
        if !other_recovered { continue; }
        if dist(world.individuals.root_pos[other], pos) < crate::MATE_RADIUS { return true; }
    }
    false
}

/// Test-only helper behind `debug_conspecific_densities`: rebuilds the
/// spatial grid and reports each alive individual's kin-weighted
/// conspecific density, in the same order individuals_state() emits.
pub fn conspecific_densities(world: &World) -> Vec<f32> {
    let alive_slots: Vec<usize> = (0..world.individuals.len())
        .filter(|&s| world.individuals.alive[s])
        .collect();
    let grid = SpatialGrid::build(alive_slots.iter().map(|&s| (s as u32, world.individuals.root_pos[s])));
    alive_slots
        .iter()
        .map(|&slot| kin_similarity_nearest(world, slot, &grid).1)
        .collect()
}

pub fn tick(world: &mut World) {
    let mut timings: Vec<(&'static str, f64)> = Vec::new();
    macro_rules! timed {
        ($label:expr, $body:expr) => {{
            let t0 = std::time::Instant::now();
            let result = $body;
            timings.push(($label, t0.elapsed().as_secs_f64() * 1000.0));
            result
        }};
    }

    world.tick_count += 1;
    world.sim_time += world.dt;
    world.day_light = (world.sim_time * std::f32::consts::TAU / crate::DAY_LENGTH).sin() * 0.5 + 0.5;
    timed!("weather", update_weather(world));

    let n = world.individuals.len();
    let alive_slots: Vec<usize> = timed!("collect_alive", (0..n).filter(|&s| world.individuals.alive[s]).collect());
    let grid = timed!("spatial_grid", SpatialGrid::build(alive_slots.iter().map(|&s| (s as u32, world.individuals.root_pos[s]))));

    // Cache FK+velocity for every alive individual once (own position may go
    // stale mid-tick if sliced by an earlier individual -- re-verified via
    // pixel_count before use, exactly like the Python/Rust-kernel version).
    let mut pos_cache: Vec<Option<Vec<[f32; 2]>>> = vec![None; n];
    let mut vel_cache: Vec<Option<Vec<[f32; 2]>>> = vec![None; n];
    let t0 = std::time::Instant::now();
    let fk_results: Vec<(usize, Vec<[f32; 2]>, Vec<[f32; 2]>)> = alive_slots
        .par_iter()
        .map(|&slot| {
            let pos = world_positions(world, slot, world.sim_time);
            let vel = pixel_velocities(world, slot, world.sim_time, 0.02);
            (slot, pos, vel)
        })
        .collect();
    for (slot, pos, vel) in fk_results {
        pos_cache[slot] = Some(pos);
        vel_cache[slot] = Some(vel);
    }
    timings.push(("fk_cache", t0.elapsed().as_secs_f64() * 1000.0));

    // Attachment: a stuck attacker drags + gradually eats its target.
    // Snapshotted BEFORE this loop mutates any root_pos: see
    // storage_anchor_position's doc comment for why a live read here would
    // let a capture-chain teleport cascade transitively in one tick.
    let root_snapshot: Vec<[f32; 2]> = world.individuals.root_pos.clone();
    let t0 = std::time::Instant::now();
    for &slot in &alive_slots {
        if world.individuals.attached_to[slot] < 0 { continue; }
        let target = match world.individuals.resolve_attached_target(slot) {
            Some(t) => t,
            // The target no longer resolves to a living individual -- it
            // died via some OTHER path (a third party's hit, starvation,
            // an over-cap cull...) since this attacker last checked. Give
            // up the chase instead of leaving a stale id sitting here: a
            // slot index would silently start pointing at whatever
            // unrelated newborn gets that freed slot next; an id just
            // stops resolving, which this correctly treats as "gone".
            None => {
                world.individuals.attached_to[slot] = -1;
                continue;
            }
        };
        let jitter = [normal(&mut world.rng, 0.0, 0.05), normal(&mut world.rng, 0.0, 0.05)];
        // Prey is held at whichever of the attacker's OWN pixels has the
        // highest evolved storage trait -- not always the root/head. A body
        // that concentrates storage in one place (a real evolved belly) or
        // grows it far out on a limb (an evolved "throat") gets to actually
        // hold prey there instead of it always symbolically snapping to the
        // head; nothing here decides that's good or picks where it forms.
        let anchor_pos = storage_anchor_position(world, slot, root_snapshot[slot]);
        world.individuals.root_pos[target] = [
            (anchor_pos[0] + jitter[0]).clamp(0.0, world.size as f32 - 1.0),
            (anchor_pos[1] + jitter[1]).clamp(0.0, world.size as f32 - 1.0),
        ];
        let drain = world.individuals.energy[target].max(0.0).min(0.4);
        world.individuals.energy[target] -= drain;
        world.individuals.energy[slot] += drain;
        // A captured target is EXCLUDED from the main `deciding` loop's own
        // energy<=0 death check whenever it happens to also be someone
        // else's attacker (attached_to[target] >= 0 keeps it out of
        // `deciding` entirely), and even when it isn't, this loop runs
        // BEFORE that check on the same tick. Previously this only killed
        // the target once its OWN pixel_count was already down to 1 -- but
        // nothing in this "hold and drain energy" path ever removes pixels,
        // so a multi-pixel target's energy could sit at/below zero forever
        // with no path to death: a pair permanently frozen together,
        // reported as individuals "fighting forever, stuck in place".
        // Draining to empty is death regardless of remaining body size.
        let mut released = false;
        if world.individuals.energy[target] <= 0.0 {
            kill(world, target);
            world.individuals.attached_to[slot] = -1;
            released = true;
        }
        // The flat 0.4/tick energy trickle above only bounds capture
        // duration for LOW-energy prey. An individual that has been
        // accumulating energy for a long lifetime (hundreds of units,
        // observed directly during earlier debugging) would take hundreds
        // of ticks to drain that way -- from the outside that reads as
        // exactly the same "locked in place, wiggling, no outcome" freeze,
        // just slower to notice. Real predation should tear off flesh, not
        // siphon an energy meter that's decoupled from the body holding it:
        // each tick, with a chance scaled by the attacker's own evolved
        // bite_force (not a flat constant), one of the target's pixels is
        // bitten off outright via the same remove_pixel path combat uses.
        // This bounds worst-case capture time by the target's pixel COUNT
        // (fixed at birth, small) instead of its energy total, and ties
        // consumption speed to a trait evolution actually shapes.
        // Prey struggles. Without this, any grip held until one party died,
        // which is how a quarter of the population ended up permanently
        // immobilised: a small attacker could lock a much larger victim
        // forever simply by touching it first. Escape odds rise with the
        // victim's size advantage, so a big animal shrugs off something that
        // grabbed above its weight while genuinely smaller prey rarely gets
        // away. This is also what stops capture from being a terminal state
        // for the world's dynamism.
        if !released {
            let attacker_size = body_size_sum(world, slot) * world.individuals.size_scale[slot];
            let victim_size = body_size_sum(world, target) * world.individuals.size_scale[target];
            let advantage = victim_size / attacker_size.max(0.01);
            let escape = (crate::STRUGGLE_ESCAPE_BASE * advantage / grip_multiplier(world, slot))
                .min(crate::STRUGGLE_ESCAPE_MAX);
            if world.rng.random::<f32>() < escape {
                world.individuals.attached_to[slot] = -1;
                released = true;
            }
        }
        if !released {
            let bite_force = world.individuals.bite_force[slot] * bite_multiplier(world, slot);
            // Consumption scales with how outmatched the prey is. A flat rate
            // meant a large predator spent just as long working through a
            // tiny victim as a huge one, so it could never take several small
            // meals in succession -- it was locked to whatever it grabbed
            // first. Now a big animal strips something much smaller than
            // itself in a few ticks and is free to hunt again, while an
            // evenly-matched struggle stays a real, drawn-out contest.
            let attacker_size = body_size_sum(world, slot) * world.individuals.size_scale[slot];
            let victim_size = body_size_sum(world, target) * world.individuals.size_scale[target];
            let dominance = (attacker_size / victim_size.max(0.01)).clamp(1.0, crate::CHEW_DOMINANCE_MAX);
            let chew_chance = (crate::CAPTURE_CHEW_CHANCE_BASE * bite_force * dominance)
                .clamp(0.0, crate::CAPTURE_CHEW_CHANCE_MAX);
            if world.rng.random::<f32>() < chew_chance {
                let target_count = world.individuals.pixel_count[target];
                if target_count > 0 {
                    let victim = world.rng.random_range(0..target_count);
                    let (hx, hy) = grid_xy(world, world.individuals.root_pos[target]);
                    let idx = (hx * world.size + hy) as usize;
                    let died = remove_pixel(world, target, victim);
                    world.individuals.energy[slot] += crate::CORPSE_ENERGY_PER_PIXEL * digestion_multiplier(world, slot);
                    world.fields.blood[idx] += crate::BLOOD_EMIT_ON_HIT;
                    if died {
                        kill(world, target);
                        world.individuals.attached_to[slot] = -1;
                    }
                }
            }
        }
    }
    timings.push(("attachment", t0.elapsed().as_secs_f64() * 1000.0));

    let attached_targets: std::collections::HashSet<usize> = alive_slots.iter()
        .filter_map(|&s| world.individuals.resolve_attached_target(s))
        .collect();

    // Everyone alive gets to think, INCLUDING an individual currently holding
    // prey. Excluding attackers here meant a predator that latched on stopped
    // running its brain entirely until it finished chewing -- it went inert,
    // could not steer, and could not move on to another meal. Measured, a
    // quarter of the population was held captive at any moment and their
    // captors were frozen alongside them, so roughly half the world was doing
    // nothing on any given tick. That is most of why the simulation looked
    // static and why "big creatures eating several small ones in a row" was
    // impossible. A captive still can't move (see `is_captured` below), which
    // is the part that should genuinely be immobilising.
    let deciding: Vec<usize> = alive_slots.iter().cloned()
        .filter(|&s| world.individuals.alive[s])
        .collect();

    // One O(n) pass for the Red Queen pressure below: how much of the
    // whole living population each lineage (keyed by color, the same
    // identity the species table and chronicle already use) currently
    // makes up. Cheap next to the per-individual work in this tick, and it
    // has to be global -- that's the entire point, see the pressure
    // computation in the sequential loop.
    let mut lineage_counts: std::collections::HashMap<[u8; 3], u32> = std::collections::HashMap::new();
    for &slot in &alive_slots {
        *lineage_counts.entry(world.individuals.color[slot]).or_insert(0) += 1;
    }
    let total_alive = alive_slots.len().max(1) as f32;
    let lineage_share: Vec<f32> = deciding
        .iter()
        .map(|&slot| {
            *lineage_counts.get(&world.individuals.color[slot]).unwrap_or(&0) as f32 / total_alive
        })
        .collect();

    // The expensive part of this loop -- sense, brain forward pass, thrust,
    // and contact-force -- is a pure function of tick-start state (reads
    // world/pos_cache/vel_cache/grid, mutates nothing, never touches the
    // RNG), so it's computed in parallel across all individuals with rayon
    // first. What's left (movement integration, reproduction, collision,
    // death) has real cross-individual mutation and stays sequential --
    // that's the actual reason this couldn't just be one big par_iter.
    let t0 = std::time::Instant::now();
    let mut pre: Vec<([f32; ACT_DIM], [f32; 2], [f32; 2], Option<ExperienceRow>, f32)> = deciding
        .par_iter()
        .map(|&slot| {
            let (s, conspecific_density) = sense(world, slot, &grid);
            let d: [f32; ACT_DIM] = world.individuals.decide(slot, &s);
            let cached_ok = pos_cache[slot].as_ref().map_or(false, |p| p.len() == world.individuals.pixel_count[slot] as usize);
            let (ind_pos, ind_vel) = if cached_ok {
                (pos_cache[slot].clone().unwrap(), vel_cache[slot].clone().unwrap())
            } else {
                (world_positions(world, slot, world.sim_time), pixel_velocities(world, slot, world.sim_time, 0.02))
            };
            let thrust = fluid_thrust_force(world, slot, &ind_pos, &ind_vel);
            let contact = contact_force(world, slot, &ind_pos, &grid, &pos_cache);
            // Deterministic sampling (no RNG call here -- this closure runs
            // in parallel and must stay a pure function of tick-start
            // state, per the comment above): a modulo on slot+tick spreads
            // coverage across the whole population over time instead of
            // always logging the same low-slot individuals.
            let sample = if (slot as u64 + world.tick_count) % crate::EXPERIENCE_SAMPLE_STRIDE == 0 {
                Some(ExperienceRow {
                    id: world.individuals.id[slot],
                    tick: world.tick_count,
                    sense: s.to_vec(),
                    action: d.to_vec(),
                    energy: world.individuals.energy[slot],
                })
            } else {
                None
            };
            // Raw density carried through rather than re-scanning
            // neighbors in the sequential phase where the drain is applied.
            (d, thrust, contact, sample, conspecific_density)
        })
        .collect();
    timings.push(("decide_parallel", t0.elapsed().as_secs_f64() * 1000.0));

    for row in pre.iter_mut().filter_map(|t| t.3.take()) {
        if world.experience_log.len() >= crate::EXPERIENCE_LOG_CAP { break; }
        world.experience_log.push(row);
    }

    let t0 = std::time::Instant::now();
    let mut t_reproduce = 0.0f64;
    let mut t_collision = 0.0f64;
    for (i, &slot) in deciding.iter().enumerate() {
        world.individuals.age[slot] += 1;
        let is_captured = attached_targets.contains(&slot);

        let (d, thrust, contact, _, _) = &pre[i];
        // Negative frequency-dependent selection (Red Queen / rare-type
        // advantage): a specialist pathogen tracks whichever host is
        // ABUNDANT, so the commonest lineage pays the highest price and
        // rare ones are effectively refuges. Keyed on GLOBAL lineage share,
        // not local crowding -- measured twice, local conspecific density
        // simply cannot distinguish a 50-member lineage from a
        // 4000-member one here, because every lineage clumps equally
        // tightly (offspring are born adjacent to their parent, so there
        // is no dispersal gradient of the kind real Janzen-Connell needs).
        // Below THRESHOLD share there is no pressure at all; above it,
        // damage grows with the SQUARE of the excess share.
        let share = lineage_share[i];
        let excess = (share - crate::PATHOGEN_SHARE_THRESHOLD).max(0.0);
        let pathogen_pressure =
            ((excess / crate::PATHOGEN_SHARE_SCALE).powi(2)).min(crate::PATHOGEN_PRESSURE_MAX);
        let (move_x, move_y, reproduce_urge, fight_urge, crawl_intent, acid_intent, light_intent) =
            (d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
        {
            let offset = world.individuals.pixel_offset[slot] as usize;
            world.pixels.memory[offset] = [d[7], d[8], d[9], d[10]];
        }

        if !is_captured {
            if move_x.abs() + move_y.abs() > 0.05 {
                let desired = move_y.atan2(move_x);
                let mut diff = (desired - world.individuals.heading[slot] + std::f32::consts::PI) % std::f32::consts::TAU - std::f32::consts::PI;
                diff = diff.clamp(-crate::TURN_RATE, crate::TURN_RATE);
                world.individuals.heading[slot] += diff;
            }

            let mass = (body_size_sum(world, slot) * world.individuals.size_scale[slot]).max(1.0);

            let root_y = world.individuals.root_pos[slot][1];
            let crawl_gate = crawl_intent.max(0.0);
            let crawl_force = if root_y < crate::CRAWL_FLOOR_THRESHOLD && world.individuals.crawl_affinity[slot] > 0.05 && crawl_gate > 0.01 {
                let h = world.individuals.heading[slot];
                let scale = crawl_gate * world.individuals.crawl_affinity[slot] * crate::CRAWL_THRUST_SCALE * mass;
                [scale * h.cos(), scale * h.sin()]
            } else { [0.0, 0.0] };

            // Sessile anchoring: resists thrust, gravity, and thermal drift,
            // but ONLY while actually resting on solid ground (rock or
            // sand) -- there's nothing to anchor to in open water, so a
            // body evolved for this gets no benefit at all until it
            // actually settles on the floor, giving a real reason to seek
            // and stay there rather than making anchoring a universal,
            // no-cost "just don't move" toggle.
            let on_solid_ground = resting_on_solid_ground(world, world.individuals.root_pos[slot]);
            let anchor = if on_solid_ground { world.individuals.anchor_strength[slot].min(1.0) } else { 0.0 };
            let mobility = 1.0 - anchor;

            // Flippers convert the same swimming effort into more thrust.
            let fin = thrust_multiplier(world, slot);
            let noise = [normal(&mut world.rng, 0.0, crate::THERMAL_NOISE), normal(&mut world.rng, 0.0, crate::THERMAL_NOISE)];
            let gravity_force = [0.0, -crate::GRAVITY * mass];
            let vel = world.individuals.velocity[slot];
            let accel = [
                (thrust[0] * mobility * fin + contact[0] + gravity_force[0] + crawl_force[0] * mobility) / mass - crate::LINEAR_DAMPING * vel[0] + noise[0] * mobility / mass.sqrt(),
                (thrust[1] * mobility * fin + contact[1] + gravity_force[1] * mobility + crawl_force[1] * mobility) / mass - crate::LINEAR_DAMPING * vel[1] + noise[1] * mobility / mass.sqrt(),
            ];
            let mut new_vel = [vel[0] + accel[0] * world.dt, vel[1] + accel[1] * world.dt];
            let speed = (new_vel[0] * new_vel[0] + new_vel[1] * new_vel[1]).sqrt();
            if speed > crate::MAX_SPEED {
                new_vel = [new_vel[0] * crate::MAX_SPEED / speed, new_vel[1] * crate::MAX_SPEED / speed];
            }
            let mut new_pos = [
                world.individuals.root_pos[slot][0] + new_vel[0] * world.dt,
                world.individuals.root_pos[slot][1] + new_vel[1] * world.dt,
            ];
            // Only the real floor (y=0, the sand seafloor) is an inelastic
            // surface -- resting on solid ground is physically real. The
            // other three boundaries are just where the world's coordinate
            // range ends, not physical surfaces; zeroing velocity there too
            // (the earlier fix) turned them into ONE-WAY absorbing traps:
            // thermal noise nudges a body into a wall, its outward velocity
            // is deleted, and it has to build fresh velocity from zero to
            // ever leave -- a ratchet that silently accumulates population
            // at edges over time, worst at corners where two walls compound.
            // Reflecting (bouncing) is the physically correct boundary for
            // those three.
            if new_pos[1] <= 0.0 && new_vel[1] < 0.0 {
                new_vel[1] = 0.0; // the floor: inelastic
            } else if new_pos[1] >= world.size as f32 - 1.0 && new_vel[1] > 0.0 {
                new_vel[1] = -new_vel[1] * 0.5; // the "ceiling": reflect
            }
            if new_pos[0] <= 0.0 && new_vel[0] < 0.0 {
                new_vel[0] = -new_vel[0] * 0.5; // side walls: reflect
            } else if new_pos[0] >= world.size as f32 - 1.0 && new_vel[0] > 0.0 {
                new_vel[0] = -new_vel[0] * 0.5;
            }
            new_pos[0] = new_pos[0].clamp(0.0, world.size as f32 - 1.0);
            new_pos[1] = new_pos[1].clamp(0.0, world.size as f32 - 1.0);

            // The sand seafloor is a real solid surface unless an
            // individual has evolved enough dig_strength to penetrate it --
            // resting on top of it, not sinking through it just because
            // gravity keeps pulling ("falling through the sand"). It's a
            // flat, full-world-width horizontal band (see terrain.rs), so
            // checking whether a body's lowest point dips below its top
            // edge is exact, and cheaper than a per-cell lookup per pixel.
            // This runs BEFORE the rock check below, not after: rock
            // clusters can overlap the sand band (see terrain.rs's
            // "on_floor" clusters), so pushing a body up to the sand
            // surface can land it inside rock at that same height, and that
            // needs to be caught, not just the original, pre-correction
            // candidate position.
            let old_root = world.individuals.root_pos[slot];
            // MUST match terrain.rs's generation exactly (round-trip
            // through the same u32 truncation), not just approximate it as
            // a float -- found via direct testing: a body corrected to
            // rest at the float value 14.4 gets truncated to grid cell 14
            // by every terrain lookup afterward, but terrain generation's
            // OWN integer floor_height (also 14.4 truncated to 14, i.e.
            // rows 0..13) means row 14 was never actually Sand at all --
            // anything "resting on the sand surface" was silently standing
            // on plain open water by the terrain grid's own classification,
            // which made anchor_strength's on-solid-ground check (and
            // anything else keying off terrain.at() at the resting height)
            // silently never fire.
            let floor_height = ((world.size as f32 * crate::terrain::SAND_FLOOR_FRACTION).max(4.0) as u32) as f32;
            let dig_power = world.individuals.dig_strength[slot];
            if dig_power < crate::SAND_DIG_THRESHOLD {
                let candidate_shape = world_positions_at(world, slot, world.sim_time, new_pos);
                let lowest_y = candidate_shape.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
                if lowest_y < floor_height {
                    let (rx, _) = grid_xy(world, new_pos);
                    let surface = sand_surface_height(world, rx, floor_height);
                    new_pos[1] += surface - lowest_y;
                    if new_vel[1] < 0.0 { new_vel[1] = 0.0; }
                }
            } else if new_pos[1] < floor_height {
                new_vel[0] *= (1.0 - crate::SAND_EXTRA_DAMPING * world.dt).max(0.0);
                new_vel[1] *= (1.0 - crate::SAND_EXTRA_DAMPING * world.dt).max(0.0);
                world.individuals.energy[slot] -= crate::SAND_DIG_COST;
            }

            // Real per-pixel terrain collision, not just the root. Computed
            // EXACTLY (not approximated by translating a cached shape): a
            // heading turn earlier in this same tick rotates the body, and
            // a pure-translation approximation would miss a limb swinging
            // into a rock wall via that rotation alone, with the root never
            // getting anywhere near it. Runs LAST, against the final
            // (post-sand-correction) candidate position, so it also catches
            // a sand-surface correction that happened to land in rock.
            let current_shape = world_positions_at(world, slot, world.sim_time, old_root);
            let candidate_shape = world_positions_at(world, slot, world.sim_time, new_pos);
            let currently_in_rock = current_shape.iter().any(|&p| {
                let (cx, cy) = grid_xy(world, p);
                world.terrain.at(cx, cy) == TerrainKind::Rock
            });
            let hits_rock = candidate_shape.iter().any(|&p| {
                let (cx, cy) = grid_xy(world, p);
                world.terrain.at(cx, cy) == TerrainKind::Rock
            });
            // If already inside rock, never permanently freeze there --
            // always allow this tick's move to proceed so there's a real
            // way out. This should be rare after growth/reproduction are
            // terrain-checked, but must never be a dead end if it happens.
            if hits_rock && !currently_in_rock {
                new_pos = old_root;
                new_vel = [0.0, 0.0];
            }

            world.individuals.velocity[slot] = new_vel;
            world.individuals.root_pos[slot] = new_pos;
            let speed_final = (new_vel[0] * new_vel[0] + new_vel[1] * new_vel[1]).sqrt();
            world.individuals.energy[slot] -= crate::MOVE_COST * speed_final;
        }

        if world.individuals.alive[slot] && !is_captured {
            let (x, y) = grid_xy(world, world.individuals.root_pos[slot]);
            let idx = (x * world.size + y) as usize;
            let eaten = world.fields.food[idx].min(crate::EAT_RATE * 0.1);
            world.fields.food[idx] -= eaten;
            world.individuals.energy[slot] += eaten * 5.0 * digestion_multiplier(world, slot);
            world.individuals.ticks_since_fed[slot] += 1;
            if eaten > 0.001 {
                world.individuals.ticks_since_fed[slot] = 0;
            }
            // Recomputed here (cheap: one terrain lookup) rather than
            // threaded through from the movement block above, since this is
            // a separate `if` scope -- a real sessile organism has lower
            // upkeep than an active swimmer, the same real reason anemones
            // and corals get by on far less than free-swimming animals of
            // similar mass.
            let anchored_here = resting_on_solid_ground(world, world.individuals.root_pos[slot])
                && world.individuals.anchor_strength[slot] > 0.05;
            let anchor_discount = if anchored_here { world.individuals.anchor_strength[slot].min(1.0) * crate::ANCHOR_METABOLISM_DISCOUNT } else { 0.0 };
            let metabolism = crate::BASE_METABOLISM
                + crate::PER_PIXEL_METABOLISM * metabolic_part_load(world, slot) * world.individuals.size_scale[slot] * (1.0 - anchor_discount)
                // Immunity isn't free: keeping resistance up costs upkeep
                // every tick, whether or not any pathogen is actually
                // around. Without this the trait would simply ratchet to
                // its cap in every lineage and the whole mechanism below
                // would quietly stop mattering.
                + crate::DISEASE_RESISTANCE_METABOLIC_COST * world.individuals.disease_resistance[slot];
            world.individuals.energy[slot] -= metabolism * world.metabolism_multiplier;

            // Janzen-Connell in one line: damage scales with how densely
            // this individual's OWN KIND is packed around it (not generic
            // crowding), so a lineage that monopolizes a region pays an
            // escalating price for exactly that success, and locally-rare
            // lineages get a survival edge. That's the documented
            // real-world mechanism that keeps diverse communities from
            // collapsing into whoever competes best -- see the trait's doc
            // comment in individuals.rs. Nothing here targets any specific
            // lineage or caps anyone's population directly.
            if pathogen_pressure > 0.01 {
                let resistance = 1.0 / (1.0 + world.individuals.disease_resistance[slot]);
                world.individuals.energy[slot] -= world.pathogen_damage_rate * pathogen_pressure * resistance;
            }
            world.fields.pheromone[idx] += crate::PHEROMONE_EMIT_BASE * world.individuals.pheromone_emission[slot] * world.dt;
            world.fields.territory[idx] += crate::TERRITORY_EMIT_BASE * world.individuals.territoriality[slot] * world.dt;
            // Constitutive, not evolved: every living body emits this just
            // by being here, unlike every other field above which is an
            // evolved trait's deliberate output.
            world.fields.quorum[idx] += crate::QUORUM_EMIT_BASE * world.dt;
            if world.individuals.acid_secretion[slot] > 0.01 && acid_intent > 0.0 {
                world.fields.acid[idx] += crate::ACID_EMIT_BASE * world.individuals.acid_secretion[slot] * acid_intent * world.dt;
            }
            if world.individuals.light_emission[slot] > 0.01 && light_intent > 0.0 {
                world.fields.light[idx] += crate::LIGHT_EMIT_BASE * world.individuals.light_emission[slot] * light_intent * world.dt;
            }
            let acid_here = world.fields.acid[idx];
            if acid_here > 0.05 {
                world.individuals.energy[slot] -= crate::ACID_DAMAGE_RATE * acid_here;
            }

            // Development: a body genuinely grows over its own lifetime now,
            // not just once at birth (as a bigger CHILD) -- but growing means
            // uniformly INFLATING the fixed body plan it was born with
            // (size_scale), never sprouting new parts. Growth costs real
            // energy and slows logarithmically the more it's already grown
            // since birth -- fast juvenile growth, diminishing adult growth,
            // with no hardcoded final size.
            let grown_since_birth = world.individuals.size_scale[slot] - 1.0;
            if world.individuals.size_scale[slot] < crate::MAX_SIZE_SCALE && world.individuals.energy[slot] > crate::GROWTH_ENERGY_THRESHOLD {
                let growth_chance = crate::GROWTH_BASE_CHANCE / (1.0 + grown_since_birth * crate::GROWTH_SLOWDOWN / crate::GROWTH_SCALE_INCREMENT);
                let tentative_scale = (world.individuals.size_scale[slot] + crate::GROWTH_SCALE_INCREMENT).min(crate::MAX_SIZE_SCALE);
                // Inflation moves every pixel further from the root -- a
                // limb resting near a rock wall can grow INTO it purely by
                // getting bigger, with no translation involved at all. The
                // ordinary movement-time rock check (above, earlier this
                // same tick) can't catch that because it only fires on
                // translation. Checked here directly: if inflating would
                // put any pixel inside rock, this tick's growth is simply
                // skipped (energy isn't spent, scale doesn't change) --
                // pinned against a wall just pauses growth, it doesn't
                // teleport or freeze the individual.
                let inflated_shape = world_positions_at(world, slot, world.sim_time, world.individuals.root_pos[slot]);
                let scale_ratio = tentative_scale / world.individuals.size_scale[slot];
                let root = world.individuals.root_pos[slot];
                let would_hit_rock = inflated_shape.iter().any(|&p| {
                    let scaled = [root[0] + (p[0] - root[0]) * scale_ratio, root[1] + (p[1] - root[1]) * scale_ratio];
                    let (cx, cy) = grid_xy(world, scaled);
                    world.terrain.at(cx, cy) == TerrainKind::Rock
                });
                if world.rng.random::<f32>() < growth_chance && !would_hit_rock {
                    world.individuals.size_scale[slot] = tentative_scale;
                    world.individuals.energy[slot] -= crate::GROWTH_ENERGY_COST;
                }
            }

            world.individuals.ticks_since_reproduced[slot] = world.individuals.ticks_since_reproduced[slot].saturating_add(1);
            combat::regenerate(world, slot);

            let t_repro_0 = std::time::Instant::now();
            let recovery_ok = !world.individuals.female[slot]
                || world.individuals.ticks_since_reproduced[slot] >= crate::FEMALE_REPRODUCTION_COOLDOWN;
            // Building a child costs what the child actually IS. This was a
            // flat 8.0 regardless of body size, which meant a thirty-part
            // animal produced a thirty-one-part offspring for the same price
            // a two-part blob paid for a three-part one -- biomass conjured
            // from nothing, and no brake whatsoever on population. Charging
            // per part makes body size a real life-history decision: small
            // bodies breed cheaply and often, large ones invest heavily and
            // rarely, and the population limits itself through the energy
            // budget instead of slamming into an artificial cap.
            let offspring_parts = world.individuals.pixel_count[slot] as f32 + 1.0;
            let repro_cost = crate::REPRODUCE_BASE_COST
                + world.repro_cost_per_part * offspring_parts;
            // Must keep a survival buffer after paying, or reproducing would
            // be a reliable way to starve immediately afterwards.
            let repro_threshold = repro_cost + crate::REPRODUCE_ENERGY_BUFFER;
            if reproduce_urge > 0.3
                && world.individuals.energy[slot] > repro_threshold
                && world.individuals.age[slot] as f32 >= crate::MATURITY_AGE * world.maturity_multiplier
                && world.individuals.size_scale[slot] >= crate::ADULT_SIZE_SCALE
                && world.individuals.ticks_since_fed[slot] < crate::RECENT_FEED_WINDOW
                && recovery_ok
                && has_nearby_mate(world, slot, &grid)
            {
                world.individuals.energy[slot] -= repro_cost;
                world.individuals.ticks_since_reproduced[slot] = 0;
                let child = crate::individuals::reproduce(&mut world.individuals, &mut world.pixels, &mut world.rng, slot);
                // A child's root_pos is parent_pos + small random offset,
                // with no terrain awareness -- if that offset (or the
                // child's own body extending from it) lands inside rock,
                // the child (and every one of ITS children, since they'd
                // spawn near a parent that can never move) would be
                // permanently entombed there, endlessly reproducing in
                // place. Checked against the child's FULL body, not just
                // its root point -- a newborn can have several pixels
                // already (founders' extra growth, or an inherited body
                // plan from a long-lived parent). The parent's own position
                // is known-good (it's alive and moving), so fall back to it
                // if the offset missed.
                let child_shape = world_positions_at(world, child, world.sim_time, world.individuals.root_pos[child]);
                let child_hits_rock = child_shape.iter().any(|&p| {
                    let (cx, cy) = grid_xy(world, p);
                    world.terrain.at(cx, cy) == TerrainKind::Rock
                });
                if child_hits_rock {
                    world.individuals.root_pos[child] = world.individuals.root_pos[slot];
                }
                world.reproductions += 1;
                // Inclusive-fitness reward: `slot` just proved it survived to
                // reproduce -- its own parent (if still alive) gets credit
                // for that, an energy bonus on top of whatever it's already
                // doing. This is the actual selection pressure that can make
                // reduced aggression toward one's own recent offspring pay
                // off over generations, not just be possible.
                if world.individuals.parent_id[slot] >= 0 {
                    if let Some(&parent_slot) = world.individuals.id_to_slot.get(&(world.individuals.parent_id[slot] as u64)) {
                        if world.individuals.alive[parent_slot] {
                            world.individuals.energy[parent_slot] += crate::REPRODUCTION_SUCCESS_REWARD;
                        }
                    }
                }
            }
            t_reproduce += t_repro_0.elapsed().as_secs_f64() * 1000.0;

            // Real animals aren't reflexively violent -- they fight when
            // hungry, defending young, over territory, or competing for a
            // mate, not just because something is nearby. This used to add
            // a hardcoded density-based boost straight to fight_urge
            // (crowd_pressure) -- exactly the kind of indiscriminate,
            // context-blind aggression trigger that produced constant
            // opportunistic violence regardless of need. Removed outright:
            // local density is ALREADY a real sense input (quorum_local,
            // see sense()), so if density-driven aggression is ever
            // actually advantageous, evolution can still find it by
            // raising fight_urge output in response to that signal --  as
            // a learned response, not an engine-enforced one. What's left
            // to make aggression non-free: a genuine energy cost on every
            // attempt (below), so a brain that fights indiscriminately
            // bleeds energy for nothing, while one that's actually
            // learned to fight only when hungry (energy_norm is already
            // sensed) keeps more of what it eats.
            let t_coll_0 = std::time::Instant::now();
            if fight_urge > 0.5 && world.individuals.attached_to[slot] < 0 {
                // Only actually charge (and attempt) this if there's
                // someone else there at all -- otherwise an isolated
                // individual with a naturally high, constant fight_urge
                // (common from random NN init) pays a real energy cost
                // every single tick for lunging at empty water. That's not
                // "fighting isn't free" anymore, it's a tax on being alone,
                // and it hit exploring/isolated individuals hardest --
                // exactly backwards from wanting mobility to be viable.
                let anyone_nearby = grid.nearby(world.individuals.root_pos[slot]).iter()
                    .any(|&o| o as usize != slot && world.individuals.alive[o as usize]);
                if anyone_nearby {
                    world.individuals.energy[slot] -= crate::ATTACK_ENERGY_COST;
                    resolve_collision(world, slot, &pos_cache, &vel_cache, &grid);
                }
            }
            t_collision += t_coll_0.elapsed().as_secs_f64() * 1000.0;
        }

        if world.individuals.alive[slot] && world.individuals.energy[slot] <= 0.0 {
            kill(world, slot);
        }
    }

    timings.push(("decide_apply", t0.elapsed().as_secs_f64() * 1000.0));
    timings.push(("  of_which_reproduce", t_reproduce));
    timings.push(("  of_which_collision", t_collision));

    let t0 = std::time::Instant::now();
    scavenge_all(world, &deciding);
    timings.push(("scavenge", t0.elapsed().as_secs_f64() * 1000.0));

    let t0 = std::time::Instant::now();
    let alive_count = (0..world.individuals.len()).filter(|&s| world.individuals.alive[s]).count();
    if alive_count > world.pop_cap {
        let mut alive: Vec<usize> = (0..world.individuals.len()).filter(|&s| world.individuals.alive[s]).collect();
        alive.sort_by(|&a, &b| world.individuals.energy[b].partial_cmp(&world.individuals.energy[a]).unwrap());
        for &slot in alive.iter().skip(world.pop_cap) {
            world.individuals.free_slot(slot); // over-cap cull: no corpse, matches Python's cap enforcement
        }
    }
    timings.push(("pop_cap", t0.elapsed().as_secs_f64() * 1000.0));

    let t0 = std::time::Instant::now();
    for c in world.corpses.iter_mut() {
        if c.root_pos[1] > 0.0 { c.root_pos[1] = (c.root_pos[1] - crate::CORPSE_SINK_RATE).max(0.0); }
    }
    world.corpses.retain(|c| c.energy > 0.01);
    timings.push(("corpses", t0.elapsed().as_secs_f64() * 1000.0));

    let t0 = std::time::Instant::now();
    world.fields.step_food_regrow(world.food_regrow_rate * world.food_regrow_multiplier, world.food_cap);
    world.fields.step_diffusion(world.size);
    timings.push(("fields", t0.elapsed().as_secs_f64() * 1000.0));

    world.timings = timings;
}

fn contact_force(world: &World, slot: usize, ind_pos: &[[f32; 2]], grid: &SpatialGrid, pos_cache: &[Option<Vec<[f32; 2]>>]) -> [f32; 2] {
    let mut push = [0f32; 2];
    let my_pos = world.individuals.root_pos[slot];
    let my_size = ind_pos.len() as f32;
    for other in grid.nearby(my_pos) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        let other_size = world.individuals.pixel_count[other] as f32;
        if dist(world.individuals.root_pos[other], my_pos) > (my_size + other_size) * 0.5 + 2.0 { continue; }
        let other_pos = match &pos_cache[other] {
            Some(p) if p.len() == world.individuals.pixel_count[other] as usize => p,
            _ => continue,
        };
        for &p in ind_pos.iter() {
            let mut best_d = f32::MAX;
            let mut best = [0f32; 2];
            for &op in other_pos.iter() {
                let d = dist(p, op);
                if d < best_d { best_d = d; best = op; }
            }
            if best_d > 1e-6 && best_d < crate::COLLISION_RADIUS {
                let overlap = crate::COLLISION_RADIUS - best_d;
                push[0] += (p[0] - best[0]) / best_d * overlap * crate::COLLISION_STIFFNESS;
                push[1] += (p[1] - best[1]) / best_d * overlap * crate::COLLISION_STIFFNESS;
            }
        }
    }
    push
}

fn resolve_collision(world: &mut World, slot: usize, pos_cache: &[Option<Vec<[f32; 2]>>], vel_cache: &[Option<Vec<[f32; 2]>>], grid: &SpatialGrid) {
    // Reuse the tick-start FK cache instead of recomputing 3 fresh FK passes
    // (position + 2 for finite-difference velocity) on every single fight
    // attempt -- crowd-driven cannibalism makes this a very hot path at high
    // density. The cache is from BEFORE this tick's movement, so hit
    // detection is up to one MAX_SPEED*dt (~0.3 world units) behind the
    // attacker's true current position -- small relative to
    // COLLISION_RADIUS (1.2), and only stale at all if attacked-then-moved
    // in the same tick. Falls back to a fresh, exact computation whenever
    // the individual's own pixel count changed since tick start (already
    // sliced by someone else this tick), same guard used everywhere else.
    let cached_ok = pos_cache[slot].as_ref().map_or(false, |p| p.len() == world.individuals.pixel_count[slot] as usize);
    let (ind_pos, ind_vel) = if cached_ok {
        (pos_cache[slot].clone().unwrap(), vel_cache[slot].clone().unwrap())
    } else {
        (world_positions(world, slot, world.sim_time), pixel_velocities(world, slot, world.sim_time, 0.02))
    };
    let speeds: Vec<f32> = ind_vel.iter().map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).collect();
    let acid_secretion = world.individuals.acid_secretion[slot];
    let fastest = speeds.iter().cloned().fold(0.0f32, |a, b| a.max(b));
    let my_pos = world.individuals.root_pos[slot];
    let my_size = ind_pos.len() as f32;

    for other in grid.nearby(my_pos) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        // Camouflage: a target whose evolved color blends into its current
        // surroundings has a real chance of simply not being noticed this
        // attempt, independent of how strong the attacker is or how tough
        // the target is -- a color-matched target is never one-shot-proof,
        // it's just sometimes invisible.
        let detect_chance = 1.0 - combat::camouflage_effectiveness(world, other) * crate::CAMOUFLAGE_DETECTION_PENALTY;
        if world.rng.random::<f32>() > detect_chance { continue; }
        // Aposematism: noticed doesn't mean pursued -- a conspicuous color
        // backed by real toughness/venom makes an attacker hesitate.
        if world.rng.random::<f32>() < combat::aposematism_deterrence(world, other) { continue; }
        let armor = combat::effective_toughness(world, other);
        let strongest_power = combat::attacker_power(world, slot, fastest);
        // A landed bite can still poison even if it can never out-muscle
        // this target's armor -- only skip entirely when NEITHER mechanism
        // could ever matter, so venom stays a real alternative to brute
        // force rather than being gated behind the same threshold as it.
        if strongest_power <= armor && acid_secretion <= 0.01 { continue; }
        let other_size = world.individuals.pixel_count[other] as f32;
        if dist(world.individuals.root_pos[other], my_pos) > (my_size + other_size) * 0.5 + 2.0 { continue; }
        let other_pos = match &pos_cache[other] {
            Some(p) if p.len() == world.individuals.pixel_count[other] as usize => p.clone(),
            _ => continue,
        };
        let _ = &vel_cache; // velocities of `other` aren't needed for the attacker's own speed check
        let other_offset = world.individuals.pixel_offset[other] as usize;

        let mut hit: Option<(usize, f32, f32)> = None; // (local_idx, distance, this attacking pixel's power)
        for (i, &p) in ind_pos.iter().enumerate() {
            let power = combat::attacker_power(world, slot, speeds[i]);
            if power <= armor && acid_secretion <= 0.01 { continue; }
            let mut best_d = f32::MAX;
            let mut best_j = 0usize;
            for (j, &op) in other_pos.iter().enumerate() {
                let d = dist(p, op);
                if d < best_d { best_d = d; best_j = j; }
            }
            if best_d < combat::hit_radius(world, other_offset, best_j) {
                hit = Some((best_j, best_d, power));
                break;
            }
        }
        if let Some((j, _, power)) = hit {
            let hit_pos = other_pos[j];
            let (hx, hy) = grid_xy(world, hit_pos);
            let idx = (hx * world.size + hy) as usize;
            let is_head = j == 0; // the root is always local pixel 0 by construction -- see world_positions
            let damage = combat::kinetic_damage(power, armor, is_head);
            world.pixels.health[other_offset + j] -= damage;
            combat::inject_venom(world, slot, hx, hy);
            let pixel_severed = world.pixels.health[other_offset + j] <= 0.0;
            world.fights += 1;
            if damage > 0.0 || pixel_severed {
                world.individuals.ticks_since_fed[slot] = 0; // a landed, damaging hit counts as successful predation
            }
            world.fields.blood[idx] += crate::BLOOD_EMIT_ON_HIT * (damage / crate::BASE_PIXEL_HEALTH).clamp(0.15, 1.0);
            if pixel_severed {
                let died = remove_pixel(world, other, j as u32);
                if died {
                    kill(world, other);
                    world.fields.blood[idx] += crate::BLOOD_EMIT_ON_DEATH;
                }
            }
            if world.individuals.alive[other] && world.rng.random::<f32>() < world.individuals.stickiness[slot] && world.individuals.attached_to[slot] < 0 {
                world.individuals.attached_to[slot] = world.individuals.id[other] as i64;
                let bonus = world.individuals.energy[other].min(crate::CAPTURE_BONUS);
                world.individuals.energy[slot] += bonus;
                world.individuals.energy[other] -= bonus;
            }
            return;
        }
    }
}

fn scavenge_all(world: &mut World, deciding: &[usize]) {
    if world.corpses.is_empty() { return; }
    for &slot in deciding {
        if !world.individuals.alive[slot] { continue; }
        let pos = world.individuals.root_pos[slot];
        for c in world.corpses.iter_mut() {
            if c.energy <= 0.0 { continue; }
            if dist(c.root_pos, pos) < crate::CORPSE_EAT_RADIUS {
                let bite = c.energy.min(crate::CORPSE_EAT_RATE);
                c.energy -= bite;
                world.individuals.energy[slot] += bite;
                world.individuals.ticks_since_fed[slot] = 0;
                world.scavenged += 1;
                break;
            }
        }
    }
}
