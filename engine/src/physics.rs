//! Forward kinematics, thrust, sensing, and the full per-tick orchestration
//! -- the Rust-native replacement for pixel_world.py's `World.tick()`. Every
//! step that used to be a Python loop over `Individual` objects is now a
//! loop over SoA component arrays with no interpreter overhead.
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::combat;
use crate::individuals::{ACT_DIM, MEM_DIM, MEMORY_OUT_IDX, SENSE_DIM};
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
    // Subtracting the body's own axis makes `heading` the direction the animal
    // actually points, rather than an arbitrary rotation of whatever shape it
    // happened to grow into. Without it a creature whose parts grew off to one
    // side moved sideways relative to where it was nominally facing, by a
    // different offset per individual -- which pools to look like pure noise,
    // and is why heading-vs-movement measured ~79 degrees at every speed.
    let heading = world.individuals.heading[slot] - world.individuals.axis_offset[slot];
    // Undulation amplitude is the evolved body rhythm scaled by how hard the
    // brain has decided to swim right now -- genetics sets the stroke, the
    // mind sets the effort.
    let amp = world.individuals.bend_amplitude[slot] * world.individuals.swim_gain[slot];
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

    // Depth along the parent chain, i.e. how far each part is from the head.
    //
    // The bending wave used the part's ARRAY INDEX as its position in the
    // wave. In an unbranched chain index happens to equal distance from the
    // head, so that worked by accident -- but a branched body's index is
    // just the order parts were grown in, which has no relationship to where
    // they sit. The result was not a travelling wave at all: branches flapped
    // incoherently against each other, thrust never summed to a consistent
    // direction, and heading-vs-movement stayed near random even with all
    // noise removed. Phasing by DEPTH makes every branch undulate as a
    // function of its real distance from the head, the way an actual animal
    // does. Parents always precede their children in this array (growth
    // appends), so one forward pass suffices.
    let mut depth = vec![0f32; count];
    for k in 0..count {
        let parent = world.pixels.parent_idx[offset + k];
        depth[k] = if parent < 0 { 0.0 } else { depth[parent as usize] + 1.0 };
    }

    // Accumulated bending along the parent chain, kept separate from the
    // rest pose. See the angle assignment below for why the two must not be
    // conflated.
    let mut cumwave = vec![0f32; count];
    let mut angles = vec![0f32; count];
    let mut positions = vec![[0f32; 2]; count];
    for k in 0..count {
        let flex = world.pixels.flex[offset + k];
        // Undulation plus the constant curvature the brain is holding. A
        // curved body pushes water to one side, which is what produces the
        // torque that turns it -- the animal steers by SHAPING ITSELF, not by
        // having its orientation overwritten.
        let raw_wave = world.pixels.mirror_sign[offset + k] * flex * amp
            * (std::f32::consts::TAU * freq * t + phase + depth[k] * crate::BODY_WAVE_NUMBER).sin();
        // Joint angle limits: a real hinge constraint on how far THIS
        // joint's animated bend can deviate from its rest pose, heritable
        // per part (pixels.rs's min_angle/max_angle). Previously every
        // joint swung through the same unbounded range regardless of what
        // kind of part it was -- a stiff plate and a whip-like tail moved
        // identically except for amplitude. A narrow range reads as a
        // rigid/braced joint, a wide one as a loose/flexible one, and nothing
        // here decides which is good; it's just now possible to evolve.
        // The joint limit bounds how far this joint may OSCILLATE about its
        // rest pose. Deliberate steering was previously added before that
        // clamp, so a hard turn was simply clipped away: measured, turn rate
        // saturated at a curvature of ~0.5 and every larger command produced
        // exactly the same rotation. The brain had no authority beyond a
        // slight bend, while the body's own asymmetry span it faster than it
        // could correct. Posture is applied outside the oscillation clamp and
        // bounded separately, so a creature can genuinely throw its body into
        // a turn.
        let oscillation = raw_wave.clamp(world.pixels.min_angle[offset + k], world.pixels.max_angle[offset + k]);
        let posture = (world.individuals.turn_curvature[slot] * flex)
            .clamp(-crate::MAX_POSTURE_BEND, crate::MAX_POSTURE_BEND);
        let wave = oscillation + posture;
        let parent = world.pixels.parent_idx[offset + k];
        // A part's own evolved `size` stretches ITS segment specifically
        // (on top of the individual-wide size_scale inflation) -- a body
        // can evolve one big limb and several small ones, not just scale
        // uniformly everywhere.
        let seg_len = scale * world.pixels.size[offset + k];
        if parent < 0 {
            cumwave[k] = wave;
            angles[k] = heading + wave;
            positions[k] = root;
        } else {
            let p = parent as usize;
            // rest_angle is an ABSOLUTE direction, not an angle relative to
            // the parent. Growth builds bodies as a grid polyomino, placing
            // each part in a compass direction from its parent and refusing
            // to occupy a cell twice -- but this chain summed rest angles
            // cumulatively, so the body actually rendered was a different
            // shape from the one growth designed: chains curled in on
            // themselves, the overlap check became meaningless, and
            // undulation travelled along a knot instead of a body. Bending
            // still has to accumulate down the chain (that is what a
            // travelling wave IS), so the wave is summed separately and
            // added to the absolute rest direction.
            cumwave[k] = cumwave[p] + wave;
            angles[k] = heading + world.pixels.rest_angle[offset + k] + cumwave[k];
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

/// Net fluid force AND the torque it exerts about the body's root.
///
/// Torque is what turns the animal. Returning only the force meant rotation
/// had to come from somewhere else -- and it came from the brain assigning a
/// heading directly, which is why pointing and moving were unrelated.
pub fn fluid_thrust_torque(world: &World, slot: usize, pos: &[[f32; 2]], vel: &[[f32; 2]]) -> ([f32; 2], f32) {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = pos.len();
    let mut force = [0f32; 2];
    let mut torque = 0f32;
    let root = if count > 0 { pos[0] } else { [0.0, 0.0] };
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
        // Drag anisotropy is a property of the PART, not a global constant.
        // A fin is a paddle: it presents a broad face to the water when swept
        // sideways, which is exactly how a real animal converts body motion
        // into thrust. Making flippers merely multiply whole-body thrust made
        // them a stat rather than an organ; giving them their own
        // perpendicular drag means a finned body genuinely pushes more water
        // per stroke, and where the fins sit on the body matters.
        let perp = crate::DRAG_PERPENDICULAR
            * crate::PART_DRAG_PERP[world.pixels.part_type[offset + k] as usize];
        let fx = -(crate::DRAG_PARALLEL * vparx + perp * vperpx) * seg_len;
        let fy = -(crate::DRAG_PARALLEL * vpary + perp * vperpy) * seg_len;
        force[0] += fx;
        force[1] += fy;
        // r x F, about the root, for the 2D scalar torque.
        let rx = pos[k][0] - root[0];
        let ry = pos[k][1] - root[1];
        torque += rx * fy - ry * fx;
    }
    (force, torque)
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
/// How long a female must recover after giving birth, scaled by the size of
/// the body she is building. A flat cooldown meant a thirty-part animal
/// turned offspring around as fast as a two-part one, so large size carried
/// no reproductive penalty at all and there was nothing separating a fast-
/// breeding small strategy from a slow-investing large one. Gestation
/// growing with offspring size is the standard size-structured population
/// regulator, and it is what lets big animals be rare without being
/// artificially capped.
pub(crate) fn gestation_ticks(world: &World, slot: usize) -> u32 {
    let parts = world.individuals.pixel_count[slot] as f32;
    crate::FEMALE_REPRODUCTION_COOLDOWN + (crate::GESTATION_TICKS_PER_PART * parts) as u32
}

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
        let other_recovered = !world.individuals.female[other] || world.individuals.ticks_since_reproduced[other] >= gestation_ticks(world, other);
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
        world.pixels.symmetric[(new_offset + w) as usize] = world.pixels.symmetric[(offset + k) as usize];
        world.pixels.mirror_sign[(new_offset + w) as usize] = world.pixels.mirror_sign[(offset + k) as usize];
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
/// Returns (an eligible mate is in range, how many living neighbours are
/// within breeding distance). The crowd count comes free from the scan that
/// was already happening for mate-finding, and feeds the space requirement
/// in the reproduction gate.
fn mate_and_crowding(world: &World, slot: usize, grid: &SpatialGrid) -> (bool, u32) {
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
    let mut crowd = 0u32;
    let mut found_mate = false;
    for other in grid.nearby_radius(pos, crate::MATE_RADIUS) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        if dist(world.individuals.root_pos[other], pos) < crate::BREEDING_SPACE_RADIUS {
            crowd += 1;
        }
        if found_mate { continue; }
        if world.individuals.female[other] == my_female { continue; }
        let other_mature = world.individuals.age[other] as f32 >= crate::MATURITY_AGE * world.maturity_multiplier;
        if !other_mature { continue; }
        let other_recovered = !world.individuals.female[other] || world.individuals.ticks_since_reproduced[other] >= gestation_ticks(world, other);
        if !other_recovered { continue; }
        if dist(world.individuals.root_pos[other], pos) < crate::MATE_RADIUS { found_mate = true; }
    }
    (found_mate, crowd)
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
        let dragged = [
            (anchor_pos[0] + jitter[0]).clamp(0.0, world.size as f32 - 1.0),
            (anchor_pos[1] + jitter[1]).clamp(0.0, world.size as f32 - 1.0),
        ];
        // Dragging moved the prey by assignment, with no terrain check at
        // all -- so a predator hauled its victim straight through solid rock.
        // Measured, 13-24% of bodies had parts embedded in rock, and since
        // anything already inside rock is deliberately allowed to keep moving
        // (so being stuck is never permanent), those bodies then passed
        // through walls indefinitely. That silently defeated the whole point
        // of the reef: a passage is only a refuge if it cannot be dragged
        // through. If the drag would put the prey in rock, it simply stays
        // where it is; the grip holds, the geometry wins.
        let drag_hits_rock = world_positions_at(world, target, world.sim_time, dragged)
            .iter()
            .any(|&p| {
                let (cx, cy) = grid_xy(world, p);
                world.terrain.at(cx, cy) == TerrainKind::Rock
            });
        if !drag_hits_rock {
            world.individuals.root_pos[target] = dragged;
        }
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
            world.deaths_predation += 1;
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
            world.deaths_predation += 1;
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
    let mut pre: Vec<([f32; ACT_DIM], [f32; 2], [f32; 2], Option<ExperienceRow>, f32, f32)> = deciding
        .par_iter()
        .map(|&slot| {
            let (s, conspecific_density) = sense(world, slot, &grid);
            let d: [f32; ACT_DIM] = world.individuals.decide(slot, &s, &world.shared_enc_w, &world.shared_enc_b);
            let cached_ok = pos_cache[slot].as_ref().map_or(false, |p| p.len() == world.individuals.pixel_count[slot] as usize);
            let (ind_pos, ind_vel) = if cached_ok {
                (pos_cache[slot].clone().unwrap(), vel_cache[slot].clone().unwrap())
            } else {
                (world_positions(world, slot, world.sim_time), pixel_velocities(world, slot, world.sim_time, 0.02))
            };
            let (thrust, torque) = fluid_thrust_torque(world, slot, &ind_pos, &ind_vel);
            let contact = contact_force(world, slot, &ind_pos, &grid, &pos_cache);
            // Deterministic sampling (no RNG call here -- this closure runs
            // in parallel and must stay a pure function of tick-start
            // state, per the comment above): a modulo on slot+tick spreads
            // coverage across the whole population over time instead of
            // always logging the same low-slot individuals.
            // Log a short CONSECUTIVE run of ticks per individual rather than
            // isolated snapshots. Samples 200 ticks apart cannot express what
            // followed an action, so nothing temporal could ever be learned
            // from them. Keeping the condition a pure function of slot and
            // tick means this still needs no state and stays safe inside the
            // parallel phase.
            let phase = (slot as u64 + world.tick_count) % crate::EXPERIENCE_SAMPLE_STRIDE;
            let sample = if phase < crate::EXPERIENCE_TRAJECTORY_LEN {
                Some(ExperienceRow {
                    id: world.individuals.id[slot],
                    tick: world.tick_count,
                    sense: s.to_vec(),
                    action: d.to_vec(),
                    energy: world.individuals.energy[slot],
                    reward: world.individuals.pending_reward[slot],
                })
            } else {
                None
            };
            // Raw density carried through rather than re-scanning
            // neighbors in the sequential phase where the drain is applied.
            (d, thrust, contact, sample, conspecific_density, torque)
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

        // Reward is a PER-TICK quantity, so it is cleared every tick for
        // everyone rather than only for individuals being logged. Clearing
        // only on sampling made each logged row carry everything accumulated
        // since that individual was last sampled -- up to ~200 ticks of
        // reproductions collapsed onto one row (observed reaching 28), which
        // is not a per-step reward at all and wrecked the training targets.
        world.individuals.pending_reward[slot] = 0.0;
        let (d, thrust, contact, _, _, torque) = &pre[i];
        let torque = *torque;
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
        // Swim effort: tanh output remapped onto a positive gain, so a brain
        // can idle to save energy or drive its body hard. Applied to the NEXT
        // tick's kinematics, since this tick's positions were already cached.
        let effort = (d[crate::individuals::SWIM_EFFORT_IDX] * 0.5 + 0.5).clamp(0.0, 1.0);
        world.individuals.swim_gain[slot] =
            crate::SWIM_GAIN_MIN + effort * (crate::SWIM_GAIN_MAX - crate::SWIM_GAIN_MIN);
        {
            let offset = world.individuals.pixel_offset[slot] as usize;
            world.pixels.memory[offset].copy_from_slice(&d[MEMORY_OUT_IDX..MEMORY_OUT_IDX + MEM_DIM]);
        }

        if !is_captured {
            // Steering is now something the body DOES, not something the
            // brain declares. The brain holds a curvature; the curved body
            // pushes water asymmetrically; the resulting torque rotates it,
            // damped by the water. A creature therefore has to learn to use
            // its own shape to go where it wants -- which is the whole point,
            // and is why the previous arrangement (assign a heading, rotate
            // the body to match) left pointing and moving unrelated.
            let desired_mag = (move_x * move_x + move_y * move_y).sqrt().min(1.0);
            let directional_steer = if desired_mag > 0.05 {
                let desired_heading = move_y.atan2(move_x);
                let heading_error = (desired_heading - world.individuals.heading[slot]
                    + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                heading_error.sin() * desired_mag
            } else {
                0.0
            };
            let posture_bias = d[crate::individuals::TURN_BIAS_IDX] * crate::TURN_POSTURE_BIAS_SCALE;
            world.individuals.turn_curvature[slot] =
                (directional_steer + posture_bias).clamp(-1.0, 1.0) * crate::TURN_CURVATURE_SCALE;

            // Test hook: hold the body's shape fixed so propulsion can be
            // measured without the brain reshaping it every tick.
            if let Some((c, g)) = world.freeze_locomotion {
                world.individuals.turn_curvature[slot] = c;
                world.individuals.swim_gain[slot] = g;
            }

            let inertia = (body_size_sum(world, slot) * world.individuals.size_scale[slot]).max(1.0)
                * crate::ROTATIONAL_INERTIA;
            let ang_acc = torque / inertia
                - world.angular_damping * world.individuals.angular_velocity[slot];
            world.individuals.angular_velocity[slot] =
                (world.individuals.angular_velocity[slot] + ang_acc * world.dt)
                    .clamp(-crate::MAX_ANGULAR_SPEED, crate::MAX_ANGULAR_SPEED);
            world.individuals.heading[slot] +=
                world.individuals.angular_velocity[slot] * world.dt;

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
            let tn = world.thermal_noise;
            let noise = [normal(&mut world.rng, 0.0, tn), normal(&mut world.rng, 0.0, tn)];
            let gravity_force = [0.0, -world.gravity * mass];
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
            // Both tests use each part's own radius: a component is in rock
            // if its BODY overlaps stone, not merely if its centre cell does.
            let currently_in_rock = shape_overlaps_rock(world, slot, &current_shape);
            let hits_rock = shape_overlaps_rock(world, slot, &candidate_shape);
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
            // Effort costs. Driving the body hard is superlinearly expensive,
            // so sprinting is a real decision with a real price rather than a
            // free setting every creature would simply max out.
            let effort_cost = world.individuals.swim_gain[slot] * world.individuals.swim_gain[slot];
            world.individuals.energy[slot] -= crate::MOVE_COST * speed_final * effort_cost;
        }

        if world.individuals.alive[slot] && !is_captured {
            let (x, y) = grid_xy(world, world.individuals.root_pos[slot]);
            let idx = (x * world.size + y) as usize;
            // Grazing is a SMALL animal's living. A big body cannot support
            // itself picking at scattered algae -- the food is dispersed and
            // it simply cannot process enough of it -- so grazing yield falls
            // away as mass rises. This is what creates an actual food chain
            // rather than one undifferentiated crowd all eating the same
            // thing: small creatures graze, and anything large enough that
            // grazing no longer pays HAS to hunt, which is where predators
            // (and any reason to be good at hunting) come from. Without it, a
            // large animal could ignore prey entirely and still thrive, which
            // is exactly what was happening -- 96 mean energy with predation
            // optional.
            let graze_mass = body_size_sum(world, slot) * world.individuals.size_scale[slot];
            let graze_efficiency = 1.0 / (1.0 + graze_mass / world.graze_mass_ref);
            let eaten = world.fields.food[idx].min(crate::EAT_RATE * 0.1);
            world.fields.food[idx] -= eaten;
            world.individuals.energy[slot] +=
                eaten * 5.0 * graze_efficiency * digestion_multiplier(world, slot);
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
                || world.individuals.ticks_since_reproduced[slot] >= gestation_ticks(world, slot);
            // Building a child costs what the child actually IS. This was a
            // flat 8.0 regardless of body size, which meant a thirty-part
            // animal produced a thirty-one-part offspring for the same price
            // a two-part blob paid for a three-part one -- biomass conjured
            // from nothing, and no brake whatsoever on population. Charging
            // per part makes body size a real life-history decision: small
            // bodies breed cheaply and often, large ones invest heavily and
            // rarely, and the population limits itself through the energy
            // budget instead of slamming into an artificial cap.
            // Breeding needs room and calm, not just energy and a partner.
            // A big animal needs proportionally more space, so crowding
            // throttles reproduction locally -- population limits itself
            // where it is dense instead of everywhere at once via a global
            // cap, and it creates real pressure to disperse or to find a
            // quiet corner of the reef to breed in.
            let (has_mate, crowd) = mate_and_crowding(world, slot, &grid);
            let allowed_crowd = (crate::BREEDING_SPACE_MAX_NEIGHBORS as f32
                / (1.0 + world.individuals.pixel_count[slot] as f32 * crate::BREEDING_SPACE_SIZE_PENALTY))
                .max(1.0);
            let space_ok = (crowd as f32) <= allowed_crowd;
            // Safety: blood in the water means something was just killed
            // here. Nothing breeds in the middle of that.
            let local_blood = crate::fields::Fields::sample(
                &world.fields.blood, world.size, world.individuals.root_pos[slot]);
            let safe_ok = local_blood < crate::BREEDING_SAFETY_BLOOD_MAX;
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
                && space_ok
                && safe_ok
                && has_mate
            {
                world.individuals.energy[slot] -= repro_cost;
                world.individuals.pending_reward[slot] += crate::REWARD_REPRODUCE;
                world.individuals.ticks_since_reproduced[slot] = 0;
                let child = crate::individuals::reproduce(&mut world.individuals, &mut world.pixels, &mut world.rng, slot, world.growth_tip_weight);
                // Inherited instinct: pull the newborn's decisions part-way
                // toward what the GPU has learned works, then let evolution
                // take it from there.
                if let Some(policy) = world.shared_policy.take() {
                    crate::individuals::distill_policy(
                        &mut world.individuals, child, &policy, crate::POLICY_DISTILL_RATE);
                    world.shared_policy = Some(policy);
                }
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
            world.deaths_starved += 1;
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
            world.deaths_popcap += 1;
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
    let cap = world.food_cap;
    world.fields.step_food_blooms(world.size, &mut world.rng, cap);
    world.fields.step_food_regrow(world.food_regrow_rate * world.food_regrow_multiplier, world.food_cap);
    world.fields.step_diffusion(world.size);
    timings.push(("fields", t0.elapsed().as_secs_f64() * 1000.0));

    world.timings = timings;
}

/// Bodies pushing each other apart. The contact distance is derived from the
/// ACTUAL size of the two parts in contact rather than a flat radius: a part
/// scaled to twice normal occupies twice the space and should repel from
/// twice as far. With a flat radius, large creatures repelled only within
/// 1.2 units while being drawn several units across, so they visibly stacked
/// and piled through one another -- they had no real physical extent.
/// Does any component of this body, at its own bounding radius, overlap rock?
///
/// Treating parts as points let large components sink into stone centre-first;
/// a component with real extent has to be tested by that extent.
fn shape_overlaps_rock(world: &World, slot: usize, shape: &[[f32; 2]]) -> bool {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let scale = world.individuals.size_scale[slot];
    shape.iter().enumerate().any(|(k, &p)| {
        let r = world.pixels.size[offset + k] * scale * 0.5;
        let span = (r.ceil() as i32).max(0);
        let (gx, gy) = grid_xy(world, p);
        for dx in -span..=span {
            for dy in -span..=span {
                let cx = gx as i32 + dx;
                let cy = gy as i32 + dy;
                if cx < 0 || cy < 0 || cx >= world.size as i32 || cy >= world.size as i32 {
                    continue;
                }
                if world.terrain.at(cx as u32, cy as u32) != TerrainKind::Rock {
                    continue;
                }
                let (ox, oy) = (cx as f32 + 0.5, cy as f32 + 0.5);
                if ((p[0] - ox).powi(2) + (p[1] - oy).powi(2)).sqrt() < r + 0.5 {
                    return true;
                }
            }
        }
        false
    })
}

fn contact_force(world: &World, slot: usize, ind_pos: &[[f32; 2]], grid: &SpatialGrid, pos_cache: &[Option<Vec<[f32; 2]>>]) -> [f32; 2] {
    let mut push = [0f32; 2];
    let my_pos = world.individuals.root_pos[slot];
    let my_size = ind_pos.len() as f32;
    let my_offset = world.individuals.pixel_offset[slot] as usize;
    let my_scale = world.individuals.size_scale[slot];
    for other in grid.nearby(my_pos) {
        let other = other as usize;
        if other == slot || !world.individuals.alive[other] { continue; }
        let other_size = world.individuals.pixel_count[other] as f32;
        if dist(world.individuals.root_pos[other], my_pos) > (my_size + other_size) * 0.5 + 2.0 { continue; }
        let other_pos = match &pos_cache[other] {
            Some(p) if p.len() == world.individuals.pixel_count[other] as usize => p,
            _ => continue,
        };
        let other_offset = world.individuals.pixel_offset[other] as usize;
        let other_scale = world.individuals.size_scale[other];
        for (pi, &p) in ind_pos.iter().enumerate() {
            let mut best_d = f32::MAX;
            let mut best = [0f32; 2];
            let mut best_oi = 0usize;
            for (oi, &op) in other_pos.iter().enumerate() {
                let d = dist(p, op);
                if d < best_d { best_d = d; best = op; best_oi = oi; }
            }
            let reach = crate::COLLISION_RADIUS
                * 0.5
                * (world.pixels.size[my_offset + pi] * my_scale
                    + world.pixels.size[other_offset + best_oi] * other_scale);
            if best_d > 1e-6 && best_d < reach {
                let overlap = reach - best_d;
                push[0] += (p[0] - best[0]) / best_d * overlap * crate::COLLISION_STIFFNESS;
                push[1] += (p[1] - best[1]) / best_d * overlap * crate::COLLISION_STIFFNESS;
            }
        }
    }

    // Rock pushes back. Rejecting a MOVE that would enter rock cannot keep
    // bodies out of walls, because a body does not only move -- it undulates.
    // A limb sweeps into stone through the bend wave alone, with the root
    // never translating at all, and swim effort now widens that stroke on
    // demand. Measured, ~18% of bodies still had parts embedded in rock even
    // after terrain-checking every translation and every drag. A repulsion
    // force handles what rejection structurally cannot, and it does it
    // smoothly: bodies get pressed out of walls instead of being frozen
    // against them, so narrow passages stay narrow and the reef stays a real
    // refuge rather than something large animals can bulldoze through.
    for (pi, &p) in ind_pos.iter().enumerate() {
        // Every component has real extent, so it must collide by its own
        // bounding radius rather than as a dimensionless point -- otherwise a
        // large part's body sits inside stone while its centre is outside it,
        // and the bigger the part the worse the overlap. This mirrors what
        // creature-to-creature contact already does.
        let part_radius = world.pixels.size[my_offset + pi] * my_scale * 0.5;
        let reach = crate::TERRAIN_REPULSION_RANGE + part_radius;
        let (gx, gy) = grid_xy(world, p);
        let span = (reach.ceil() as i32).max(1);
        for dx in -span..=span {
            for dy in -span..=span {
                let cx = gx as i32 + dx;
                let cy = gy as i32 + dy;
                if cx < 0 || cy < 0 || cx >= world.size as i32 || cy >= world.size as i32 {
                    continue;
                }
                if world.terrain.at(cx as u32, cy as u32) != TerrainKind::Rock {
                    continue;
                }
                // Push away from that cell's centre, strongest when deepest in.
                let (ox, oy) = (cx as f32 + 0.5, cy as f32 + 0.5);
                let (vx, vy) = (p[0] - ox, p[1] - oy);
                let d = (vx * vx + vy * vy).sqrt();
                if d < reach {
                    let overlap = reach - d;
                    if d > 1e-4 {
                        push[0] += vx / d * overlap * crate::TERRAIN_REPULSION_STIFFNESS;
                        push[1] += vy / d * overlap * crate::TERRAIN_REPULSION_STIFFNESS;
                    }
                }
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
    // How many separate victims this body can strike in one sweep. A single
    // `return` after the first hit meant a huge animal landed exactly one
    // blow per tick while every small creature around it landed its own --
    // predation was one-out, many-in, so being large was a liability rather
    // than an advantage. A long body sweeping through a shoal should catch
    // several of them at once.
    let my_mass = body_size_sum(world, slot) * world.individuals.size_scale[slot];
    let max_targets = (1.0 + my_mass / crate::SWEEP_MASS_PER_TARGET) as u32;
    let mut victims: u32 = 0;

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
            // Gape-limited predation: if the prey is small enough relative to
            // this body -- and a wider gape comes from having mouths -- it is
            // swallowed whole instead of being chipped a part at a time. This
            // is what "eating small things in one sweep" actually requires;
            // chewing prey down pixel by pixel meant a large predator was
            // occupied for many ticks by a creature it should simply have
            // engulfed.
            let prey_mass = body_size_sum(world, other) * world.individuals.size_scale[other];
            // Mouths widen the gape, but only so far. Dividing by the raw
            // bite multiplier (which reaches 3.5) dropped the required size
            // advantage to 1.14, so near-equal creatures swallowed each
            // other whole and nothing could ever accumulate size. A floor
            // keeps engulfing a genuine big-eats-small act.
            let gape = (crate::ENGULF_SIZE_RATIO / bite_multiplier(world, slot).max(0.001))
                .max(crate::ENGULF_GAPE_MIN);
            if my_mass >= prey_mass * gape {
                let (ex, ey) = grid_xy(world, world.individuals.root_pos[other]);
                let eidx = (ex * world.size + ey) as usize;
                let meal = world.individuals.pixel_count[other] as f32
                    * crate::CORPSE_ENERGY_PER_PIXEL
                    * crate::ENGULF_EFFICIENCY
                    * digestion_multiplier(world, slot);
                world.individuals.energy[slot] += meal;
                world.individuals.ticks_since_fed[slot] = 0;
                world.fields.blood[eidx] += crate::BLOOD_EMIT_ON_DEATH;
                world.fights += 1;
                kill(world, other);
            world.deaths_predation += 1;
                victims += 1;
                if victims >= max_targets { return; }
                continue;
            }
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
            world.deaths_predation += 1;
                    world.fields.blood[idx] += crate::BLOOD_EMIT_ON_DEATH;
                }
            }
            if world.individuals.alive[other] && world.rng.random::<f32>() < world.individuals.stickiness[slot] && world.individuals.attached_to[slot] < 0 {
                world.individuals.attached_to[slot] = world.individuals.id[other] as i64;
                let bonus = world.individuals.energy[other].min(crate::CAPTURE_BONUS);
                world.individuals.energy[slot] += bonus;
                world.individuals.energy[other] -= bonus;
            }
            victims += 1;
            if victims >= max_targets { return; }
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
