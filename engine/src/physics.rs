//! Forward kinematics, thrust, sensing, and the full per-tick orchestration
//! -- the Rust-native replacement for pixel_world.py's `World.tick()`. Every
//! step that used to be a Python loop over `Individual` objects is now a
//! loop over SoA component arrays with no interpreter overhead.
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::combat;
use crate::individuals::{ACT_DIM, MEM_DIM, MEMORY_OUT_IDX, SENSE_DIM};
use crate::spatial::{PartGrid, SpatialGrid};
use crate::terrain::TerrainKind;
use crate::{Corpse, ExperienceRow, Weather, World};

fn normal(rng: &mut rand_pcg::Pcg64, mean: f32, std: f32) -> f32 {
    if std <= 0.0 { return mean; }
    Normal::new(mean, std).unwrap().sample(rng)
}

/// The world is a CYLINDER: left and right are joined, top and bottom are not.
///
/// Horizontally there is no edge, so an animal swimming west arrives from the
/// east and no population accumulates in a corner. Vertically there genuinely
/// are two different places -- a surface where plankton enters the water and a
/// floor where what is not eaten settles -- and joining them would destroy the
/// only axis in the world that means anything: depth. So `wrap_delta` applies
/// to x alone, and every y difference stays as it is.
#[inline]
pub(crate) fn wrap_delta(d: f32, size: f32) -> f32 {
    let half = size * 0.5;
    if d > half {
        d - size
    } else if d < -half {
        d + size
    } else {
        d
    }
}

#[inline]
pub(crate) fn wrap_pos(v: f32, size: f32) -> f32 {
    let m = v % size;
    if m < 0.0 { m + size } else { m }
}

fn dist_wrapped(a: [f32; 2], b: [f32; 2], size: f32) -> f32 {
    let dx = wrap_delta(a[0] - b[0], size);
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
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
        // Each component beats on its OWN phase and its own rate, modulated
        // live by its own neural unit. The body rhythm is still the carrier --
        // an animal has one heartbeat, not forty independent ones -- but every
        // joint can lead or lag it and run faster or slower against it.
        //
        // This is what makes a circular stroke possible. A joint in the plane
        // has one degree of freedom, so a limb tip only traces a circle when
        // successive joints are driven about a quarter cycle apart; with a
        // single shared phase every limb could only wave flat, back and forth.
        // It is also how real appendages work: cilia beat in circles, a fin's
        // rays lag one another to throw a wave along the edge, and a rowing
        // limb runs a fast power stroke against a slow recovery.
        let part_drive = 1.0 + crate::PART_DRIVE_AUTHORITY * world.pixels.drive[offset + k];
        let raw_wave = world.pixels.mirror_sign[offset + k] * flex * amp * part_drive.max(0.0)
            * (std::f32::consts::TAU * freq * world.pixels.freq_mult[offset + k] * t
                + phase
                + world.pixels.phase_offset[offset + k]
                + depth[k] * crate::BODY_WAVE_NUMBER)
                .sin();
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

/// Where a body's mass actually sits, and how hard it is to spin about that
/// point. Parts are weighted by their evolved size, so a heavy armoured flank
/// pulls the balance point toward itself exactly as it should.
pub fn mass_center_and_moment(world: &World, slot: usize, pos: &[[f32; 2]]) -> ([f32; 2], f32) {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let mut m_total = 0f32;
    let mut cx = 0f32;
    let mut cy = 0f32;
    for (k, p) in pos.iter().enumerate() {
        let m = world.pixels.size[offset + k].max(0.01);
        m_total += m;
        cx += m * p[0];
        cy += m * p[1];
    }
    if m_total <= 1e-8 || pos.is_empty() {
        return (if pos.is_empty() { [0.0, 0.0] } else { pos[0] }, 1.0);
    }
    let com = [cx / m_total, cy / m_total];
    // Second moment about that centre. This is the real reason a long animal
    // is slow to turn and a compact one is nimble -- using bare mass, as
    // before, made a 30-part body no harder to spin than a ball of the same
    // weight, so shape had no consequence for manoeuvrability at all.
    let mut moment = 0f32;
    for (k, p) in pos.iter().enumerate() {
        let m = world.pixels.size[offset + k].max(0.01);
        let dx = p[0] - com[0];
        let dy = p[1] - com[1];
        moment += m * (dx * dx + dy * dy);
    }
    (com, moment)
}

/// Net fluid force AND the torque it exerts about the body's CENTRE OF MASS.
///
/// Torque is what turns the animal. Returning only the force meant rotation
/// had to come from somewhere else -- and it came from the brain assigning a
/// heading directly, which is why pointing and moving were unrelated.
///
/// The pivot has to be the centre of mass, not the root. Taking moments about
/// the head made every animal swing its entire body around its nose, which is
/// not how anything unattached moves through water -- a free body rotates
/// about its own balance point -- and it is immediately obvious on screen.
pub fn fluid_thrust_torque(world: &World, slot: usize, pos: &[[f32; 2]], vel: &[[f32; 2]]) -> ([f32; 2], f32) {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = pos.len();
    let mut force = [0f32; 2];
    let mut torque = 0f32;
    let (pivot, _) = mass_center_and_moment(world, slot, pos);
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
        // r x F, about the centre of mass, for the 2D scalar torque.
        let rx = pos[k][0] - pivot[0];
        let ry = pos[k][1] - pivot[1];
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
    // Wraps rather than clamps: on a torus a body at x = -0.3 is at the far
    // right edge, not pinned against a wall that no longer exists. Clamping
    // here would silently pile everything that crossed the seam into the
    // outermost row of the field grids.
    let n = world.size as f32;
    let x = wrap_pos(pos[0], n) as u32;
    let y = pos[1].clamp(0.0, n - 1.0) as u32;
    (x.min(world.size - 1), y.min(world.size - 1))
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
    // Upkeep is charged on AREA, not on a part count. Two animals with the
    // same number of parts are not equally expensive to run if one is built
    // from slender tentacles and the other from armour slabs, and counting
    // parts said they were. Every part now costs what it physically occupies
    // -- girth squared, which in two dimensions is its area -- scaled by how
    // expensive that kind of tissue is to keep alive. Normalised so a typical
    // plain body part still costs about one unit, and so this did not
    // silently rescale the whole energy economy.
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let mut load = 0.0;
    for k in 0..count {
        let g = crate::pixels::girth(&world.pixels, offset + k);
        let kind = world.pixels.part_type[offset + k] as usize;
        load += (g * g / crate::PART_AREA_REF) * world.part_metabolism[kind];
    }
    // Kleiber's law, in its two-dimensional form. Upkeep used to be strictly
    // linear in body size, which is
    // both biologically wrong and, here, the quiet reason a worm always beats
    // an animal: a 16-part body paid 16x the running cost of a 1-part body
    // with nothing whatsoever offsetting it, so complexity was a pure tax and
    // selection removed it as fast as growth added it. Real metabolic rate
    // scales as roughly mass^(3/4) across twenty-seven orders of magnitude of
    // body mass (Kleiber 1932; West, Brown & Enquist 1997), which means large
    // animals enjoy a substantial per-gram energy DISCOUNT -- that discount is
    // a large part of why being big is viable at all. The mechanism behind it
    // is Rubner's surface law: an animal exchanges with the world across its
    // BOUNDARY while its cost is carried by its bulk. In three dimensions
    // that is surface over volume and gives 2/3; here the world is flat, so
    // the boundary is a perimeter and the bulk is an area, perimeter grows as
    // the square root of area, and the honest two-dimensional exponent is
    // near 0.5. The exponent is
    // normalised at one part, so the smallest bodies are unaffected and this
    // only ever makes large bodies cheaper to run, never small ones dearer.
    load.max(1.0).powf(world.metabolic_exponent)
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
        let d = dist_wrapped(other_pos, pos, world.size as f32);
        if d > range || d < 1e-6 { continue; }
        let other_size = body_size_sum(world, other) * world.individuals.size_scale[other];
        let n = world.size as f32;
        let delta = [wrap_delta(other_pos[0] - pos[0], n), other_pos[1] - pos[1]];
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
pub(crate) fn sense(world: &World, slot: usize, grid: &SpatialGrid) -> ([f32; SENSE_DIM], f32) {
    let pos = world.individuals.root_pos[slot];
    // Smell is SHORT range; sight is what finds food far off. The food
    // gradient used to be sampled far away for everyone, gated on nothing, so
    // a blind creature foraged exactly as well as a sighted one and eyes were
    // pure cost -- measured, eyes sat at 2.8% of all tissue against a ~5%
    // random baseline, i.e. actively selected against, which also means
    // nothing could perceive anything and navigation intelligence had no
    // foothold. Chemoreception in water really is diffuse and local while
    // vision is directional and long-ranged, so the split is the honest model
    // as well as the useful one.
    let eyes = world.individuals.part_counts[slot][crate::pixels::PART_EYE as usize] as i32;
    let smell_range = if eyes > 0 {
        (world.blind_smell_range + eyes * world.sight_range_per_eye).min(crate::FOOD_SMELL_RANGE)
    } else {
        world.blind_smell_range
    };
    let (fgx, fgy) = crate::fields::Fields::gradient_at_range(&world.fields.food, world.size, pos, smell_range);
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
    let wn = world.size as f32;
    let home_dx = (wrap_delta(home[0] - pos[0], wn) / crate::HOME_RANGE_NORM).tanh();
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
    // Shelter: how enclosed it is here, and which way cover lies. Gated on
    // having eyes, like the other spatial senses -- a blind body feels rock
    // only by running into it.
    if vision_range_of(world, slot) > 0.0 {
        let n = world.size as usize;
        let sample = |px: f32, py: f32| -> f32 {
            let gx = (px as i32).clamp(0, world.size as i32 - 1) as usize;
            let gy = (py as i32).clamp(0, world.size as i32 - 1) as usize;
            world.terrain.enclosure[gx * n + gy]
        };
        let r = crate::SHELTER_SENSE_RANGE;
        let here = sample(pos[0], pos[1]);
        out[30] = here;
        out[31] = (sample(pos[0] + r, pos[1]) - sample(pos[0] - r, pos[1])).clamp(-1.0, 1.0);
        out[32] = (sample(pos[0], pos[1] + r) - sample(pos[0], pos[1] - r)).clamp(-1.0, 1.0);
    }
    // PROPRIOCEPTION: which way this body is actually travelling, expressed
    // in its OWN frame, and how fast.
    //
    // Measured, this is the missing piece behind "they swim with the tail
    // frontward". For any single body, thrust is perfectly locked to its
    // shape -- consistency R=1.000 across every heading. But ACROSS bodies
    // the direction it pushes relative to its nominal heading scatters
    // completely (R=0.259 over fifteen body plans, with several pushing at
    // 160-170 degrees, i.e. genuinely backwards). So "heading" is simply not
    // the direction an animal swims, and which way a given body goes is a
    // property of the body it happened to grow.
    //
    // The animal had no channel telling it any of this. It could not perceive
    // that it was swimming backwards, so no brain, of any size, however
    // trained, could have learned to correct it -- the feedback was not
    // there. This is deliberately NOT fixed by rotating thrust to match
    // heading, which would hand the animal a correct body for free; motion
    // should stay a consequence of how the body moves, and the animal should
    // have to learn to use the one it has. What it gets is the sense needed
    // to make that learnable.
    let vel = world.individuals.velocity[slot];
    let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
    if speed > 1e-5 {
        let (sin_h, cos_h) = world.individuals.heading[slot].sin_cos();
        // Rotate velocity into the body frame: +x is where the animal points.
        out[33] = (vel[0] * cos_h + vel[1] * sin_h) / speed;
        out[34] = (-vel[0] * sin_h + vel[1] * cos_h) / speed;
    }
    out[35] = (speed / crate::MAX_SPEED).clamp(0.0, 1.0) * 2.0 - 1.0;
    out[36..36 + MEM_DIM].copy_from_slice(&mem);
    (out, conspecific_density)
}

/// Removes pixel `local_idx` and every descendant, re-indexing survivors --
/// matches pixel_world.py's `remove_pixel`. Returns true if the individual
/// should die (root removed, or nothing left).
/// Tentacles sweep smaller animals toward the mouth.
///
/// Now that eating requires a mouth to be physically touching the prey, a
/// large animal surrounded by small ones could still only eat whatever
/// happened to drift into its jaws. Real suspension and tentacular feeders do
/// not wait: they sweep prey inward. A tentacle that reaches a smaller animal
/// drags it toward the nearest mouth on the same body, so a big animal can
/// gather a crowd of small ones and work through them, and the placement of
/// tentacles relative to the mouth becomes something worth evolving rather
/// than decoration.
///
/// It costs nothing for the overwhelming majority of the population: a body
/// needs both a tentacle and a mouth before any of this runs at all.
fn tentacle_herding(world: &mut World, grid: &SpatialGrid, pos_cache: &[Option<Vec<[f32; 2]>>]) {
    let slots: Vec<usize> = (0..world.individuals.alive.len())
        .filter(|&s| {
            world.individuals.alive[s]
                && world.individuals.part_counts[s][crate::pixels::PART_TENTACLE as usize] > 0
                && world.individuals.part_counts[s][crate::pixels::PART_MOUTH as usize] > 0
        })
        .collect();
    if slots.is_empty() { return; }
    let t = world.sim_time;
    let wn = world.size as f32;
    for slot in slots {
        let off = world.individuals.pixel_offset[slot] as usize;
        let count = world.individuals.pixel_count[slot] as usize;
        let scale = world.individuals.size_scale[slot];
        // Same cache reuse as the biting path: recomputing a whole body's
        // forward kinematics per tick for this is pure waste when it was
        // already computed this tick, and the length check rejects anything
        // the attachment loop has since altered.
        let owned;
        let my_pos: &[[f32; 2]] = match &pos_cache[slot] {
            Some(v) if v.len() == count => v,
            _ => {
                owned = world_positions(world, slot, t);
                &owned
            }
        };
        if my_pos.len() < count { continue; }
        let mouths: Vec<[f32; 2]> = (0..count)
            .filter(|&k| world.pixels.part_type[off + k] == crate::pixels::PART_MOUTH)
            .map(|k| my_pos[k])
            .collect();
        let tentacles: Vec<([f32; 2], f32)> = (0..count)
            .filter(|&k| world.pixels.part_type[off + k] == crate::pixels::PART_TENTACLE)
            .map(|k| (my_pos[k], crate::pixels::girth(&world.pixels, off + k) * scale))
            .collect();
        if mouths.is_empty() || tentacles.is_empty() { continue; }
        let my_mass = body_size_sum(world, slot) * scale;

        for other in grid.nearby_radius(world.individuals.root_pos[slot], crate::TENTACLE_REACH) {
            let other = other as usize;
            if other == slot || !world.individuals.alive[other] { continue; }
            // Only something you could actually swallow is worth gathering.
            let their_mass = body_size_sum(world, other) * world.individuals.size_scale[other];
            if their_mass * crate::TENTACLE_PREY_RATIO > my_mass { continue; }
            let their_root = world.individuals.root_pos[other];
            // Is any tentacle actually on it?
            let mut gripped = false;
            for (tp, tg) in &tentacles {
                let dx = wrap_delta(their_root[0] - tp[0], wn);
                let dy = their_root[1] - tp[1];
                let reach = crate::TENTACLE_REACH * tg.max(0.2);
                if dx * dx + dy * dy < reach * reach { gripped = true; break; }
            }
            if !gripped { continue; }
            // Drag it toward the nearest mouth.
            let mut best = mouths[0];
            let mut best_d2 = f32::MAX;
            for m in &mouths {
                let d2 = wrap_delta(m[0] - their_root[0], wn).powi(2)
                    + (m[1] - their_root[1]).powi(2);
                if d2 < best_d2 { best_d2 = d2; best = *m; }
            }
            let d = best_d2.sqrt();
            if d < 1e-4 { continue; }
            let pull = crate::TENTACLE_PULL * world.dt;
            let ux = wrap_delta(best[0] - their_root[0], wn) / d;
            let uy = (best[1] - their_root[1]) / d;
            world.individuals.velocity[other][0] += ux * pull;
            world.individuals.velocity[other][1] += uy * pull;
            // Newton's third law: hauling prey in tugs the hauler back, in
            // proportion to how outweighed it is, so a big animal barely
            // feels it and a marginal one gets pulled about by its own catch.
            let react = pull * (their_mass / my_mass.max(0.01)).min(1.0);
            world.individuals.velocity[slot][0] -= ux * react;
            world.individuals.velocity[slot][1] -= uy * react;
        }
    }
}

/// Which part of the prey does this predator actually get its jaws around?
///
/// Eating used to delete a uniformly random pixel of the victim, anywhere on
/// its body, with no reference to where the attacker's mouth was or whether
/// it had a mouth at all. That made a mouth decorative -- which is precisely
/// why mouths were being selected against, sitting at 2.1% of all tissue
/// against a ~5% random baseline. Three rules now decide it, and each one is
/// a real anatomical constraint:
///
///   * There has to be a mouth, and it has to be TOUCHING the part. An animal
///     with no mouth cannot eat another animal at all, so carrying one is
///     worth its upkeep, and where it sits on the body matters.
///   * The part has to fit in the mouth. A gape can only take something
///     smaller than itself, which is the oldest size rule in predation and
///     the reason big mouths are worth growing.
///   * Armour cannot be cut. A plated part turns a bite, so armour buys real
///     protection against being eaten -- though not against blunt combat
///     damage, which still goes through the normal armour arithmetic, so a
///     fully plated animal is tough rather than immortal.
///
/// Returns the local index of the part to bite off, or None if the predator
/// cannot get a bite this tick.
fn bite_target(
    world: &World,
    attacker: usize,
    prey: usize,
    pos_cache: &[Option<Vec<[f32; 2]>>],
) -> Option<u32> {
    let a_off = world.individuals.pixel_offset[attacker] as usize;
    let a_count = world.individuals.pixel_count[attacker] as usize;
    let p_off = world.individuals.pixel_offset[prey] as usize;
    let p_count = world.individuals.pixel_count[prey] as usize;
    if a_count == 0 || p_count == 0 { return None; }

    let wn = world.size as f32;
    let a_scale = world.individuals.size_scale[attacker];
    let p_scale = world.individuals.size_scale[prey];
    // Reuse this tick's cached forward kinematics rather than recomputing two
    // whole bodies for every attached pair, every tick. The cache is only
    // valid for a body that has not been altered since it was built -- this
    // very loop bites parts off victims -- so the length check is what makes
    // the reuse safe, and anything that fails it is rebuilt.
    let cached = |slot: usize| -> Option<&Vec<[f32; 2]>> {
        match &pos_cache[slot] {
            Some(v) if v.len() == world.individuals.pixel_count[slot] as usize => Some(v),
            _ => None,
        }
    };
    let a_owned;
    let a_pos: &[[f32; 2]] = match cached(attacker) {
        Some(v) => v,
        None => {
            a_owned = world_positions(world, attacker, world.sim_time);
            &a_owned
        }
    };
    let p_owned;
    let p_pos: &[[f32; 2]] = match cached(prey) {
        Some(v) => v,
        None => {
            p_owned = world_positions(world, prey, world.sim_time);
            &p_owned
        }
    };

    let mut best: Option<(f32, u32)> = None;
    for ak in 0..a_count.min(a_pos.len()) {
        if world.pixels.part_type[a_off + ak] != crate::pixels::PART_MOUTH { continue; }
        let gape = crate::pixels::girth(&world.pixels, a_off + ak) * a_scale;
        for pk in 0..p_count.min(p_pos.len()) {
            // Armour turns the bite outright.
            if world.pixels.part_type[p_off + pk] == crate::pixels::PART_ARMOR { continue; }
            let bit = crate::pixels::girth(&world.pixels, p_off + pk) * p_scale;
            // It has to fit in the mouth.
            if bit >= gape { continue; }
            let dx = wrap_delta(p_pos[pk][0] - a_pos[ak][0], wn);
            let dy = p_pos[pk][1] - a_pos[ak][1];
            let d2 = dx * dx + dy * dy;
            let reach = crate::COLLISION_RADIUS * 0.5 * (gape + bit) + crate::BITE_REACH_SLACK;
            if d2 > reach * reach { continue; }
            if best.map_or(true, |(bd, _)| d2 < bd) {
                best = Some((d2, pk as u32));
            }
        }
    }
    best.map(|(_, k)| k)
}

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

    // Biting through a part that is NOT a leaf severs everything beyond it,
    // and that flesh has to go somewhere. It used to simply vanish, so a
    // predator biting an animal in half destroyed most of the body outright
    // and nobody, including the predator, got to eat it. Now the severed
    // limb falls away as carrion: real matter, at the place it was cut off,
    // available to whatever finds it. The bitten part itself is not included
    // -- that one is being eaten.
    if to_remove.len() > 1 {
        let positions = world_positions(world, slot, world.sim_time);
        let severed: Vec<[f32; 2]> = to_remove
            .iter()
            .filter(|&&k| k != local_idx)
            .filter_map(|&k| positions.get(k as usize).copied())
            .collect();
        if !severed.is_empty() {
            let anchor = severed[0];
            let local_shape = severed.iter().map(|p| [p[0] - anchor[0], p[1] - anchor[1]]).collect();
            world.corpses.push(crate::Corpse {
                root_pos: anchor,
                local_shape,
                color: world.individuals.color[slot],
                energy: world.meal_energy_per_part * severed.len() as f32,
                initial_energy: world.meal_energy_per_part * severed.len() as f32,
            });
        }
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

/// Total area of one kind of tissue on this body, in the same units the
/// metabolic economy is charged in. Area rather than a part count, because
/// what a filtering mesh or an armour plate does depends on how much of it
/// there is, not how many pieces it was grown in.
pub(crate) fn organ_area(world: &World, slot: usize, kind: u8) -> f32 {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let mut area = 0.0;
    for k in 0..count {
        // Dead tissue performs no function, whatever it used to be.
        if world.pixels.part_type[offset + k] != kind || world.pixels.dead[offset + k] { continue; }
        let g = crate::pixels::girth(&world.pixels, offset + k);
        area += g * g / crate::PART_AREA_REF;
    }
    area * world.individuals.size_scale[slot]
}

/// Organ area weighted by how hard each organ is BEATING.
///
/// An organ's function should follow from what it does, not merely from its
/// presence. A filtering mesh pumps water: how much it moves depends on how
/// fast and how strongly it beats, which is exactly how sessile suspension
/// feeders make a living without going anywhere -- a barnacle, a tube worm and
/// a mussel all sit still and drive water through a comb. Tying the organ to
/// its own actuation makes that strategy available, gives the per-component
/// rate and drive something to be selected FOR, and means an animal can invest
/// in pumping harder rather than only in growing more mesh.
pub(crate) fn organ_beat_area(world: &World, slot: usize, kind: u8) -> f32 {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let mut total = 0.0;
    for k in 0..count {
        if world.pixels.part_type[offset + k] != kind || world.pixels.dead[offset + k] { continue; }
        let g = crate::pixels::girth(&world.pixels, offset + k);
        let area = g * g / crate::PART_AREA_REF;
        // Stroke rate times stroke strength: what the organ actually moves.
        let beat = world.pixels.freq_mult[offset + k]
            * (1.0 + crate::PART_DRIVE_AUTHORITY * world.pixels.drive[offset + k]).max(0.0)
            * world.pixels.flex[offset + k];
        total += area * beat.min(crate::ORGAN_BEAT_MAX);
    }
    total * world.individuals.size_scale[slot]
}

/// How much energy this body can actually hold.
///
/// Built from the evolved per-part `storage` trait and from gut tissue, each
/// weighted by the area of the part providing it, so storage is a thing an
/// animal grows rather than a free universal buffer.
pub(crate) fn storage_capacity(world: &World, slot: usize) -> f32 {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    let mut cap = crate::ENERGY_CAP_BASE;
    for k in 0..count {
        let g = crate::pixels::girth(&world.pixels, offset + k);
        let area = g * g / crate::PART_AREA_REF;
        // What the tissue is, plus how much of a store this particular part
        // has evolved to be. The tissue term is what makes a belly a belly;
        // the evolved term keeps it something selection can still shape.
        let kind = world.pixels.part_type[offset + k] as usize;
        let per = crate::pixels::PART_STORAGE[kind]
            + world.pixels.storage[offset + k] * crate::ENERGY_CAP_PER_STORAGE_TRAIT;
        cap += per * area * crate::ENERGY_CAP_SCALE;
    }
    cap * world.individuals.size_scale[slot]
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
        energy: world.meal_energy_per_part * count as f32,
        initial_energy: world.meal_energy_per_part * count as f32,
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
        if dist_wrapped(world.individuals.root_pos[other], pos, world.size as f32) < crate::BREEDING_SPACE_RADIUS {
            crowd += 1;
        }
        if found_mate { continue; }
        if world.individuals.female[other] == my_female { continue; }
        // Female choice. A female only accepts a partner carrying real
        // reserves, so energy becomes a display of condition rather than a
        // private buffer: males that merely survive do not breed, males that
        // are actually thriving do. This is sexual selection, one of the
        // strongest directional forces in real evolution. Without it, any
        // creature that manages to stay alive beside another reproduces
        // regardless of how well it is doing -- which is most of why a
        // crowded world fills with indistinguishable small breeders.
        if my_female
            && world.individuals.energy[other]
                < world.individuals.energy[slot] * crate::MATE_CHOICE_ENERGY_RATIO
        {
            continue;
        }
        let other_mature = world.individuals.age[other] as f32 >= crate::MATURITY_AGE * world.maturity_multiplier;
        if !other_mature { continue; }
        let other_recovered = !world.individuals.female[other] || world.individuals.ticks_since_reproduced[other] >= gestation_ticks(world, other);
        if !other_recovered { continue; }
        if dist_wrapped(world.individuals.root_pos[other], pos, world.size as f32) < crate::MATE_RADIUS { found_mate = true; }
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
    let grid = SpatialGrid::build(world.size as f32, alive_slots.iter().map(|&s| (s as u32, world.individuals.root_pos[s])));
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
    let grid = timed!("spatial_grid", SpatialGrid::build(world.size as f32, alive_slots.iter().map(|&s| (s as u32, world.individuals.root_pos[s]))));

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
    // How far each body actually reaches from its own root, and the largest
    // such reach in the world. Collision used to broad-phase with the spatial
    // grid's default `nearby`, which searches one cell in each direction --
    // about three world units around the ROOT. That was adequate when animals
    // were three-part blobs. Bodies now run to forty parts and span twenty
    // units or more, so two animals lying completely across one another were
    // never even tested for contact unless their roots nearly touched. That
    // is the mechanical reason they were seen stacking on top of each other,
    // and no amount of crowding cost could fix it, because the overlap was
    // never detected in the first place.
    // Broad phase for contact, over components rather than over animals. See
    // PartGrid: indexing whole bodies by their root forced every query to be
    // wide enough to reach the largest animal in the world.
    let part_grid = PartGrid::build(world.size as f32, alive_slots.iter().flat_map(|&slot| {
        pos_cache[slot]
            .iter()
            .flat_map(move |ps| ps.iter().enumerate().map(move |(k, p)| (slot as u32, k as u32, *p)))
    }));
    let max_girth = alive_slots
        .iter()
        .map(|&slot| {
            let off = world.individuals.pixel_offset[slot] as usize;
            let cnt = world.individuals.pixel_count[slot] as usize;
            (0..cnt)
                .map(|k| crate::pixels::girth(&world.pixels, off + k) * world.individuals.size_scale[slot])
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
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
        // Draining a held victim is EATING, so it needs a mouth like every
        // other way of eating. Gating only the biting path left this one wide
        // open: an animal with no mouth at all could still grapple something
        // and siphon it to death, and since that is how most kills actually
        // happen, mouths stayed nearly worthless -- measured at 2.1% of
        // tissue on the live world while predation accounted for 18562 of
        // 19997 deaths. A mouthless animal can still hold on and still fight,
        // it just cannot feed on what it is holding.
        let mouths = world.individuals.part_counts[slot][crate::pixels::PART_MOUTH as usize];
        let drain = if mouths > 0 {
            world.individuals.energy[target].max(0.0).min(0.4)
        } else {
            0.0
        };
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
            // A much larger victim should throw off a small attacker quickly.
            // The old ceiling meant even a giant needed several ticks to shed
            // a limpet, and with many attackers that is a permanent state.
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
            // Being outmatched has to COST something. The lower clamp was 1.0,
            // meaning a three-part creature latched onto a twenty-five-part
            // armoured animal tore parts off at exactly the rate an
            // equal-sized rival would -- so size bought no protection at all
            // through this path, and big creatures were dismantled by small
            // ones. Direct hits already respect armour (damage is power minus
            // armour, and a small attacker's power is far below a large
            // body's armour, so it lands nothing); chewing bypassed all of
            // that. Now a small attacker can barely tear flesh from something
            // much larger, which is what makes being big worth its upkeep.
            let dominance = (attacker_size / victim_size.max(0.01))
                .clamp(crate::CHEW_DOMINANCE_MIN, crate::CHEW_DOMINANCE_MAX);
            let chew_chance = (crate::CAPTURE_CHEW_CHANCE_BASE * bite_force * dominance)
                .clamp(0.0, crate::CAPTURE_CHEW_CHANCE_MAX);
            if world.rng.random::<f32>() < chew_chance {
                let target_count = world.individuals.pixel_count[target];
                if let Some(victim) = bite_target(world, slot, target, &pos_cache) {
                    let _ = target_count;
                    let (hx, hy) = grid_xy(world, world.individuals.root_pos[target]);
                    let idx = (hx * world.size + hy) as usize;
                    let died = remove_pixel(world, target, victim);
                    // Same principle for chewing: a bite takes a share of
                    // what the victim was carrying, not a fixed ration.
                    let victim_parts = world.individuals.pixel_count[target].max(1) as f32;
                    let reserve_bite = world.individuals.energy[target].max(0.0)
                        * crate::PREDATION_RESERVE_SHARE
                        / victim_parts;
                    world.individuals.energy[target] -= reserve_bite;
                    let bite_gain = (world.meal_energy_per_part + reserve_bite)
                        * digestion_multiplier(world, slot);
                    world.individuals.energy[slot] += bite_gain;
                    world.income_predation += bite_gain as f64;
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

    let t0 = std::time::Instant::now();
    tentacle_herding(world, &grid, &pos_cache);
    timings.push(("herding", t0.elapsed().as_secs_f64() * 1000.0));

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
    // Frequency dependence has to be measured against what "common" actually
    // means in THIS world, not against a fixed number.
    //
    // The threshold was an absolute 8% share, which is only meaningful when
    // there are many lineages. Measured, the world had two or three effective
    // lineages -- so every animal alive was far over the threshold, every
    // lineage sat at maximum pressure, and what was supposed to be a penalty
    // on being COMMON became a flat tax on being alive. It consumed 45% of all
    // the energy in the world, nearly as much as basal metabolism.
    //
    // Worse, it was a runaway: as lineages died the survivors' shares rose,
    // which raised the pressure, which killed more lineages. A mechanism meant
    // to PRESERVE diversity was actively driving the world to a monoculture and
    // then to extinction.
    //
    // Scoring against the mean share (one over the number of lineages) makes
    // it relative, as it always should have been: a lineage at its fair share
    // pays nothing however few lineages there are, and only genuine
    // over-representation is penalised.
    let n_lineages = lineage_counts.len().max(1) as f32;
    let fair_share = 1.0 / n_lineages;
    let share_threshold =
        (fair_share * crate::PATHOGEN_FAIR_SHARE_MULT).max(crate::PATHOGEN_SHARE_THRESHOLD);
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
    let mut pre: Vec<([f32; ACT_DIM], [f32; 2], [f32; 2], Option<ExperienceRow>, f32, f32, [f32; 2], f32, [f32; 2], f32, [f32; crate::individuals::HIDDEN_DIM])> = deciding
        .par_iter()
        .map(|&slot| {
            let (s, conspecific_density) = sense(world, slot, &grid);
            let (mut d, hidden): ([f32; ACT_DIM], [f32; crate::individuals::HIDDEN_DIM]) =
                world.individuals.decide(&world.pixels, slot, &s, &world.shared_enc_w, &world.shared_enc_b);
            // Scrambling control. Blended deterministically from slot and
            // tick rather than from the RNG, because this closure runs in
            // parallel and has to stay a pure function of tick-start state.
            if world.brain_noise > 0.0 {
                let mix = world.brain_noise.clamp(0.0, 1.0);
                for (j, out) in d.iter_mut().enumerate() {
                    let h = (slot as u64)
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(world.tick_count.wrapping_mul(0xBF58476D1CE4E5B9))
                        .wrapping_add(j as u64 * 0x94D049BB133111EB);
                    let h = (h ^ (h >> 31)).wrapping_mul(0xD6E8FEB86659FD93);
                    let r = ((h >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
                    *out = *out * (1.0 - mix) + r * mix;
                }
            }
            let cached_ok = pos_cache[slot].as_ref().map_or(false, |p| p.len() == world.individuals.pixel_count[slot] as usize);
            let (ind_pos, ind_vel) = if cached_ok {
                (pos_cache[slot].clone().unwrap(), vel_cache[slot].clone().unwrap())
            } else {
                (world_positions(world, slot, world.sim_time), pixel_velocities(world, slot, world.sim_time, 0.02))
            };
            let (thrust, torque) = fluid_thrust_torque(world, slot, &ind_pos, &ind_vel);
            // Computed here, where the body's positions already exist, rather
            // than rebuilding forward kinematics for it in the sequential
            // phase.
            let (com, moment) = mass_center_and_moment(world, slot, &ind_pos);
            let (contact, separation, pressure) = contact_force(world, slot, &ind_pos, &part_grid, &pos_cache, max_girth);
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
            (d, thrust, contact, sample, conspecific_density, torque, com, moment, separation, pressure, hidden)
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
    let mut align_sum = 0.0f32;
    let mut align_n = 0.0f32;
    let mut pressure_sum = 0.0f32;
    let mut pressure_n = 0.0f32;
    let mut pressure_max = 0.0f32;
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
        let (d, thrust, contact, _, crowding, torque, com, moment, separation, pressure, hidden) = &pre[i];
        let pressure = *pressure;
        // Actuate every organ from its own neural unit, for the next tick.
        let hidden = *hidden;
        crate::individuals::Individuals::apply_part_drive(
            &world.individuals, &mut world.pixels, slot, &hidden);
        pressure_sum += pressure;
        pressure_n += 1.0;
        if pressure > pressure_max { pressure_max = pressure; }
        let crowding = *crowding;
        let torque = *torque;
        let (com, moment) = (*com, *moment);
        let separation = *separation;
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
        let excess = (share - share_threshold).max(0.0);
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

            // The real second moment about the balance point, so a long body
            // is genuinely sluggish to turn and a compact one is nimble.
            let inertia = (moment * world.individuals.size_scale[slot]).max(1.0)
                * crate::ROTATIONAL_INERTIA;
            let ang_acc = torque / inertia
                - world.angular_damping * world.individuals.angular_velocity[slot];
            world.individuals.angular_velocity[slot] =
                (world.individuals.angular_velocity[slot] + ang_acc * world.dt)
                    .clamp(-crate::MAX_ANGULAR_SPEED, crate::MAX_ANGULAR_SPEED);
            let dtheta = world.individuals.angular_velocity[slot] * world.dt;
            world.individuals.heading[slot] += dtheta;
            // Every part's position is rebuilt each tick by forward kinematics
            // from the root and the heading, so turning the heading alone
            // sweeps the whole body around the ROOT -- the animal pivots on
            // its nose. Carrying the root around the centre of mass by the
            // same angle leaves the balance point where it was and makes the
            // body turn about its middle, which is what a free body in water
            // actually does. The client rebuilds positions from root and
            // heading the same way, so it follows without any change there.
            if dtheta.abs() > 1e-9 {
                let (sin_d, cos_d) = dtheta.sin_cos();
                let rx = world.individuals.root_pos[slot][0] - com[0];
                let ry = world.individuals.root_pos[slot][1] - com[1];
                world.individuals.root_pos[slot] = [
                    com[0] + rx * cos_d - ry * sin_d,
                    com[1] + rx * sin_d + ry * cos_d,
                ];
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
            let tn = world.thermal_noise;
            let noise = [normal(&mut world.rng, 0.0, tn), normal(&mut world.rng, 0.0, tn)];
            let gravity_force = [0.0, -world.gravity * mass];
            let vel = world.individuals.velocity[slot];
            let accel = [
                (thrust[0] * mobility * fin + contact[0] + gravity_force[0] + crawl_force[0] * mobility) / mass - crate::LINEAR_DAMPING * vel[0] + noise[0] * mobility / mass.sqrt(),
                (thrust[1] * mobility * fin + contact[1] + gravity_force[1] * mobility + crawl_force[1] * mobility) / mass - crate::LINEAR_DAMPING * vel[1] + noise[1] * mobility / mass.sqrt(),
            ];
            // MOTOR COMPETENCE: did the animal go where it meant to go?
            //
            // Until now the only rewards were reproducing and holding a full
            // larder, so nothing whatsoever connected a decision to its
            // consequence. An animal could command "swim left" every tick of
            // its life, drift right, and receive exactly the same feedback as
            // one that swam left perfectly -- which is why creatures visibly
            // trying to move can fail at it forever without improving. The
            // information needed to correct it was never in the reward.
            //
            // Comparing intent with outcome is the foundation of motor
            // learning in anything that moves: what a controller needs is not
            // "was that good for you" but "did that do what you asked". The
            // signed cosine between intended and actual direction gives both
            // halves symmetrically -- moving as intended is rewarded, moving
            // OPPOSITE to intent is punished by the same measure, and drifting
            // sideways scores near zero.
            //
            // Weighted by how hard it was actually trying, so an animal
            // coasting deliberately is not marked down for not accelerating,
            // and by speed, so the signal is about real motion rather than
            // intent in still water.
            let new_vel_pre = [vel[0] + accel[0] * world.dt, vel[1] + accel[1] * world.dt];
            {
                let want = (move_x * move_x + move_y * move_y).sqrt();
                let got = (new_vel_pre[0] * new_vel_pre[0] + new_vel_pre[1] * new_vel_pre[1]).sqrt();
                if want > 0.05 && got > 1e-4 {
                    let align = (move_x * new_vel_pre[0] + move_y * new_vel_pre[1]) / (want * got);
                    let effort_w = want.min(1.0);
                    let speed_w = (got / crate::MAX_SPEED).min(1.0);
                    world.individuals.pending_reward[slot] +=
                        crate::REWARD_MOTOR_MATCH * align * effort_w * speed_w;
                    align_sum += align;
                    align_n += 1.0;
                }
            }
            let mut new_vel = [vel[0] + accel[0] * world.dt, vel[1] + accel[1] * world.dt];
            let speed = (new_vel[0] * new_vel[0] + new_vel[1] * new_vel[1]).sqrt();
            if speed > crate::MAX_SPEED {
                new_vel = [new_vel[0] * crate::MAX_SPEED / speed, new_vel[1] * crate::MAX_SPEED / speed];
            }
            // Positional correction for whatever this body is currently
            // inside of -- another animal, or rock. Applied as a direct
            // displacement rather than through the force term, because a
            // force has to fight mass and damping to undo an overlap and a
            // heavy body simply never wins that fight. Only a FRACTION of the
            // penetration is undone per tick, and the whole correction is
            // capped, so bodies ease apart instead of being flung: a solver
            // that removed the full overlap at once would inject energy and
            // make dense crowds explode.
            let mut corr = [
                separation[0] * world.contact_correction * mobility,
                separation[1] * world.contact_correction * mobility,
            ];
            let corr_mag = (corr[0] * corr[0] + corr[1] * corr[1]).sqrt();
            if corr_mag > crate::CONTACT_CORRECTION_MAX {
                let k = crate::CONTACT_CORRECTION_MAX / corr_mag;
                corr = [corr[0] * k, corr[1] * k];
            }
            let mut new_pos = [
                world.individuals.root_pos[slot][0] + new_vel[0] * world.dt + corr[0],
                world.individuals.root_pos[slot][1] + new_vel[1] * world.dt + corr[1],
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
            // The world has no edges any more -- opposite sides are joined,
            // so a body leaving one side arrives at the other. This removes
            // the boundary entirely rather than making it better behaved:
            // there is no wall to be pinned against, no corner to accumulate
            // in, and no edge population that lives differently from the
            // middle purely because of where it happens to be.
            // Left and right are joined; top and bottom are not. The
            // vertical axis is the only one in this world that means
            // anything -- plankton enters at the surface and sinks, so depth
            // is a real gradient with a rich end and a poor end. Wrapping it
            // would make "swim up to feed" meaningless.
            let n = world.size as f32;
            new_pos[0] = wrap_pos(new_pos[0], n);
            if new_pos[1] <= 0.0 && new_vel[1] < 0.0 {
                new_vel[1] = 0.0; // the floor: inelastic
            } else if new_pos[1] >= n - 1.0 && new_vel[1] > 0.0 {
                new_vel[1] = -new_vel[1] * 0.5; // the surface: reflect
            }
            new_pos[1] = new_pos[1].clamp(0.0, n - 1.0);

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
            let move_spend = crate::MOVE_COST * speed_final * effort_cost;
            world.individuals.energy[slot] -= move_spend;
            world.spend_movement += move_spend as f64;
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
            // Filter mesh raises the mass at which grazing stops paying, in
            // proportion to the filtering surface carried. That is the whole
            // organ: a second way to be large. Without it the only route to a
            // big body is hunting, so every large animal in the world is a
            // predator and there is one trophic ladder instead of a food web.
            // A filter only works on water it actually moves through. A
            // basking shark does not hover -- it swims constantly with its
            // mouth open, and the food it gets is set by the volume it
            // sweeps, which is mesh area times speed. Without that condition
            // the organ was a free lunch and it broke the world: it raised
            // the mass at which grazing pays by roughly fourteen times, which
            // cancelled the whole reason large animals had to hunt, so 97% of
            // the population grew a filter, sat still, and idled on 723 mean
            // energy. Everything stopped moving, which is exactly what was
            // reported by eye.
            //
            // Tying yield to swept volume makes the strategy cost what it
            // should: a filter feeder has to keep swimming to eat, and pays
            // the movement energy to do it.
            // Weighted by beat: a mesh that is pumping strains water even
            // when the animal itself is going nowhere.
            let filter_area = organ_beat_area(world, slot, crate::pixels::PART_FILTER);
            let vel = world.individuals.velocity[slot];
            let speed_frac =
                ((vel[0] * vel[0] + vel[1] * vel[1]).sqrt() / crate::FILTER_FLOW_SPEED).clamp(0.0, 1.0);
            let graze_ref =
                world.graze_mass_ref * (1.0 + crate::FILTER_GRAZE_BONUS * filter_area * speed_frac);
            let graze_efficiency = 1.0 / (1.0 + graze_mass / graze_ref);
            // Feeding happens along the WHOLE BODY, not at one point.
            //
            // Grazing used to sample a single cell -- the root -- so a
            // thirty-five part animal spanning twenty cells of water harvested
            // one of them. That was survivable when food regrew in place
            // underneath it, and fatal once food became a thin haze drifting
            // past: the world starved at every plankton level tested,
            // including the most generous, with starvation holding near half
            // of all deaths. The supply was never the problem; the intake was.
            //
            // An animal filtering water collects across its whole surface, so
            // reach scales with the body, which is also what makes being large
            // a viable way to live on plankton rather than a slow death.
            // Intake is bounded by EXPOSED SURFACE, not by part count. See
            // individuals::exposed_surface for why: charging feeding per
            // component made intake scale as N against upkeep at N^0.6, so
            // more tissue always paid, in any arrangement, and no structure an
            // animal grew could ever be useless. Real filtration scales as
            // roughly W^0.66-0.70 -- shallower than metabolism's W^0.75 --
            // which is what gives real animals a finite best size instead of
            // an unbounded reason to grow.
            let mut eaten = 0.0f32;
            let surface = crate::individuals::exposed_surface(
                &world.pixels,
                world.individuals.pixel_offset[slot],
                world.individuals.pixel_count[slot],
            ) * world.individuals.size_scale[slot];
            // A mouth or a filter mesh is a feeding surface; plain flank is
            // not. Structural tissue can still absorb a little -- small
            // animals really do -- but an animal that wants to live on
            // plankton has to grow the apparatus for it.
            // Feeding apparatus counted by how hard it is working, so pumping
            // is a real alternative to swimming for getting water through a
            // filter -- which is the whole living of every sessile suspension
            // feeder in the sea.
            let feeding_organs = organ_beat_area(world, slot, crate::pixels::PART_FILTER)
                + organ_beat_area(world, slot, crate::pixels::PART_MOUTH) * 0.4;
            // Sublinear in surface, which is the part that was still wrong.
            //
            // Charging intake on exposed surface was supposed to stop bigger
            // always being better, and for a compact body it does -- perimeter
            // grows as sqrt(N). But an ELONGATED body's perimeter grows as N,
            // and elongation is exactly what this pressure selected for, so
            // intake went straight back to scaling as N against upkeep at
            // N^0.75 and large animals won unboundedly again. Measured, a
            // four-part founder ran at 0.98 of its own upkeep -- a net loss
            // with food everywhere -- while a forty-part animal ran at 1.74,
            // so founders could not establish and the world crashed from 98 to
            // 4 within a few hundred ticks at every food level tried.
            //
            // Real filtration scales as W^0.66-0.70, SHALLOWER than
            // metabolism's W^0.75. Putting that exponent on the surface term
            // restores the ordering for every body shape rather than only for
            // compact ones, and it makes small animals viable again -- which
            // is a precondition for any size structure at all.
            let owned_g;
            let graze_pos: &[[f32; 2]] = match &pos_cache[slot] {
                Some(v) if v.len() == world.individuals.pixel_count[slot] as usize => v,
                _ => {
                    owned_g = world_positions(world, slot, world.sim_time);
                    &owned_g
                }
            };
            // SWEPT VOLUME, not surface area.
            //
            // Exposed surface turned out to apply no pressure on shape at all,
            // and the reason is worth keeping. These bodies are trees on a
            // lattice with only one or two children per node, so almost every
            // component has two or three of its four sides open no matter how
            // the animal is arranged: a chain of twenty and a bush of twenty
            // have nearly the same exposed surface. The measure could not tell
            // them apart, so morphology stayed arbitrary and elongation fell
            // back to 0.4.
            //
            // What a filter feeder actually collects is the water it passes
            // THROUGH, and that is frontal width times distance travelled --
            // the span of the body perpendicular to its motion, times its
            // speed. This does distinguish shapes, and it distinguishes them
            // the way the real animals are distinguished: a wide fan held
            // across the flow gathers, a compact lump does not, and neither
            // gathers anything at all sitting still. It is also why real
            // suspension feeders are built as combs, nets, fans and crowns
            // rather than as balls.
            let vx = vel[0];
            let vy = vel[1];
            let speed = (vx * vx + vy * vy).sqrt();
            let mut swept = 0.0f32;
            if speed > 1e-4 && graze_pos.len() > 1 {
                // Unit vector perpendicular to travel.
                let (px, py) = (-vy / speed, vx / speed);
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for q in graze_pos.iter() {
                    let proj = q[0] * px + q[1] * py;
                    if proj < lo { lo = proj; }
                    if proj > hi { hi = proj; }
                }
                let frontal = (hi - lo).max(0.0);
                swept = frontal * speed.min(crate::MAX_SPEED);
            }
            // A body still absorbs a little through its own surface even when
            // stationary, so drifting is a meagre living rather than instant
            // death -- but only a meagre one.
            let raw = swept * crate::GRAZE_SWEPT_RATE
                + surface * crate::GRAZE_SURFACE_RATE
                + feeding_organs * crate::GRAZE_ORGAN_RATE;
            let intake_capacity = raw.max(0.0).powf(crate::GRAZE_SURFACE_EXPONENT);
            let n_cells = world.individuals.pixel_count[slot].max(1) as f32;
            let per_part = intake_capacity / n_cells;
            let mut cells: Vec<usize> = graze_pos
                .iter()
                .map(|p| {
                    let (gx, gy) = grid_xy(world, *p);
                    (gx * world.size + gy) as usize
                })
                .collect();
            // A body doubled back on itself would otherwise harvest the same
            // water twice over.
            cells.sort_unstable();
            cells.dedup();
            for c in cells {
                let here = world.fields.food[c];
                if here <= 0.0 { continue; }
                // Saturating intake (a Holling type II functional response on
                // concentration). Two things follow from it, and the second is
                // the one that matters.
                //
                // First, diminishing returns: rich water is not proportionally
                // better than adequate water, because there is a limit to how
                // fast a body can process what it strains.
                //
                // Second, and the reason this is here: it leaves the resource
                // a REFUGE. Grazing used to take the cell down to zero, so a
                // population boom stripped the water bare and then starved in
                // it -- which is precisely the boom-bust the world has been
                // oscillating through, between roughly twenty animals and
                // five hundred. A filter feeder genuinely cannot do this:
                // below some concentration, straining water costs more than
                // the food in it is worth, so the last of a resource is never
                // harvested. That unharvestable remainder is what lets a
                // depleted patch recover, and it is the classic stabiliser for
                // consumer-resource cycles.
                let take = if world.graze_half_saturation <= 0.0 {
                    here.min(per_part)
                } else {
                    // Clamped to what is actually present. The saturating form
                    // exceeds `here` whenever per_part is larger than the
                    // half-saturation constant -- which it is, once intake was
                    // raised -- so grazing was removing more plankton than the
                    // cell contained and driving the food field NEGATIVE
                    // (measured at -0.0284 mean). A negative concentration then
                    // feeds back into every gradient and every subsequent
                    // intake calculation as though the water owed food.
                    (per_part * here / (here + world.graze_half_saturation)).min(here)
                };
                if take <= 1e-6 { continue; }
                world.fields.food[c] = here - take;
                eaten += take;
            }
            let gain = eaten * world.plankton_calories * graze_efficiency
                * digestion_multiplier(world, slot);
            world.individuals.energy[slot] += gain;
            world.income_plankton += gain as f64;
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
            // A body plan that has just changed is shielded briefly. Its
            // controller was inherited for the OLD body and has not had a
            // chance to adapt, so judging it at full pressure punishes
            // morphological innovation for a reason that says nothing about
            // whether the new shape is any good -- which is how morphology
            // converges early on whatever was safe and stops exploring.
            //
            // Named `selection_relief`, NOT `pressure`: there is already a
            // `pressure` in this scope holding the animal's space pressure,
            // and calling this one the same thing shadowed it. The crowding
            // block below then computed its threshold against 1.0 instead of
            // against a measured pressure of 18, so `over` was always zero and
            // the entire density regulator was silently switched off -- while
            // still reporting its deaths as 0% and looking, from outside, like
            // a mechanism that simply did not work.
            let protect = world.individuals.innovation_protect[slot];
            let selection_relief = if protect > 0 {
                world.individuals.innovation_protect[slot] = protect - 1;
                crate::INNOVATION_PROTECT_METABOLISM
            } else {
                1.0
            };
            let metab_spend = metabolism * world.metabolism_multiplier * selection_relief;
            world.individuals.energy[slot] -= metab_spend;
            world.spend_metabolism += metab_spend as f64;

            // Energy is now BOUNDED by what the body can actually hold.
            // Before this an animal simply accumulated without limit -- one
            // was observed sitting on 898 units, an enormous buffer against
            // every hazard in the world that cost nothing to carry and was
            // not attached to any organ. A real store has to be built and
            // fed. Capacity comes from the evolved per-part `storage` trait
            // and from gut tissue specifically, weighted by the area of the
            // part doing the storing, so a deep-bellied animal can bank a
            // long fast and a slender one lives hand to mouth. With food
            // scarce, that buffer is what carries a body between meals, which
            // is what makes storage worth its upkeep.
            let cap = storage_capacity(world, slot);
            if world.individuals.energy[slot] > cap {
                world.individuals.energy[slot] = cap;
            }

            // A full larder is worth something in itself. Kept deliberately
            // small relative to the reproduction reward: an earlier attempt
            // to reward energy directly was measured making animals hoard
            // instead of breed -- mean energy tripled while births fell 73%
            // -- so this is a nudge toward keeping reserves, not a reason to
            // stop living. It is scored as a FRACTION of capacity so it
            // rewards being well-fed for your build rather than simply being
            // large.
            if cap > 0.0 {
                world.individuals.pending_reward[slot] +=
                    crate::REWARD_ENERGY_STOCK * (world.individuals.energy[slot] / cap).clamp(0.0, 1.0);
            }

            // Crowding costs. Bodies were piling on top of one another with no
            // penalty at all, so a creature could sit in a heap, breed, and do
            // nothing else -- which is most of what the world had become.
            // Competition for space is a real and continuous cost in nature:
            // packed animals interfere with each other's feeding, are stressed
            // by proximity, and pay for it. Charged above a threshold so an
            // ordinary family group is free and only genuine crush is
            // punished, and scaled by body size because a large animal needs
            // proportionally more room.
            let crowd_excess = (crowding - crate::CROWDING_TOLERANCE).max(0.0);
            if crowd_excess > 0.0 {
                let mass = body_size_sum(world, slot) * world.individuals.size_scale[slot];
                world.individuals.energy[slot] -= crate::CROWDING_ENERGY_COST
                    * crowd_excess
                    * (1.0 + mass * crate::CROWDING_SIZE_FACTOR);
            }

            // Density-dependent mortality. The crowding cost above is a gentle
            // energy tax on kin density around the root, and it never
            // regulated anything: the world ran to 1778 animals and 30799
            // components in a 240x240 space with starvation at 5.8% of deaths,
            // so nothing was limiting numbers at all. Food abundance was doing
            // no work and neither was the tax.
            //
            // Density-dependent mortality is the standard regulator in real
            // populations, and it is a pressure rather than a rule about
            // behaviour: crushed animals interfere, foul their surroundings
            // and fail, and the intensity rises faster than linearly with how
            // packed they are. Measured against actual bodies pressing on this
            // one, blind to kinship -- a sibling takes up exactly as much room
            // as a stranger -- so a lineage cannot escape it by simply filling
            // the world with its own copies, which is precisely what it had
            // been doing.
            //
            // Nothing here scripts a response. It makes space worth having,
            // which is what gives dispersal, spacing and defending a patch
            // something to be selected FOR. The animal can already sense
            // crowding and already has an aggression output; this is the
            // reason to use them.
            let over = (pressure - world.space_pressure_tolerance).max(0.0);
            if over > 0.0 {
                let mass = body_size_sum(world, slot) * world.individuals.size_scale[slot];
                let crowd_spend =
                    crate::SPACE_PRESSURE_ENERGY_COST * over * over * (1.0 + mass * crate::CROWDING_SIZE_FACTOR);
                world.individuals.energy[slot] -= crowd_spend;
                world.spend_crowding += crowd_spend as f64;
                let risk = (world.space_pressure_mortality * over * over * selection_relief)
                    .min(crate::SPACE_PRESSURE_MORTALITY_MAX);
                if risk > 0.0 && world.rng.random::<f32>() < risk {
                    kill(world, slot);
                    world.deaths_crowding += 1;
                    continue;
                }
            }

            // Contested ground. Sitting inside someone else's scent marks is
            // expensive: it is the cost of trespassing on a defended range,
            // and it is what gives territorial marking a consequence rather
            // than leaving it a decorative field. A creature near its OWN home
            // pays nothing, so holding a range is worth something.
            let local_marks = crate::fields::Fields::sample(
                &world.fields.territory, world.size, world.individuals.root_pos[slot]);
            if local_marks > crate::TRESPASS_MARK_THRESHOLD {
                let home = world.individuals.home_pos[slot];
                let pos = world.individuals.root_pos[slot];
                let from_home = dist_wrapped(home, pos, world.size as f32);
                if from_home > crate::BREEDING_SPACE_RADIUS {
                    world.individuals.energy[slot] -=
                        crate::TRESPASS_ENERGY_COST * (local_marks - crate::TRESPASS_MARK_THRESHOLD);
                }
            }

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
                // Charged as a multiple of the host's OWN upkeep rather than
                // as a flat energy drain.
                //
                // A fixed rate silently stops mattering whenever the energy
                // economy inflates: 0.02 per tick was a real burden when
                // animals lived on twenty or a hundred units, and is nothing at
                // all to a body banking two thousand. Measured, the world had
                // collapsed to an effective 1.68 lineages with one at 75% while
                // this mechanism was nominally running -- it was simply too
                // small to notice. Scaling it to the host's metabolism means a
                // disease costs the same FRACTION of a living regardless of how
                // rich the world becomes, which is both how disease actually
                // burdens an organism and the only version of this that cannot
                // quietly become decorative again.
                let upkeep = crate::PER_PIXEL_METABOLISM
                    * metabolic_part_load(world, slot)
                    * world.individuals.size_scale[slot];
                let toll = world.pathogen_damage_rate * pathogen_pressure * resistance * upkeep;
                let before = world.individuals.energy[slot];
                world.individuals.energy[slot] -= toll;
                world.spend_disease += toll as f64;
                // Attribute the death to the thing that actually caused it.
                if before > 0.0 && world.individuals.energy[slot] <= 0.0 {
                    world.deaths_disease += 1;
                }
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
            // Breeding costs a real FRACTION of what this animal can hold, not
            // a small absolute number.
            //
            // The old cost was base plus 1.2 per part -- about 26 energy for a
            // twenty-part animal sitting on hundreds -- so an adult could breed
            // dozens of times and offspring were nearly free. That is pure
            // r-selection, and it produced exactly what it should: a mean age
            // of 580 against an observed maximum of 8198, a population turning
            // over every 285 ticks, and animals that persist by sheer numbers
            // rather than by being good at anything. Being competent could not
            // pay when being numerous was so cheap.
            //
            // Pricing it against storage capacity also makes it immune to the
            // failure that has bitten twice already: an absolute constant goes
            // inert the moment the energy economy inflates past it.
            let capacity = storage_capacity(world, slot);
            let repro_cost = (crate::REPRODUCE_BASE_COST
                + world.repro_cost_per_part * offspring_parts)
                .max(capacity * world.repro_capacity_fraction);
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
                world.spend_reproduction += repro_cost as f64;
                world.individuals.pending_reward[slot] += crate::REWARD_REPRODUCE;
                world.individuals.ticks_since_reproduced[slot] = 0;
                let child = crate::individuals::reproduce(&mut world.individuals, &mut world.pixels, &mut world.rng, slot, world.growth_tip_weight);
                // Parental investment: most of what the parent spent goes INTO
                // the offspring rather than evaporating. A newborn used to
                // start on a flat 10 units whatever it cost to make, so there
                // was no way to trade quantity for quality -- every child was
                // equally underfed and equally likely to die, and the only
                // strategy available was to make more of them. An endowed
                // offspring can actually survive its first famine, which is
                // what makes producing fewer, better-provisioned young a
                // strategy the world can discover.
                world.individuals.energy[child] =
                    (repro_cost * crate::REPRO_ENDOWMENT_SHARE).max(10.0);
                // Inherited instinct: pull the newborn's decisions part-way
                // toward what the GPU has learned works, then let evolution
                // take it from there.
                if let Some(policy) = world.shared_policy.take() {
                    crate::individuals::distill_policy(
                        &mut world.individuals, child, &policy, world.policy_distill_rate);
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

    world.mean_food = if world.fields.food.is_empty() {
        0.0
    } else {
        world.fields.food.iter().sum::<f32>() / world.fields.food.len() as f32
    };
    world.mean_motor_align = if align_n > 0.0 { align_sum / align_n } else { 0.0 };
    world.mean_pressure = if pressure_n > 0.0 { pressure_sum / pressure_n } else { 0.0 };
    world.max_pressure = pressure_max;
    timings.push(("decide_apply", t0.elapsed().as_secs_f64() * 1000.0));
    timings.push(("  of_which_reproduce", t_reproduce));
    timings.push(("  of_which_collision", t_collision));

    let t0 = std::time::Instant::now();
    scavenge_all(world, &deciding);
    timings.push(("scavenge", t0.elapsed().as_secs_f64() * 1000.0));

    // Enforce the larder ONCE, at the end of the tick, over every income path.
    //
    // The cap was applied in the metabolism block, which runs before predation,
    // the capture bonus and scavenging all add their energy -- so three of the
    // four ways of feeding bypassed it completely. An animal whose storage
    // could hold about 135 was observed sitting on 1862, which is not a
    // reserve, it is immunity, and it is what turns a good year into a
    // population explosion: a body that can bank twelve times its own capacity
    // converts a windfall straight into a burst of offspring, and the crash
    // follows. Capping every path is what makes storage a real constraint
    // rather than a suggestion that only grazers happen to obey.
    let t0 = std::time::Instant::now();
    for &slot in &alive_slots {
        if !world.individuals.alive[slot] { continue; }
        let cap = storage_capacity(world, slot);
        if world.individuals.energy[slot] > cap {
            world.spend_capped += (world.individuals.energy[slot] - cap) as f64;
            world.individuals.energy[slot] = cap;
        }
    }
    timings.push(("storage_cap", t0.elapsed().as_secs_f64() * 1000.0));

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
    // Whale fall. Every so often something very large dies somewhere above and
    // its body comes down -- a single enormous, concentrated windfall in a
    // world whose ordinary food is a thin drifting haze.
    //
    // Worth having for the same reason it matters in the real deep sea: it is
    // a completely different KIND of resource from marine snow. Snow rewards
    // steady filtering along the drift; a carcass rewards noticing one,
    // getting to it fast, and holding it against everything else that noticed.
    // That gives scavenging, competition over a fixed point, and a reason to
    // travel far and quickly -- none of which a uniform haze can select for.
    if world.rng.random::<f32>() < crate::WHALE_FALL_CHANCE {
        let n = world.size as f32;
        let x = world.rng.random_range(0.0..n);
        let parts = world.rng.random_range(crate::WHALE_FALL_MIN_PARTS..crate::WHALE_FALL_MAX_PARTS);
        // A slab of carcass rather than a point, so several animals can feed
        // on it at once and have to share it.
        let w = (parts as f32).sqrt().ceil() as i32;
        let shape: Vec<[f32; 2]> = (0..parts as i32)
            .map(|i| [(i % w) as f32 * 0.9, (i / w) as f32 * 0.9])
            .collect();
        world.corpses.push(crate::Corpse {
            root_pos: [x, n - 2.0],
            local_shape: shape,
            color: [232, 228, 214],
            energy: world.meal_energy_per_part * parts as f32 * crate::WHALE_FALL_RICHNESS,
            initial_energy: world.meal_energy_per_part * parts as f32 * crate::WHALE_FALL_RICHNESS,
        });
        world.whale_falls += 1;
    }
    for c in world.corpses.iter_mut() {
        if c.root_pos[1] > 0.0 { c.root_pos[1] = (c.root_pos[1] - crate::CORPSE_SINK_RATE).max(0.0); }
    }
    // Carrion SMELLS. This was missing entirely: a carcass emitted nothing, so
    // the only way to find one was to collide with it, and a whale fall worth
    // thousands of units could sink past a starving animal a few body lengths
    // away without it ever knowing. That is not how any of this works in the
    // sea -- a whale fall releases an enormous chemical plume and is found by
    // scavengers from a very long way off, which is exactly why it gathers a
    // crowd worth competing in.
    //
    // Emitted into the existing blood field, because animals already sense its
    // gradient: the machinery to smell a carcass was already there, nothing was
    // putting a smell into it. Scent scales with how much carcass is left, so a
    // fresh whale fall is a beacon and a stripped one barely registers.
    for i in 0..world.corpses.len() {
        let (pos, energy) = (world.corpses[i].root_pos, world.corpses[i].energy);
        if energy <= 0.01 { continue; }
        let (cx, cy) = grid_xy(world, pos);
        let idx = (cx * world.size + cy) as usize;
        let scent = (energy * crate::CARRION_SCENT_PER_ENERGY).min(crate::CARRION_SCENT_MAX);
        world.fields.blood[idx] += scent * world.dt;
    }
    // Carrion ROTS. Corpses only ever lost energy by being eaten, so anything
    // nobody got round to eating stayed in the world forever: 1465 corpses
    // carrying 27258 points had accumulated, which is a 2.3 MB state payload
    // published six times a second and tens of thousands of draw calls a
    // frame. That is the whole reason the view is clunky, and it is also just
    // wrong -- a carcass on the seabed is consumed by things far smaller than
    // anything modelled here, and it does not last indefinitely.
    //
    // Decaying returns some of it to the water as well, so an unclaimed body
    // feeds the plankton instead of vanishing: the nutrients go back into the
    // system the way they actually do.
    for i in 0..world.corpses.len() {
        // Big carcasses rot SLOWLY. Decay happens at the surface, and a large
        // body has far less surface per unit of itself than a small one -- the
        // same square-cube argument that governs everything else here. A
        // sardine is gone in a day and a whale fall in the real deep sea feeds
        // a community for decades.
        //
        // Flat decay made a whale fall last about as long as a minnow, so the
        // single most interesting event in this world came and went in a
        // thousand ticks and was almost never there to be found. Scaling by
        // the inverse root of its mass makes a great carcass a place that
        // persists, which is the entire reason a whale fall matters
        // ecologically: it is not just a lot of food, it is a lot of food that
        // stays put long enough to be worth crossing an ocean for.
        let e = world.corpses[i].energy.max(1.0);
        let decay = crate::CORPSE_DECAY_RATE
            * (crate::CORPSE_DECAY_MASS_REF / e).sqrt().clamp(0.08, 1.0);
        let lost = world.corpses[i].energy * decay;
        world.corpses[i].energy -= lost;
        let pos = world.corpses[i].root_pos;
        let (cx, cy) = grid_xy(world, pos);
        let idx = (cx * world.size + cy) as usize;
        world.fields.food[idx] =
            (world.fields.food[idx] + lost * crate::CORPSE_DECAY_TO_FOOD).min(world.food_cap);
        // Shrink the remains to match, so a rotting carcass is visibly going.
        let c = &mut world.corpses[i];
        if c.initial_energy > 0.0 && !c.local_shape.is_empty() {
            let frac = (c.energy / c.initial_energy).clamp(0.0, 1.0);
            let want = ((c.local_shape.len() as f32) * frac).ceil() as usize;
            if want < c.local_shape.len() {
                c.local_shape.truncate(want.max(1));
            }
        }
    }
    world.corpses.retain(|c| c.energy > crate::CORPSE_MIN_ENERGY);
    // Hard ceiling as a backstop, keeping the richest: a pathological case
    // should degrade the oldest scraps rather than the framerate.
    if world.corpses.len() > crate::MAX_CORPSES {
        world.corpses.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));
        world.corpses.truncate(crate::MAX_CORPSES);
    }
    timings.push(("corpses", t0.elapsed().as_secs_f64() * 1000.0));

    let t0 = std::time::Instant::now();
    let _cap = world.food_cap;
    // Marine snow replaces regrowth-in-place. See Fields::step_marine_snow:
    // food now enters at the surface and sinks, so it must be swum to rather
    // than sat on.
    let phase = world.tick_count as f32;
    let a = (phase / crate::SNOW_BLOOM_PERIOD_A * std::f32::consts::TAU).sin();
    let b = (phase / crate::SNOW_BLOOM_PERIOD_B * std::f32::consts::TAU).sin();
    let cycle = (a * 0.6 + b * 0.4) * 0.5 + 0.5; // 0..1
    // Subtracting a floor and clamping at zero gives real famines rather than
    // a signal that merely dips.
    let bloom = ((cycle - crate::SNOW_BLOOM_FLOOR).max(0.0) / (1.0 - crate::SNOW_BLOOM_FLOOR))
        * world.snow_strength
        * world.food_regrow_multiplier;
    world.fields.step_marine_snow(
        world.size,
        &mut world.rng,
        world.food_cap,
        crate::SNOW_SINK_RATE,
        bloom,
        world.snow_plumes,
        world.production_rows,
        phase,
    );
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
        let r = crate::pixels::girth(&world.pixels, offset + k) * scale * 0.5;
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

fn contact_force(
    world: &World,
    slot: usize,
    ind_pos: &[[f32; 2]],
    part_grid: &PartGrid,
    pos_cache: &[Option<Vec<[f32; 2]>>],
    max_girth: f32,
) -> ([f32; 2], [f32; 2], f32) {
    // Space pressure: how many OTHER animals' components are pressing into
    // the neighbourhood of this animal's own, per component of its body.
    //
    // The existing crowding measure counted genetically similar neighbours
    // around the root, which is wrong twice over. It ignored a crush of
    // unrelated animals entirely -- a body could be buried in strangers at no
    // cost -- and counting roots ignores that a forty-part animal occupies far
    // more room than a three-part one. This counts bodies against bodies, and
    // is blind to kinship on purpose: competition for physical space does not
    // care who your relatives are.
    let mut foreign_near = 0.0f32;
    let wn = world.size as f32;
    let mut push = [0f32; 2];
    // Contact was a pure force: divided by mass, fought by damping, and
    // integrated over time. That is fine for a light touch and hopeless for a
    // real overlap -- a heavy body barely accelerates out of one, so two
    // animals that end up inside each other simply stay there, which is what
    // 70% of components sitting within contact range of another animal's
    // components actually means. Every serious contact solver therefore also
    // corrects POSITION directly, moving overlapping bodies apart by a
    // fraction of their penetration each step (Baumgarte stabilisation, and
    // the same idea position-based dynamics is built on). Accumulated here,
    // applied to the root once, in the sequential phase.
    let mut separation = [0f32; 2];
    let my_mass = body_size_sum(world, slot).max(0.01) * world.individuals.size_scale[slot];
    let my_offset = world.individuals.pixel_offset[slot] as usize;
    let my_scale = world.individuals.size_scale[slot];
    // One query per COMPONENT, at that component's own contact reach, instead
    // of one query per animal at the world's largest body radius followed by
    // an all-pairs scan. Same result, a fraction of the work.
    let my_off = world.individuals.pixel_offset[slot] as usize;
    for (pi, &p) in ind_pos.iter().enumerate() {
        let my_girth = crate::pixels::girth(&world.pixels, my_off + pi) * my_scale;
        let query = crate::COLLISION_RADIUS * 0.5 * (my_girth + max_girth);
        part_grid.for_each_near(p, query, |other_u, oi_u| {
            let other = other_u as usize;
            if other == slot || !world.individuals.alive[other] { return; }
            let oi = oi_u as usize;
            // The cached positions and this grid are both built BEFORE the
            // attachment loop, which bites parts off victims and reallocates
            // their storage -- so a body's offset and count can both have
            // moved since. Requiring the cached length to still match the
            // live part count rejects exactly those bodies; checking only
            // that the index is inside the cache does not, and reads past the
            // end of the arena for anything that shrank this tick.
            let op = match &pos_cache[other] {
                Some(v) if v.len() == world.individuals.pixel_count[other] as usize => v[oi],
                _ => return,
            };
            let dx = wrap_delta(p[0] - op[0], wn);
            let dy = p[1] - op[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= 1e-12 { return; }
            let other_off = world.individuals.pixel_offset[other] as usize;
            let other_scale = world.individuals.size_scale[other];
            let reach = crate::COLLISION_RADIUS
                * 0.5
                * (my_girth + crate::pixels::girth(&world.pixels, other_off + oi) * other_scale);
            // Count only components genuinely pressing on this one. The grid
            // walks whole CELLS, so the candidates it hands back cover a block
            // several times wider than any contact -- counting all of them
            // overstated pressure by more than an order of magnitude and
            // pinned essentially every animal at the maximum death rate, which
            // emptied the world.
            let crowd_reach = reach * crate::SPACE_PRESSURE_RANGE;
            if d2 < crowd_reach * crowd_reach {
                foreign_near += 1.0;
            }
            if d2 >= reach * reach { return; }
            let d = d2.sqrt();
            let overlap = reach - d;
            let ux = dx / d;
            let uy = dy / d;
            push[0] += ux * overlap * world.collision_stiffness;
            push[1] += uy * overlap * world.collision_stiffness;
            // The lighter body yields more, so a small animal is shouldered
            // aside by a large one rather than the two splitting the
            // correction evenly. Both bodies compute their own share
            // independently and the pair separates.
            let other_mass = body_size_sum(world, other).max(0.01) * other_scale;
            let share = other_mass / (my_mass + other_mass);
            separation[0] += ux * overlap * share;
            separation[1] += uy * overlap * share;
        });
    }

    let mut terrain_correction = [0f32; 2];
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
        let part_radius = crate::pixels::girth(&world.pixels, my_offset + pi) * my_scale * 0.5;
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
                        terrain_correction[0] += vx / d * overlap;
                        terrain_correction[1] += vy / d * overlap;
                    }
                }
            }
        }
    }
    // Rock gets a positional correction too, and a firmer one: stone does not
    // yield, so the whole of the penetration is the body's to undo. Terrain
    // repulsion alone was already known not to keep bodies out of walls on
    // its own, for exactly the reason contact force did not keep them out of
    // each other.
    if terrain_correction[0] != 0.0 || terrain_correction[1] != 0.0 {
        separation[0] += terrain_correction[0];
        separation[1] += terrain_correction[1];
    }
    let pressure = if ind_pos.is_empty() { 0.0 } else { foreign_near / ind_pos.len() as f32 };
    (push, separation, pressure)
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
                // Eating an animal yields what that animal actually CONTAINS:
                // its tissue, plus a share of the reserves it had banked.
                //
                // A flat rate per part meant a fat, well-fed animal was worth
                // exactly the same as a starving one, and it conjured energy
                // out of nothing rather than moving it up the chain. Measured,
                // predation supplied 1% of the world's energy while causing
                // 44% of its deaths -- killing was common and nearly
                // worthless, so nothing could ever make a living as a
                // predator. Paying out of the prey's own reserves is what
                // makes hunting a fat animal worth more than hunting a thin
                // one, and it is how a food chain actually carries energy:
                // a predator eats what its prey spent its life accumulating.
                let meal = (world.individuals.pixel_count[other] as f32
                    * world.meal_energy_per_part
                    + world.individuals.energy[other].max(0.0) * crate::PREDATION_RESERVE_SHARE)
                    * crate::ENGULF_EFFICIENCY
                    * digestion_multiplier(world, slot);
                world.individuals.energy[slot] += meal;
                world.income_predation += meal as f64;
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
            // A component that has been killed is not necessarily torn off.
            // A blow that only just kills leaves the tissue dead in place --
            // still attached, still carried, useless -- while one that lands
            // with real force in excess of what the part could take severs it
            // outright. Armour is never cut, only killed: a plate turns a
            // blade even when the animal behind it has lost.
            let overkill = damage / crate::BASE_PIXEL_HEALTH.max(0.001);
            let is_armour =
                world.pixels.part_type[other_offset + j] == crate::pixels::PART_ARMOR;
            let tears_off = pixel_severed && !is_armour && overkill >= crate::SEVER_OVERKILL;
            world.fights += 1;
            if damage > 0.0 || pixel_severed {
                world.individuals.ticks_since_fed[slot] = 0; // a landed, damaging hit counts as successful predation
            }
            world.fields.blood[idx] += crate::BLOOD_EMIT_ON_HIT * (damage / crate::BASE_PIXEL_HEALTH).clamp(0.15, 1.0);
            if tears_off {
                let died = remove_pixel(world, other, j as u32);
                if died {
                    kill(world, other);
                    world.deaths_predation += 1;
                    world.fields.blood[idx] += crate::BLOOD_EMIT_ON_DEATH;
                }
            } else if pixel_severed {
                // Killed but held on. The root is the exception: an animal
                // whose head dies is dead, not walking around with a dead
                // head.
                if j == 0 {
                    kill(world, other);
                    world.deaths_predation += 1;
                    world.fields.blood[idx] += crate::BLOOD_EMIT_ON_DEATH;
                } else {
                    world.pixels.dead[other_offset + j] = true;
                    // Left at zero it would re-trigger this branch every hit;
                    // dead tissue simply has no health left to lose.
                    world.pixels.health[other_offset + j] = 0.0;
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
        // Scavenging needs a mouth, like every other way of eating.
        let mouths = world.individuals.part_counts[slot][crate::pixels::PART_MOUTH as usize];
        if mouths == 0 { continue; }
        // A bigger mouth takes a bigger bite. A flat rate meant an enormous
        // carcass took the same thousands of animal-ticks to strip whatever
        // was eating it, so a whale fall sat there apparently untouched.
        let bite_rate = crate::CORPSE_EAT_RATE
            * (1.0 + organ_area(world, slot, crate::pixels::PART_MOUTH) * crate::CORPSE_BITE_PER_MOUTH);
        let size = world.size as f32;
        for c in world.corpses.iter_mut() {
            if c.energy <= 0.0 { continue; }
            // Wrapped: the world is a cylinder, and a carcass just across the
            // seam is right next to you, not half a world away.
            if dist_wrapped(c.root_pos, pos, size) < crate::CORPSE_EAT_RADIUS {
                let bite = c.energy.min(bite_rate);
                c.energy -= bite;
                // Feeding on a carcass costs work, and the cost depends on
                // where it is. High in the column the body is still sinking,
                // so an animal has to swim to stay with it and tear against
                // nothing -- expensive, and it loses the scraps it frees.
                // Settled on the bottom the carcass holds still and can be
                // braced against, so eating is nearly free. That turns a whale
                // fall into a resource whose value depends on depth: a
                // fast-moving animal can reach one early and pay for the
                // privilege, while a patient bottom-dweller waits for it to
                // arrive and eats it cheaply.
                let height = (c.root_pos[1] / size).clamp(0.0, 1.0);
                let effort = crate::CORPSE_FEED_COST_AT_TOP * height * height * bite;
                world.individuals.energy[slot] += bite - effort;
                world.income_scavenge += (bite - effort) as f64;
                world.individuals.ticks_since_fed[slot] = 0;
                world.scavenged += 1;
                // A carcass visibly goes as it is eaten. Its energy was
                // dropping all along, but nothing about it changed on screen
                // until it vanished outright, so a whale fall looked like it
                // was never being consumed. Now the remains shrink to match
                // what is left of them.
                if c.initial_energy > 0.0 && !c.local_shape.is_empty() {
                    let frac = (c.energy / c.initial_energy).clamp(0.0, 1.0);
                    let want = ((c.local_shape.len() as f32) * frac).ceil() as usize;
                    if want < c.local_shape.len() {
                        c.local_shape.truncate(want.max(1));
                    }
                }
                break;
            }
        }
    }
}
