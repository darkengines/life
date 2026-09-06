//! The "individuals" component table (SoA) plus the "brain" (tiny MLP) and
//! growth/reproduction logic that operates across it and the shared pixel
//! arena. Dead slots are tombstoned and recycled (free_slots), never causing
//! a full-population rebuild the way the old `[i for i in individuals if
//! i.alive]` Python pattern did every single tick.
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;

use crate::pixels::PixelArena;

pub const MEM_DIM: usize = 4;
// [food_gx, food_gy, pheromone_gx, pheromone_gy, blood_gx, blood_gy,
//  acid_gx, acid_gy, light_gx, light_gy, forward_food, forward_light,
//  energy_norm, size_norm, kin_similarity_nearest, quorum_local,
//  threat_dx, threat_dy, threat_proximity, prey_dx, prey_dy, prey_proximity,
//  mate_dx, mate_dy, mate_proximity, day_light,
//  home_dx, home_dy, local_territory_mark, conspecific_density,
//  *root_memory]
pub const SENSE_DIM: usize = 30 + MEM_DIM;
pub const HIDDEN_DIM: usize = 12;
// [move_x, move_y, reproduce_urge, fight_urge, crawl_intent, acid_intent,
//  light_intent, *new_memory]
pub const ACT_DIM: usize = 7 + MEM_DIM;
pub const FIGHT_URGE_IDX: usize = 3;

pub struct Individuals {
    pub alive: Vec<bool>,
    pub id: Vec<u64>,
    pub root_pos: Vec<[f32; 2]>,
    pub velocity: Vec<[f32; 2]>,
    pub heading: Vec<f32>,
    pub energy: Vec<f32>,
    pub age: Vec<u32>,
    pub color: Vec<[u8; 3]>,
    pub bend_amplitude: Vec<f32>,
    pub bend_frequency: Vec<f32>,
    pub bend_phase: Vec<f32>,
    pub bite_force: Vec<f32>,
    pub stickiness: Vec<f32>,
    pub toughness: Vec<f32>,
    pub pheromone_emission: Vec<f32>,
    pub acid_secretion: Vec<f32>,
    pub light_emission: Vec<f32>,
    pub crawl_affinity: Vec<f32>,
    pub dig_strength: Vec<f32>, // how effectively this individual can penetrate the sand seafloor -- most start weak
    // Sessile/anemone-like anchoring: resists thrust, gravity, and thermal
    // drift while resting on solid ground (rock or sand), doing nothing at
    // all in open water (there's nothing to anchor TO). Combined with the
    // ALREADY-existing stickiness/attachment mechanic, a lineage that
    // evolves both together gets real anemone behavior for free -- root in
    // place, wait, grab whatever wanders close -- with no new combat code.
    pub anchor_strength: Vec<f32>,
    // Stigmergic territoriality (scent-marking / home-range formation, per
    // real ethology: animals redirect toward a central "den" and treat
    // unfamiliar-but-strong scent concentrations as a deterrent -- see
    // `TERRITORY_EMIT_BASE`'s doc comment in lib.rs). `territoriality` is a
    // heritable trait scaling passive mark-deposit rate (like
    // pheromone_emission); `home_pos` is fixed at birth (this individual's
    // own spawn/birth location, its "den") and never moves or mutates --
    // it's state, not an evolved trait. Nothing here forces homing or
    // defense; both are only POSSIBLE now that the brain can sense
    // direction-to-home and local mark strength (see SENSE_DIM).
    pub territoriality: Vec<f32>,
    pub home_pos: Vec<[f32; 2]>,
    // Janzen-Connell / conspecific-negative-density-dependence: specialist
    // pathogens hurt most exactly where genetically-similar individuals are
    // packed densely, which is what stops any one lineage from simply
    // taking the whole world (a real, well-documented diversity-maintaining
    // mechanism, not a balance knob invented here). This trait blunts that
    // damage -- but carries a real metabolic cost (see
    // DISEASE_RESISTANCE_METABOLIC_COST), so maxing it isn't free and the
    // arms race stays open rather than resolving to "everyone immune".
    pub disease_resistance: Vec<f32>,
    // Cached count of each body-part kind (see pixels.rs). Derived data, not
    // genome: recomputed only when a body actually changes (birth, growth, a
    // part bitten off), so the per-tick effect lookups stay O(1) instead of
    // rescanning every pixel of every individual every tick.
    pub part_counts: Vec<[u8; crate::pixels::PART_KIND_COUNT as usize]>,
    pub memory_transmission_rate: Vec<f32>,
    pub weight_transmission_rate: Vec<f32>,
    // The STABLE ID (never a slot index) of whoever this individual is
    // latched onto, or -1. Storing a slot index here was a real bug: if the
    // target died via any path OTHER than the attachment loop's own kill
    // (a third party's combat hit, starvation's general energy<=0 check,
    // an over-population cull, ...), nothing cleared this attacker's
    // reference -- and once that freed slot got recycled for a brand new,
    // completely unrelated newborn, this attacker would silently start
    // dragging THAT individual around instead, mid-simulation, for no
    // visible reason ("some creatures suddenly rush to some place and
    // die"). An id resolved through `id_to_slot` fails closed instead:
    // free_slot() already removes the dead id from that map, so a stale
    // reference here simply stops resolving to anyone, exactly like
    // parent_id already had to be id-based for the same reason.
    pub attached_to: Vec<i64>,
    pub female: Vec<bool>, // randomly assigned each birth (not inherited/evolved -- a coin flip, like real sex determination)
    pub birth_size: Vec<u32>,       // pixel_count at birth -- fixed for life; body PLAN doesn't change post-birth
    pub size_scale: Vec<f32>,       // uniform inflation of that fixed plan -- juvenile->adult growth is getting
                                     // BIGGER (every joint length scales up), never sprouting new parts
    pub kin_signature: Vec<[f32; crate::KIN_DIM]>, // heritable-with-drift "scent" -- see KIN_DIM's doc comment
    pub parent_id: Vec<i64>,        // stable id (not slot -- slots recycle) of this individual's parent, or -1 for a founder
    pub id_to_slot: std::collections::HashMap<u64, usize>, // for finding a still-alive parent by id in O(1)
    pub ticks_since_fed: Vec<u32>,  // ticks since food/predation/scavenging last succeeded -- reproduction requires this be recent
    pub ticks_since_reproduced: Vec<u32>, // females only: a real recovery period between births, like gestation/nursing
    pub pixel_offset: Vec<u32>,
    pub pixel_count: Vec<u32>,
    pub brain_w1: Vec<f32>,
    pub brain_b1: Vec<f32>,
    pub brain_w2: Vec<f32>,
    pub brain_b2: Vec<f32>,
    pub free_slots: Vec<usize>,
    next_id: u64,
}

fn clip(v: f32, lo: f32, hi: f32) -> f32 { v.max(lo).min(hi) }

fn normal(rng: &mut Pcg64, mean: f32, std: f32) -> f32 {
    if std <= 0.0 { return mean; }
    Normal::new(mean, std).unwrap().sample(rng)
}

/// The one recurring shape every heritable scalar trait in this file takes:
/// drift from the parent's value by gaussian noise, clipped to a valid
/// range. Adding a new evolvable trait is "one field + one call to this",
/// not a bespoke mutation formula copy-pasted at each of the 2-3 places a
/// trait needs to propagate (founder init aside, which is deliberately its
/// own thing -- generation zero has no parent to drift from).
fn inherit_scalar(rng: &mut Pcg64, parent_val: f32, std: f32, lo: f32, hi: f32) -> f32 {
    clip(parent_val + normal(rng, 0.0, std), lo, hi)
}

impl Individuals {
    pub fn with_capacity(cap: usize) -> Self {
        Individuals {
            alive: Vec::with_capacity(cap),
            id: Vec::with_capacity(cap),
            root_pos: Vec::with_capacity(cap),
            velocity: Vec::with_capacity(cap),
            heading: Vec::with_capacity(cap),
            energy: Vec::with_capacity(cap),
            age: Vec::with_capacity(cap),
            color: Vec::with_capacity(cap),
            bend_amplitude: Vec::with_capacity(cap),
            bend_frequency: Vec::with_capacity(cap),
            bend_phase: Vec::with_capacity(cap),
            bite_force: Vec::with_capacity(cap),
            stickiness: Vec::with_capacity(cap),
            toughness: Vec::with_capacity(cap),
            pheromone_emission: Vec::with_capacity(cap),
            acid_secretion: Vec::with_capacity(cap),
            light_emission: Vec::with_capacity(cap),
            crawl_affinity: Vec::with_capacity(cap),
            dig_strength: Vec::with_capacity(cap),
            anchor_strength: Vec::with_capacity(cap),
            territoriality: Vec::with_capacity(cap),
            home_pos: Vec::with_capacity(cap),
            disease_resistance: Vec::with_capacity(cap),
            part_counts: Vec::with_capacity(cap),
            memory_transmission_rate: Vec::with_capacity(cap),
            weight_transmission_rate: Vec::with_capacity(cap),
            attached_to: Vec::with_capacity(cap),
            female: Vec::with_capacity(cap),
            birth_size: Vec::with_capacity(cap),
            size_scale: Vec::with_capacity(cap),
            kin_signature: Vec::with_capacity(cap),
            parent_id: Vec::with_capacity(cap),
            id_to_slot: std::collections::HashMap::with_capacity(cap),
            ticks_since_fed: Vec::with_capacity(cap),
            ticks_since_reproduced: Vec::with_capacity(cap),
            pixel_offset: Vec::with_capacity(cap),
            pixel_count: Vec::with_capacity(cap),
            brain_w1: Vec::new(),
            brain_b1: Vec::new(),
            brain_w2: Vec::new(),
            brain_b2: Vec::new(),
            free_slots: Vec::new(),
            next_id: 0,
        }
    }

    pub fn len(&self) -> usize { self.alive.len() }

    fn alloc_slot(&mut self) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let slot = self.alive.len();
            self.alive.push(false);
            self.id.push(0);
            self.root_pos.push([0.0, 0.0]);
            self.velocity.push([0.0, 0.0]);
            self.heading.push(0.0);
            self.energy.push(0.0);
            self.age.push(0);
            self.color.push([0, 0, 0]);
            self.bend_amplitude.push(0.0);
            self.bend_frequency.push(0.0);
            self.bend_phase.push(0.0);
            self.bite_force.push(0.0);
            self.stickiness.push(0.0);
            self.toughness.push(0.0);
            self.pheromone_emission.push(0.0);
            self.acid_secretion.push(0.0);
            self.light_emission.push(0.0);
            self.crawl_affinity.push(0.0);
            self.dig_strength.push(0.0);
            self.anchor_strength.push(0.0);
            self.territoriality.push(0.0);
            self.home_pos.push([0.0, 0.0]);
            self.disease_resistance.push(0.0);
            self.part_counts.push([0; crate::pixels::PART_KIND_COUNT as usize]);
            self.memory_transmission_rate.push(0.0);
            self.weight_transmission_rate.push(0.0);
            self.attached_to.push(-1);
            self.female.push(false);
            self.birth_size.push(1);
            self.size_scale.push(1.0);
            self.kin_signature.push([0.0; crate::KIN_DIM]);
            self.parent_id.push(-1);
            self.ticks_since_fed.push(0);
            self.ticks_since_reproduced.push(u32::MAX); // never reproduced yet -- cooldown trivially satisfied
            self.pixel_offset.push(0);
            self.pixel_count.push(0);
            self.brain_w1.extend(std::iter::repeat(0.0).take(HIDDEN_DIM * SENSE_DIM));
            self.brain_b1.extend(std::iter::repeat(0.0).take(HIDDEN_DIM));
            self.brain_w2.extend(std::iter::repeat(0.0).take(ACT_DIM * HIDDEN_DIM));
            self.brain_b2.extend(std::iter::repeat(0.0).take(ACT_DIM));
            slot
        }
    }

    /// Resolves an attacker's `attached_to` (a stable id) to its target's
    /// CURRENT slot, or None if it no longer refers to a living individual
    /// -- whether because the target died and nothing overwrote this yet,
    /// or (impossible by construction, but checked anyway) the id simply
    /// isn't in the arena. The one place this lookup should happen; every
    /// caller gets the same fail-closed behavior instead of re-deriving it.
    pub fn resolve_attached_target(&self, attacker_slot: usize) -> Option<usize> {
        let target_id = self.attached_to[attacker_slot];
        if target_id < 0 {
            return None;
        }
        let slot = *self.id_to_slot.get(&(target_id as u64))?;
        if self.alive[slot] { Some(slot) } else { None }
    }

    pub fn free_slot(&mut self, slot: usize) {
        self.alive[slot] = false;
        self.attached_to[slot] = -1;
        self.id_to_slot.remove(&self.id[slot]);
        self.free_slots.push(slot);
    }

    fn brain_w1_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * HIDDEN_DIM * SENSE_DIM;
        &mut self.brain_w1[s..s + HIDDEN_DIM * SENSE_DIM]
    }
    fn brain_b1_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * HIDDEN_DIM;
        &mut self.brain_b1[s..s + HIDDEN_DIM]
    }
    fn brain_w2_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * ACT_DIM * HIDDEN_DIM;
        &mut self.brain_w2[s..s + ACT_DIM * HIDDEN_DIM]
    }
    fn brain_b2_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * ACT_DIM;
        &mut self.brain_b2[s..s + ACT_DIM]
    }

    pub fn brain_w1(&self, slot: usize) -> &[f32] {
        let s = slot * HIDDEN_DIM * SENSE_DIM;
        &self.brain_w1[s..s + HIDDEN_DIM * SENSE_DIM]
    }
    pub fn brain_b1(&self, slot: usize) -> &[f32] {
        let s = slot * HIDDEN_DIM;
        &self.brain_b1[s..s + HIDDEN_DIM]
    }
    pub fn brain_w2(&self, slot: usize) -> &[f32] {
        let s = slot * ACT_DIM * HIDDEN_DIM;
        &self.brain_w2[s..s + ACT_DIM * HIDDEN_DIM]
    }
    pub fn brain_b2(&self, slot: usize) -> &[f32] {
        let s = slot * ACT_DIM;
        &self.brain_b2[s..s + ACT_DIM]
    }

    fn randomize_brain(&mut self, slot: usize, rng: &mut Pcg64) {
        for v in self.brain_w1_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.6); }
        for v in self.brain_b1_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.1); }
        for v in self.brain_w2_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.6); }
        for v in self.brain_b2_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.1); }
        // A founder's random brain otherwise puts fight_urge's baseline
        // (bias + hidden-layer noise) essentially uniformly across its whole
        // range -- with a tanh output and this much weight variance, a
        // healthy fraction of founders come out reflexively aggressive
        // toward literally everything nearby, before any sensory evidence
        // (crowding, an actual threat) is involved. FOUNDERS ONLY: children
        // inherit their bias from an already-selected parent via
        // inherit_brain below, so this is a starting prior on generation
        // zero's temperament, not a permanent cap -- evolution is free to
        // push it back up across generations if aggression keeps paying off.
        self.brain_b2_mut(slot)[FIGHT_URGE_IDX] -= 1.0;
    }

    /// Partial weight transmission: each weight independently transmits
    /// (inherited + mutated) or is freshly reinitialized -- same mechanic
    /// as the Python reference's TinyNN.inherit.
    fn inherit_brain(&mut self, parent_slot: usize, child_slot: usize, transmission_rate: f32, rng: &mut Pcg64) {
        fn mix(rng: &mut Pcg64, val: f32, rate: f32, scale: f32) -> f32 {
            if rng.random::<f32>() < rate {
                val + normal(rng, 0.0, crate::MUTATION_STD * scale)
            } else {
                normal(rng, 0.0, scale)
            }
        }
        let w1: Vec<f32> = self.brain_w1(parent_slot).to_vec();
        let b1: Vec<f32> = self.brain_b1(parent_slot).to_vec();
        let w2: Vec<f32> = self.brain_w2(parent_slot).to_vec();
        let b2: Vec<f32> = self.brain_b2(parent_slot).to_vec();
        for (dst, src) in self.brain_w1_mut(child_slot).iter_mut().zip(w1.iter()) { *dst = mix(rng, *src, transmission_rate, 0.6); }
        for (dst, src) in self.brain_b1_mut(child_slot).iter_mut().zip(b1.iter()) { *dst = mix(rng, *src, transmission_rate, 0.1); }
        for (dst, src) in self.brain_w2_mut(child_slot).iter_mut().zip(w2.iter()) { *dst = mix(rng, *src, transmission_rate, 0.6); }
        for (dst, src) in self.brain_b2_mut(child_slot).iter_mut().zip(b2.iter()) { *dst = mix(rng, *src, transmission_rate, 0.1); }
    }

    pub fn decide(&self, slot: usize, sense: &[f32; SENSE_DIM]) -> [f32; ACT_DIM] {
        let w1 = self.brain_w1(slot);
        let b1 = self.brain_b1(slot);
        let w2 = self.brain_w2(slot);
        let b2 = self.brain_b2(slot);
        let mut h = [0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut acc = b1[j];
            for k in 0..SENSE_DIM {
                acc += w1[j * SENSE_DIM + k] * sense[k];
            }
            h[j] = acc.tanh();
        }
        let mut out = [0f32; ACT_DIM];
        for j in 0..ACT_DIM {
            let mut acc = b2[j];
            for k in 0..HIDDEN_DIM {
                acc += w2[j * HIDDEN_DIM + k] * h[k];
            }
            out[j] = acc.tanh();
        }
        out
    }
}

pub const REST_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Rest-pose (un-bent) integer grid position of every pixel, by walking the
/// parent chain -- matches pixel_world.py's `_rest_grid_positions`.
fn rest_grid_positions(pixels: &PixelArena, offset: u32, count: u32) -> Vec<(i32, i32)> {
    let mut pos = vec![(0i32, 0i32); count as usize];
    for k in 0..count as usize {
        let parent = pixels.parent_idx[offset as usize + k];
        if parent < 0 {
            pos[k] = (0, 0);
        } else {
            let ra = pixels.rest_angle[offset as usize + k];
            let d = angle_to_dir(ra);
            let p = pos[parent as usize];
            pos[k] = (p.0 + d.0, p.1 + d.1);
        }
    }
    pos
}

fn angle_to_dir(angle: f32) -> (i32, i32) {
    (angle.cos().round() as i32, angle.sin().round() as i32)
}
fn dir_to_angle(d: (i32, i32)) -> f32 {
    (d.1 as f32).atan2(d.0 as f32)
}

/// Tip-biased weighted growth: appends one pixel to an existing individual,
/// matching pixel_world.py's `add_one_grown_pixel` exactly (grid-adjacent,
/// strong bias toward extending a tip in its own direction).
/// Recomputes an individual's cached body-part tally. Must be called after
/// anything that changes its pixels: birth, growth, or losing a part in
/// combat. Cheap (bodies are tens of pixels at most) and rare, which is the
/// entire reason the counts are cached rather than derived per tick.
pub fn recompute_part_counts(individuals: &mut Individuals, pixels: &PixelArena, slot: usize) {
    let offset = individuals.pixel_offset[slot] as usize;
    let count = individuals.pixel_count[slot] as usize;
    let mut tally = [0u8; crate::pixels::PART_KIND_COUNT as usize];
    for k in 0..count {
        let t = pixels.part_type[offset + k] as usize;
        if t < tally.len() {
            tally[t] = tally[t].saturating_add(1);
        }
    }
    individuals.part_counts[slot] = tally;
}

pub fn grow_one_pixel(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, slot: usize) -> bool {
    let offset = individuals.pixel_offset[slot];
    let count = individuals.pixel_count[slot];
    let grid_pos = rest_grid_positions(pixels, offset, count);
    let occupied: std::collections::HashSet<(i32, i32)> = grid_pos.iter().cloned().collect();
    let mut has_child = vec![false; count as usize];
    for k in 0..count as usize {
        let p = pixels.parent_idx[offset as usize + k];
        if p >= 0 { has_child[p as usize] = true; }
    }

    let mut candidates: Vec<(usize, (i32, i32))> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    for k in 0..count as usize {
        let is_tip = !has_child[k];
        let own_dir = if pixels.parent_idx[offset as usize + k] >= 0 {
            Some(pixels.rest_angle[offset as usize + k])
        } else {
            None
        };
        let pos = grid_pos[k];
        for &d in REST_DIRECTIONS.iter() {
            let np = (pos.0 + d.0, pos.1 + d.1);
            if occupied.contains(&np) { continue; }
            let extends_tip = is_tip && own_dir.map_or(false, |od| {
                let od_dir = angle_to_dir(od);
                od_dir == d
            });
            candidates.push((k, d));
            weights.push(if extends_tip { 8.0 } else { 1.0 });
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let total: f32 = weights.iter().sum();
    let mut r = rng.random::<f32>() * total;
    let mut choice = 0;
    for (i, w) in weights.iter().enumerate() {
        if r < *w { choice = i; break; }
        r -= w;
    }
    let (parent_local, dir) = candidates[choice];
    let new_angle = dir_to_angle(dir);

    // Reallocate one pixel larger, copy, append, free the old block.
    let new_offset = pixels.allocate(count + 1);
    for k in 0..count as usize {
        pixels.parent_idx[new_offset as usize + k] = pixels.parent_idx[offset as usize + k];
        pixels.rest_angle[new_offset as usize + k] = pixels.rest_angle[offset as usize + k];
        pixels.flex[new_offset as usize + k] = pixels.flex[offset as usize + k];
        pixels.memory[new_offset as usize + k] = pixels.memory[offset as usize + k];
        pixels.storage[new_offset as usize + k] = pixels.storage[offset as usize + k];
        pixels.size[new_offset as usize + k] = pixels.size[offset as usize + k];
        pixels.min_angle[new_offset as usize + k] = pixels.min_angle[offset as usize + k];
        pixels.max_angle[new_offset as usize + k] = pixels.max_angle[offset as usize + k];
        pixels.health[new_offset as usize + k] = pixels.health[offset as usize + k];
        pixels.part_type[new_offset as usize + k] = pixels.part_type[offset as usize + k];
    }
    let parent_flex = pixels.flex[new_offset as usize + parent_local];
    let parent_storage = pixels.storage[new_offset as usize + parent_local];
    let parent_size = pixels.size[new_offset as usize + parent_local];
    let parent_min_angle = pixels.min_angle[new_offset as usize + parent_local];
    let parent_max_angle = pixels.max_angle[new_offset as usize + parent_local];
    pixels.parent_idx[new_offset as usize + count as usize] = parent_local as i32;
    pixels.rest_angle[new_offset as usize + count as usize] = new_angle;
    pixels.flex[new_offset as usize + count as usize] = clip(parent_flex + normal(rng, 0.0, 0.2), 0.0, 1.0);
    pixels.storage[new_offset as usize + count as usize] = clip(parent_storage + normal(rng, 0.0, 0.15), 0.0, 1.0);
    pixels.memory[new_offset as usize + count as usize] = [normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1)];
    let new_size = inherit_scalar(rng, parent_size, crate::PART_SIZE_MUTATION_STD, crate::PART_SIZE_MIN, crate::PART_SIZE_MAX);
    let (mut new_min, mut new_max) = (
        inherit_scalar(rng, parent_min_angle, crate::PART_ANGLE_MUTATION_STD, -std::f32::consts::PI, std::f32::consts::PI),
        inherit_scalar(rng, parent_max_angle, crate::PART_ANGLE_MUTATION_STD, -std::f32::consts::PI, std::f32::consts::PI),
    );
    if new_min > new_max { std::mem::swap(&mut new_min, &mut new_max); }
    // A new part is usually plain body; occasionally it differentiates into
    // an organ. Specialisation being RARE per birth is the point -- an animal
    // with a useful set of organs has to accumulate them over generations and
    // keep paying for them, so it only persists if the combination actually
    // earns its upkeep. Nothing here biases which organ appears.
    pixels.part_type[new_offset as usize + count as usize] =
        if rng.random::<f32>() < crate::PART_DIFFERENTIATION_CHANCE {
            rng.random_range(1..crate::pixels::PART_KIND_COUNT)
        } else {
            crate::pixels::PART_BODY
        };
    pixels.size[new_offset as usize + count as usize] = new_size;
    pixels.min_angle[new_offset as usize + count as usize] = new_min;
    pixels.max_angle[new_offset as usize + count as usize] = new_max;
    pixels.health[new_offset as usize + count as usize] = crate::BASE_PIXEL_HEALTH * new_size;

    if count > 0 {
        pixels.free(offset, count);
    }
    individuals.pixel_offset[slot] = new_offset;
    individuals.pixel_count[slot] = count + 1;
    recompute_part_counts(individuals, pixels, slot);
    true
}

pub fn spawn_founder(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, pos: [f32; 2], color: [u8; 3]) -> usize {
    let slot = individuals.alloc_slot();
    let offset = pixels.allocate(1);
    pixels.parent_idx[offset as usize] = -1;
    pixels.rest_angle[offset as usize] = 0.0;
    pixels.flex[offset as usize] = rng.random_range(0.3..1.0);
    pixels.storage[offset as usize] = rng.random_range(0.0..0.4);
    pixels.memory[offset as usize] = [normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1)];
    let root_size = rng.random_range(0.6..1.4);
    pixels.size[offset as usize] = root_size;
    pixels.min_angle[offset as usize] = -rng.random_range(0.2..2.2);
    pixels.max_angle[offset as usize] = rng.random_range(0.2..2.2);
    pixels.health[offset as usize] = crate::BASE_PIXEL_HEALTH * root_size;

    individuals.alive[slot] = true;
    individuals.id[slot] = individuals.next_id;
    individuals.next_id += 1;
    individuals.root_pos[slot] = pos;
    individuals.velocity[slot] = [0.0, 0.0];
    individuals.heading[slot] = rng.random_range(0.0..std::f32::consts::TAU);
    individuals.energy[slot] = 10.0;
    individuals.age[slot] = 0;
    individuals.color[slot] = color;
    individuals.bend_amplitude[slot] = rng.random_range(0.1..0.6);
    individuals.bend_frequency[slot] = rng.random_range(0.5..2.0);
    individuals.bend_phase[slot] = rng.random_range(0.0..std::f32::consts::TAU);
    individuals.bite_force[slot] = rng.random_range(0.2..1.2);
    individuals.stickiness[slot] = rng.random_range(0.0..1.0);
    individuals.toughness[slot] = rng.random_range(0.0..1.2);
    individuals.pheromone_emission[slot] = rng.random_range(0.2..1.0);
    individuals.acid_secretion[slot] = rng.random_range(0.0..0.3);
    individuals.light_emission[slot] = rng.random_range(0.0..0.3);
    individuals.crawl_affinity[slot] = rng.random_range(0.0..0.3);
    // Weighted toward weak: digging into the sand seafloor is meant to be a
    // real, not-trivial capability to evolve, not something most founders
    // already have for free.
    individuals.dig_strength[slot] = rng.random_range(0.0..0.5) * rng.random_range(0.0..0.5);
    // Same "weighted toward weak" shape as dig_strength -- being sessile
    // is a real commitment (can't chase food or flee), not a free trait.
    individuals.anchor_strength[slot] = rng.random_range(0.0..0.5) * rng.random_range(0.0..0.5);
    // Same "weighted toward weak" shape again -- most founders invest little
    // in marking/defending a home range; a founder's spawn point IS its den.
    individuals.territoriality[slot] = rng.random_range(0.0..0.6) * rng.random_range(0.0..0.6);
    individuals.home_pos[slot] = pos;
    individuals.disease_resistance[slot] = rng.random_range(0.0..0.5);
    individuals.memory_transmission_rate[slot] = rng.random_range(0.2..0.8);
    individuals.weight_transmission_rate[slot] = rng.random_range(0.2..0.8);
    individuals.attached_to[slot] = -1;
    individuals.female[slot] = rng.random::<bool>();
    // Explicit resets: slots are RECYCLED (a dead individual's old field
    // values persist until overwritten), not just freshly zero-initialized,
    // so anything not set here would silently leak the previous occupant's
    // state into a brand new individual.
    individuals.ticks_since_fed[slot] = 0;
    individuals.ticks_since_reproduced[slot] = u32::MAX;
    individuals.size_scale[slot] = 1.0;
    for d in 0..crate::KIN_DIM {
        individuals.kin_signature[slot][d] = rng.random_range(-1.0..1.0);
    }
    individuals.parent_id[slot] = -1;
    individuals.pixel_offset[slot] = offset;
    individuals.pixel_count[slot] = 1;
    pixels.part_type[offset as usize] = crate::pixels::PART_BODY;
    recompute_part_counts(individuals, pixels, slot);
    individuals.randomize_brain(slot, rng);
    individuals.id_to_slot.insert(individuals.id[slot], slot);
    slot
}

/// Full reproduction: copy parent's pixel data (with partial memory
/// transmission), grow one new pixel, mutate all traits -- matches
/// pixel_world.py's `mutated_child_at`.
pub fn reproduce(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, parent: usize) -> usize {
    let child = individuals.alloc_slot();
    let parent_offset = individuals.pixel_offset[parent];
    let parent_count = individuals.pixel_count[parent];
    let mem_rate = clip(individuals.memory_transmission_rate[parent] + normal(rng, 0.0, 0.05), 0.0, 1.0);

    let new_offset = pixels.allocate(parent_count);
    for k in 0..parent_count as usize {
        pixels.parent_idx[new_offset as usize + k] = pixels.parent_idx[parent_offset as usize + k];
        pixels.rest_angle[new_offset as usize + k] = pixels.rest_angle[parent_offset as usize + k];
        pixels.flex[new_offset as usize + k] = pixels.flex[parent_offset as usize + k];
        pixels.storage[new_offset as usize + k] = pixels.storage[parent_offset as usize + k];
        pixels.part_type[new_offset as usize + k] = pixels.part_type[parent_offset as usize + k];
        pixels.size[new_offset as usize + k] = pixels.size[parent_offset as usize + k];
        pixels.min_angle[new_offset as usize + k] = pixels.min_angle[parent_offset as usize + k];
        pixels.max_angle[new_offset as usize + k] = pixels.max_angle[parent_offset as usize + k];
        pixels.health[new_offset as usize + k] = crate::BASE_PIXEL_HEALTH * pixels.size[parent_offset as usize + k]; // a child starts unwounded, even if the parent is scarred
        let parent_mem = pixels.memory[parent_offset as usize + k];
        let mut mem = [0f32; 4];
        for d in 0..4 {
            mem[d] = if rng.random::<f32>() < mem_rate { parent_mem[d] } else { normal(rng, 0.0, 0.1) };
        }
        pixels.memory[new_offset as usize + k] = mem;
    }

    let color = if rng.random::<f32>() < crate::COLOR_MUTATION_RATE {
        let base = individuals.color[parent];
        [0, 1, 2].map(|i| clip(base[i] as f32 + normal(rng, 0.0, crate::COLOR_MUTATION_STD), 20.0, 255.0) as u8)
    } else {
        individuals.color[parent]
    };

    individuals.alive[child] = true;
    individuals.id[child] = individuals.next_id;
    individuals.next_id += 1;
    individuals.root_pos[child] = [
        individuals.root_pos[parent][0] + normal(rng, 0.0, crate::BIRTH_OFFSET_STD),
        individuals.root_pos[parent][1] + normal(rng, 0.0, crate::BIRTH_OFFSET_STD),
    ];
    individuals.velocity[child] = [0.0, 0.0];
    individuals.heading[child] = individuals.heading[parent] + normal(rng, 0.0, 0.2);
    individuals.energy[child] = 10.0;
    individuals.age[child] = 0;
    individuals.color[child] = color;
    individuals.bend_amplitude[child] = clip(individuals.bend_amplitude[parent] + normal(rng, 0.0, 0.08), 0.0, 1.5);
    individuals.bend_frequency[child] = clip(individuals.bend_frequency[parent] + normal(rng, 0.0, 0.15), 0.2, 3.0);
    individuals.bend_phase[child] = individuals.bend_phase[parent] + normal(rng, 0.0, 0.3);
    individuals.bite_force[child] = clip(individuals.bite_force[parent] + normal(rng, 0.0, 0.15), 0.0, 3.0);
    individuals.stickiness[child] = clip(individuals.stickiness[parent] + normal(rng, 0.0, 0.08), 0.0, 1.0);
    individuals.toughness[child] = clip(individuals.toughness[parent] + normal(rng, 0.0, 0.15), 0.0, 3.0);
    individuals.pheromone_emission[child] = clip(individuals.pheromone_emission[parent] + normal(rng, 0.0, 0.08), 0.0, 1.0);
    individuals.acid_secretion[child] = clip(individuals.acid_secretion[parent] + normal(rng, 0.0, 0.05), 0.0, 1.0);
    individuals.light_emission[child] = clip(individuals.light_emission[parent] + normal(rng, 0.0, 0.05), 0.0, 1.0);
    individuals.crawl_affinity[child] = clip(individuals.crawl_affinity[parent] + normal(rng, 0.0, 0.08), 0.0, 1.0);
    individuals.dig_strength[child] = clip(individuals.dig_strength[parent] + normal(rng, 0.0, 0.05), 0.0, 1.5);
    individuals.anchor_strength[child] = inherit_scalar(rng, individuals.anchor_strength[parent], 0.06, 0.0, 1.5);
    individuals.territoriality[child] = inherit_scalar(rng, individuals.territoriality[parent], 0.08, 0.0, 1.5);
    individuals.disease_resistance[child] = inherit_scalar(rng, individuals.disease_resistance[parent], 0.07, 0.0, 1.5);
    // A child's home range is ITS OWN birth site, not inherited from the
    // parent's -- exactly like a real animal's den is wherever it was born/
    // settled, not a copied coordinate. This is why offspring naturally
    // disperse into new home ranges instead of all "belonging" to one
    // ancestral spot forever.
    individuals.home_pos[child] = individuals.root_pos[child];
    individuals.memory_transmission_rate[child] = mem_rate;
    individuals.weight_transmission_rate[child] = clip(individuals.weight_transmission_rate[parent] + normal(rng, 0.0, 0.05), 0.0, 1.0);
    individuals.attached_to[child] = -1;
    individuals.female[child] = rng.random::<bool>();
    for d in 0..crate::KIN_DIM {
        individuals.kin_signature[child][d] = clip(individuals.kin_signature[parent][d] + normal(rng, 0.0, crate::KIN_MUTATION_STD), -2.0, 2.0);
    }
    individuals.parent_id[child] = individuals.id[parent] as i64;
    individuals.pixel_offset[child] = new_offset;
    individuals.pixel_count[child] = parent_count;

    let wt_rate = individuals.weight_transmission_rate[child];
    individuals.inherit_brain(parent, child, wt_rate, rng);

    recompute_part_counts(individuals, pixels, child);
    grow_one_pixel(individuals, pixels, rng, child); // one body-plan variation at birth, matches Python
    individuals.birth_size[child] = individuals.pixel_count[child];
    individuals.size_scale[child] = 1.0; // starts at the same baseline size as its birth plan, regardless of how big the parent had inflated to
    individuals.ticks_since_fed[child] = 0;
    individuals.ticks_since_reproduced[child] = u32::MAX;
    individuals.id_to_slot.insert(individuals.id[child], child);
    child
}
