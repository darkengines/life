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
/// Width of the signal each component passes to its children. Small on
/// purpose: the capacity of this network comes from having MANY units spread
/// through the body, not from any one of them being wide.
pub const NEURITE_DIM: usize = 4;
/// Bodies up to this many components run the per-part network entirely on the
/// stack. Sized past what anything in this world actually grows to, so the
/// heap path is a safety net rather than a code path.
pub const NEURITE_STACK: usize = 96;

/// A random little transform for a newly grown component.
///
/// Scaled so signals neither die out nor saturate as they travel down a long
/// body -- a chain of thirty parts multiplies thirty of these in sequence, and
/// getting the scale wrong makes every deep body either silent or pinned at
/// +/-1. Random rather than zero because the entire point is that a new part
/// contributes a NEW feature the readout can pick up, which a zero matrix
/// could never do.
pub fn random_neurite(rng: &mut Pcg64) -> [f32; NEURITE_DIM * NEURITE_DIM] {
    let mut w = [0.0f32; NEURITE_DIM * NEURITE_DIM];
    let scale = 1.0 / (NEURITE_DIM as f32).sqrt();
    for v in w.iter_mut() {
        *v = normal(rng, 0.0, scale);
    }
    w
}

/// Inherit a parent component's transform with drift, so a lineage keeps what
/// its body has learned to compute instead of re-rolling it every birth.
pub fn inherit_neurite(
    rng: &mut Pcg64,
    src: &[f32; NEURITE_DIM * NEURITE_DIM],
) -> [f32; NEURITE_DIM * NEURITE_DIM] {
    let mut w = *src;
    for v in w.iter_mut() {
        *v = (*v + normal(rng, 0.0, crate::NEURITE_MUTATION_STD)).clamp(-3.0, 3.0);
    }
    w
}
// [food_gx, food_gy, pheromone_gx, pheromone_gy, blood_gx, blood_gy,
//  acid_gx, acid_gy, light_gx, light_gy, forward_food, forward_light,
//  energy_norm, size_norm, kin_similarity_nearest, quorum_local,
//  threat_dx, threat_dy, threat_proximity, prey_dx, prey_dy, prey_proximity,
//  mate_dx, mate_dy, mate_proximity, day_light,
//  home_dx, home_dy, local_territory_mark, conspecific_density,
//  shelter_here, shelter_dx, shelter_dy,
//  drift_forward, drift_lateral, speed_norm, *root_memory]
//
// The three proprioceptive channels are what make "learning to swim"
// possible at all. Thrust is perfectly locked to a given body (R=1.000 across
// headings) but the direction a body pushes relative to where it points
// varies completely from body to body (R=0.259 across fifteen plans, several
// of them pushing almost exactly backwards). An animal had no way to sense
// that, so swimming tail-first was not a behaviour it could learn out of --
// the feedback did not exist. Reporting its own velocity in its own frame
// leaves the physics alone and makes the problem observable.
//
// The shelter channels close a real gap: creatures could not perceive terrain
// at all, discovering rock only by colliding with it. Sheltering measurably
// pays -- small bodies survive 42% inside the deep reef against 20% for large
// ones -- but nothing could navigate toward it, so that payoff was
// unreachable and could never select for anything.
pub const SENSE_DIM: usize = 36 + MEM_DIM;

/// Names for every sense channel, in order, so an inspector can show what a
/// number actually means rather than an index. Kept beside SENSE_DIM so the
/// two cannot drift apart unnoticed.
pub const SENSE_LABELS: [&str; SENSE_DIM] = [
    "food gradient x", "food gradient y",
    "pheromone x", "pheromone y",
    "blood x", "blood y",
    "acid x", "acid y",
    "light x", "light y",
    "food ahead", "light ahead",
    "energy", "body size", "kin similarity", "crowd (any kind)",
    "threat dx", "threat dy", "threat near",
    "prey dx", "prey dy", "prey near",
    "mate dx", "mate dy", "mate near",
    "daylight",
    "home dx", "home dy", "territory mark", "crowd (own kind)",
    "shelter here", "shelter dx", "shelter dy",
    "drift forward", "drift sideways", "speed",
    "memory 0", "memory 1", "memory 2", "memory 3",
];

/// Names for every action output, in order.
pub const ACT_LABELS: [&str; ACT_DIM] = [
    "move x", "move y", "eat urge", "fight urge", "mate urge",
    "pheromone emit", "acid emit", "swim effort", "turn",
    "memory 0", "memory 1", "memory 2", "memory 3",
];
pub const HIDDEN_DIM: usize = 12;
/// Width of the shared perception latent. Every individual's raw senses are
/// compressed through ONE encoder shared by the whole world, and each
/// individual then decides from that latent with its OWN evolved decoder.
///
/// The reason is measured: evolved brains were performing no better than
/// random ones at steering toward food. Each individual was having to
/// rediscover, by mutation alone, how to extract meaning from 34 raw sensory
/// channels -- roughly 560 weights per animal, with credit assignment coming
/// only from whether it happened to survive. Sharing the perception stage
/// means feature extraction is learned once across the whole population's
/// experience, while the decision stays private and evolvable, which is what
/// keeps behavioural diversity intact. It also cuts each individual's evolved
/// parameters roughly in half, so mutation has far less to search.
pub const LATENT_DIM: usize = 12;
// [move_x, move_y, reproduce_urge, fight_urge, crawl_intent, acid_intent,
//  light_intent, swim_effort, *new_memory]
//
// swim_effort is what finally gives a brain real agency over its own body.
// Until it existed, undulation was driven purely by the evolved constants
// bend_amplitude/bend_frequency, so a creature swam at a fixed genetic rate
// and the brain could only STEER -- it could not accelerate toward prey,
// sprint away from a predator, or stop to conserve energy. That is almost
// certainly why fleeing never evolved at all: there was no way to flee.
pub const ACT_DIM: usize = 9 + MEM_DIM;
pub const SWIM_EFFORT_IDX: usize = 7;
/// Asymmetry the brain imposes on its own bending wave. This is how a real
/// swimmer turns: it curves its body, the curved body pushes water
/// asymmetrically, and the resulting TORQUE rotates it. Heading used to be a
/// variable the brain simply assigned, with the body rotated to match, so
/// orientation was imposed rather than earned and had no physical
/// relationship to how the body was actually moving. Measured consequence:
/// the angle between where a creature pointed and where it actually went
/// averaged ~91 degrees, i.e. steering did not control movement at all --
/// which caps how much any brain can ever matter.
pub const TURN_BIAS_IDX: usize = 8;
pub const MEMORY_OUT_IDX: usize = 9;
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
    // How hard this individual is currently swimming, set from its own brain
    // output each tick (see SWIM_EFFORT_IDX). Multiplies the undulation
    // amplitude in the forward-kinematics pass, so it feeds straight through
    // to thrust. State, not genome.
    pub swim_gain: Vec<f32>,
    /// Constant curvature currently held in the body, set by the brain.
    pub turn_curvature: Vec<f32>,
    /// Rotation rate, integrated from fluid torque -- NOT assigned. A body
    /// keeps spinning until the water stops it.
    pub angular_velocity: Vec<f32>,
    /// Angle, in the body's own frame, from the root toward the body's centre
    /// of mass -- its anterior/posterior axis.
    ///
    /// Without this, `heading` merely ROTATED the body: a creature whose parts
    /// happened to grow off to one side moved sideways relative to the
    /// direction it was nominally facing, and the offset differed per
    /// individual according to the shape it happened to grow. Pooled over a
    /// population that is exactly a uniform random offset, which is why the
    /// angle between heading and actual movement measured ~79 degrees and
    /// stayed flat no matter how fast a creature was going. Subtracting this
    /// makes `heading` mean what it claims: the direction the body points.
    pub axis_offset: Vec<f32>,
    /// The angle, in this body's own frame, at which it actually pushes.
    ///
    /// Measured from the animal's own propulsion rather than inferred from its
    /// shape, and this distinction is the difference between an animal that can
    /// swim somewhere and one that cannot. `axis_offset` derives "forward" from
    /// where the body's mass sits relative to its root, which turned out not to
    /// predict thrust at all: across fifteen body plans the angle between where
    /// a body POINTS and where it PUSHES scattered almost uniformly (consistency
    /// R = 0.259, individual offsets reaching +/-170 degrees). Steering toward a
    /// target therefore sent many animals directly away from it, and the
    /// population-wide alignment between intent and motion measured NEGATIVE.
    ///
    /// This is not handing an animal a correct body. Which way a body pushes is
    /// a fact about that body, and an animal that has been swimming for a while
    /// unavoidably knows it -- every real swimmer calibrates intent against
    /// sensed motion, and none of them are born knowing their own hydrodynamics
    /// either. What it removes is an engine-side error: "forward" was being
    /// defined by centroid geometry when the only definition that means anything
    /// is which way the thing actually goes.
    pub thrust_offset: Vec<f32>,
    /// Body density relative to the water, heritable. 1.0 is neutral: the
    /// animal neither rises nor sinks and can go wherever it swims.
    ///
    /// There was no buoyancy at all, which turned out to be the single largest
    /// obstacle to anything ever swimming anywhere. Measured by forcing every
    /// animal's intent to one direction and watching where they actually went:
    /// commanded DOWN they complied (alignment +0.76, 78% within 45 degrees),
    /// commanded UP they went down anyway (-0.81), and commanded sideways they
    /// went down (near zero). They were not steering badly, they were falling,
    /// and no amount of intelligence can steer a stone.
    ///
    /// Every animal that lives in open water solves this, and solves it the
    /// same way: match your density to the water. Fish carry a swim bladder,
    /// sharks an oil-rich liver, cephalopods pump ammonium. Making it a
    /// heritable trait rather than a constant means an animal can also choose
    /// NOT to be neutral -- a bottom-dweller that sinks costs nothing to stay
    /// down, and something hunting near the surface can float.
    pub buoyancy: Vec<f32>,
    /// Which way this body actually TURNS when it curves itself, learned from
    /// its own rotation.
    ///
    /// Steering here works by holding a curvature: a bent body pushes water
    /// asymmetrically and the resulting torque rotates it. But whether a given
    /// curvature rotates an animal left or right depends on where its mass and
    /// its surfaces happen to sit, and these bodies are grown by mutation, so
    /// for a good fraction of them the relationship is INVERTED. Such an animal
    /// commands a left turn, rotates right, commands harder, rotates further
    /// wrong -- a control loop with the sign flipped does not merely fail, it
    /// actively runs away from its target.
    ///
    /// Measured with the brain removed and intent forced in one direction,
    /// animals managed 18-28% within 45 degrees of where they were told to go,
    /// in every direction equally. This is the same lesson as thrust_offset one
    /// level down: the map from command to outcome is a property of the
    /// individual body, and an animal has to discover it. Correlating commanded
    /// curvature against the rotation that followed is exactly what an efference
    /// copy is for, and every real motor system does it.
    pub steer_sign: Vec<f32>,
    // Cached count of each body-part kind (see pixels.rs). Derived data, not
    // genome: recomputed only when a body actually changes (birth, growth, a
    // part bitten off), so the per-tick effect lookups stay O(1) instead of
    // rescanning every pixel of every individual every tick.
    pub part_counts: Vec<[u8; crate::pixels::PART_KIND_COUNT as usize]>,
    // Reward accumulated since this individual was last sampled into the
    // experience log. Reproduction is the event worth rewarding -- it is the
    // only thing that actually propagates a policy -- and crediting it at the
    // moment it happens is what a replay-trained network needs, since energy
    // deltas alone never explain why an action mattered. Zeroed on sampling.
    pub pending_reward: Vec<f32>,
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
    pub female: Vec<bool>, // assigned each birth, biased toward the rarer sex -- see `reproduce`
    /// Fraction of the living population that is female, refreshed each tick.
    /// Feeds the sex-ratio correction above.
    pub female_share: f32,
    /// How hermaphroditic this animal is, heritable, 0 to 1.
    ///
    /// Deliberately a trait rather than a global switch. Whether separate sexes
    /// or hermaphroditism is the better strategy is not a fact about animals in
    /// general, it is a fact about the density they live at: when mates are
    /// plentiful, separate sexes are cheaper, and when finding one is the
    /// binding constraint, being able to breed with whoever you meet is worth
    /// far more than specialising. That is Baker's law, and it is why sessile
    /// and low-density marine invertebrates are so overwhelmingly
    /// hermaphroditic while dense schooling animals are not.
    ///
    /// Making it evolvable means this world resolves the question for itself,
    /// under its own densities, instead of being told the answer -- and if
    /// density changes, the answer is free to change with it. It is not free:
    /// carrying both sets of machinery costs upkeep, which is exactly why
    /// separate sexes persist wherever mates are easy to find.
    pub hermaphrodite: Vec<f32>,
    pub birth_size: Vec<u32>,       // pixel_count at birth -- fixed for life; body PLAN doesn't change post-birth
    pub size_scale: Vec<f32>,       // uniform inflation of that fixed plan -- juvenile->adult growth is getting
                                     // BIGGER (every joint length scales up), never sprouting new parts
    pub kin_signature: Vec<[f32; crate::KIN_DIM]>, // heritable-with-drift "scent" -- see KIN_DIM's doc comment
    pub parent_id: Vec<i64>,        // stable id (not slot -- slots recycle) of this individual's parent, or -1 for a founder
    pub id_to_slot: std::collections::HashMap<u64, usize>, // for finding a still-alive parent by id in O(1)
    /// Ticks of reduced selection pressure remaining, for an animal born with
    /// a body plan meaningfully different from its parent's.
    ///
    /// Co-optimising a body and the brain that drives it has a known failure
    /// mode: a mutated morphology is judged while running a controller
    /// inherited for the OLD body, so it underperforms for reasons that have
    /// nothing to do with whether the new shape is any good, and morphological
    /// change is punished on arrival. Morphology then converges early on
    /// whatever was safe, which is exactly what "the dominant creatures are
    /// not very effective" looks like from outside. Cheney, Bongard,
    /// SunSpiral & Lipson's answer is to protect recent morphological
    /// innovation briefly so control has time to readapt to the new body.
    /// Cached storage capacity before size scaling. Like exposed surface, it
    /// is a function of the body PLAN -- girth, tissue type and the evolved
    /// storage trait -- so walking the whole component list for it three times
    /// per animal per tick was recomputing a constant.
    pub storage_capacity_base: Vec<f32>,
    /// Cached exposed surface, recomputed only when the body changes.
    pub exposed_surface: Vec<f32>,
    pub innovation_protect: Vec<u32>,
    pub ticks_since_fed: Vec<u32>,  // ticks since food/predation/scavenging last succeeded -- reproduction requires this be recent
    pub ticks_since_reproduced: Vec<u32>, // females only: a real recovery period between births, like gestation/nursing
    pub pixel_offset: Vec<u32>,
    pub pixel_count: Vec<u32>,
    pub brain_w1: Vec<f32>,
    pub brain_b1: Vec<f32>,
    pub brain_w2: Vec<f32>,
    /// Readout from what the BODY computed (see body_signal) into actions.
    /// Evolved per individual like the rest of the decoder.
    pub brain_w3: Vec<f32>,
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
            storage_capacity_base: Vec::with_capacity(cap),
            exposed_surface: Vec::with_capacity(cap),
            innovation_protect: Vec::with_capacity(cap),
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
            swim_gain: Vec::with_capacity(cap),
            turn_curvature: Vec::with_capacity(cap),
            angular_velocity: Vec::with_capacity(cap),
            axis_offset: Vec::with_capacity(cap),
            thrust_offset: Vec::with_capacity(cap),
            buoyancy: Vec::with_capacity(cap),
            steer_sign: Vec::with_capacity(cap),
            part_counts: Vec::with_capacity(cap),
            pending_reward: Vec::with_capacity(cap),
            memory_transmission_rate: Vec::with_capacity(cap),
            weight_transmission_rate: Vec::with_capacity(cap),
            attached_to: Vec::with_capacity(cap),
            female: Vec::with_capacity(cap),
            female_share: 0.5,
            hermaphrodite: Vec::with_capacity(cap),
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
            brain_w3: Vec::new(),
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
            self.swim_gain.push(1.0);
            self.turn_curvature.push(0.0);
            self.angular_velocity.push(0.0);
            self.axis_offset.push(0.0);
            self.thrust_offset.push(0.0);
            self.buoyancy.push(1.0);
            self.steer_sign.push(1.0);
            self.part_counts.push([0; crate::pixels::PART_KIND_COUNT as usize]);
            self.pending_reward.push(0.0);
            self.memory_transmission_rate.push(0.0);
            self.weight_transmission_rate.push(0.0);
            self.attached_to.push(-1);
            self.female.push(false);
            self.hermaphrodite.push(0.0);
            self.birth_size.push(1);
            self.size_scale.push(1.0);
            self.kin_signature.push([0.0; crate::KIN_DIM]);
            self.parent_id.push(-1);
            self.storage_capacity_base.push(0.0);
            self.exposed_surface.push(0.0);
            self.innovation_protect.push(0);
            self.ticks_since_fed.push(0);
            self.ticks_since_reproduced.push(u32::MAX); // never reproduced yet -- cooldown trivially satisfied
            self.pixel_offset.push(0);
            self.pixel_count.push(0);
            self.brain_w1.extend(std::iter::repeat(0.0).take(HIDDEN_DIM * LATENT_DIM));
            self.brain_b1.extend(std::iter::repeat(0.0).take(HIDDEN_DIM));
            self.brain_w2.extend(std::iter::repeat(0.0).take(ACT_DIM * HIDDEN_DIM));
            self.brain_w3.extend(std::iter::repeat(0.0).take(ACT_DIM * NEURITE_DIM));
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

    pub fn brain_w1_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * HIDDEN_DIM * LATENT_DIM;
        &mut self.brain_w1[s..s + HIDDEN_DIM * LATENT_DIM]
    }
    pub fn brain_b1_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * HIDDEN_DIM;
        &mut self.brain_b1[s..s + HIDDEN_DIM]
    }
    pub fn brain_w2_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * ACT_DIM * HIDDEN_DIM;
        &mut self.brain_w2[s..s + ACT_DIM * HIDDEN_DIM]
    }
    pub fn brain_w3_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * ACT_DIM * NEURITE_DIM;
        &mut self.brain_w3[s..s + ACT_DIM * NEURITE_DIM]
    }
    pub fn brain_b2_mut(&mut self, slot: usize) -> &mut [f32] {
        let s = slot * ACT_DIM;
        &mut self.brain_b2[s..s + ACT_DIM]
    }

    pub fn brain_w1(&self, slot: usize) -> &[f32] {
        let s = slot * HIDDEN_DIM * LATENT_DIM;
        &self.brain_w1[s..s + HIDDEN_DIM * LATENT_DIM]
    }
    pub fn brain_b1(&self, slot: usize) -> &[f32] {
        let s = slot * HIDDEN_DIM;
        &self.brain_b1[s..s + HIDDEN_DIM]
    }
    pub fn brain_w2(&self, slot: usize) -> &[f32] {
        let s = slot * ACT_DIM * HIDDEN_DIM;
        &self.brain_w2[s..s + ACT_DIM * HIDDEN_DIM]
    }
    pub fn brain_w3(&self, slot: usize) -> &[f32] {
        let s = slot * ACT_DIM * NEURITE_DIM;
        &self.brain_w3[s..s + ACT_DIM * NEURITE_DIM]
    }
    pub fn brain_b2(&self, slot: usize) -> &[f32] {
        let s = slot * ACT_DIM;
        &self.brain_b2[s..s + ACT_DIM]
    }

    pub fn randomize_brain(&mut self, slot: usize, rng: &mut Pcg64) {
        for v in self.brain_w1_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.6); }
        for v in self.brain_b1_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.1); }
        for v in self.brain_w2_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.6); }
        for v in self.brain_w3_mut(slot).iter_mut() { *v = normal(rng, 0.0, 0.4); }
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
        let w3: Vec<f32> = self.brain_w3(parent_slot).to_vec();
        let b2: Vec<f32> = self.brain_b2(parent_slot).to_vec();
        for (dst, src) in self.brain_w1_mut(child_slot).iter_mut().zip(w1.iter()) { *dst = mix(rng, *src, transmission_rate, 0.6); }
        for (dst, src) in self.brain_b1_mut(child_slot).iter_mut().zip(b1.iter()) { *dst = mix(rng, *src, transmission_rate, 0.1); }
        for (dst, src) in self.brain_w2_mut(child_slot).iter_mut().zip(w2.iter()) { *dst = mix(rng, *src, transmission_rate, 0.6); }
        for (dst, src) in self.brain_w3_mut(child_slot).iter_mut().zip(w3.iter()) { *dst = mix(rng, *src, transmission_rate, 0.4); }
        for (dst, src) in self.brain_b2_mut(child_slot).iter_mut().zip(b2.iter()) { *dst = mix(rng, *src, transmission_rate, 0.1); }
    }

    /// Shared perception: raw senses -> latent. One encoder for the entire
    /// world, so this is where a GPU-trained representation would be dropped
    /// in; nothing else about the individual changes when it improves.
    pub fn encode(sense: &[f32; SENSE_DIM], enc_w: &[f32], enc_b: &[f32]) -> [f32; LATENT_DIM] {
        let mut z = [0f32; LATENT_DIM];
        for j in 0..LATENT_DIM {
            let mut acc = enc_b[j];
            let row = j * SENSE_DIM;
            for k in 0..SENSE_DIM {
                acc += enc_w[row + k] * sense[k];
            }
            z[j] = acc.tanh();
        }
        z
    }

    /// Private decision: latent -> action, using this individual's own
    /// evolved weights. Perception is shared; what to DO about it is not.
    /// The same computation as `decide`, but reporting every intermediate
    /// stage: the shared perception latent, the individual's own hidden layer,
    /// and its action outputs. For inspecting one animal's brain live, so what
    /// it is actually computing can be looked at rather than guessed at.
    pub fn decide_traced(
        &self,
        pixels: &PixelArena,
        slot: usize,
        sense: &[f32; SENSE_DIM],
        enc_w: &[f32],
        enc_b: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let z = Self::encode(sense, enc_w, enc_b);
        let w1 = self.brain_w1(slot);
        let b1 = self.brain_b1(slot);
        let w2 = self.brain_w2(slot);
        let b2 = self.brain_b2(slot);
        let mut h = vec![0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut acc = b1[j];
            for k in 0..LATENT_DIM {
                acc += w1[j * LATENT_DIM + k] * z[k];
            }
            h[j] = acc.tanh();
        }
        let mut h_arr = [0f32; HIDDEN_DIM];
        h_arr.copy_from_slice(&h);
        let body = self.body_signal(pixels, slot, &h_arr);
        let w3 = self.brain_w3(slot);
        let mut out = vec![0f32; ACT_DIM];
        for j in 0..ACT_DIM {
            let mut acc = b2[j];
            for k in 0..HIDDEN_DIM {
                acc += w2[j * HIDDEN_DIM + k] * h[k];
            }
            for k in 0..NEURITE_DIM {
                acc += w3[j * NEURITE_DIM + k] * body[k];
            }
            out[j] = acc.tanh();
        }
        (z.to_vec(), h, out, body.to_vec())
    }

    /// Runs the body's network and WRITES each component's drive back into the
    /// arena, so the next tick's kinematics actuate every organ individually.
    ///
    /// This is the fine control: rather than the brain setting one effort dial
    /// for the whole animal, each component's own unit decides how hard that
    /// component beats. Run in the sequential phase because it mutates the
    /// arena; a one-tick lag between deciding and moving is of no consequence
    /// at these timescales.
    pub fn apply_part_drive(
        individuals: &Individuals,
        pixels: &mut PixelArena,
        slot: usize,
        h: &[f32; HIDDEN_DIM],
    ) {
        let offset = individuals.pixel_offset[slot] as usize;
        let count = individuals.pixel_count[slot] as usize;
        if count == 0 { return; }
        let mut sig = vec![[0.0f32; NEURITE_DIM]; count];
        for d in 0..NEURITE_DIM {
            sig[0][d] = h[d % HIDDEN_DIM];
        }
        pixels.drive[offset] = sig[0][0];
        for k in 1..count {
            let parent = pixels.parent_idx[offset + k];
            if parent < 0 { continue; }
            let p = parent as usize;
            if p >= k { continue; }
            // Dead tissue conducts nothing and actuates nothing.
            if pixels.dead[offset + k] {
                pixels.drive[offset + k] = -1.0;
                continue;
            }
            let w = &pixels.neurite[offset + k];
            let b = &pixels.memory[offset + k];
            for row in 0..NEURITE_DIM {
                let mut acc = b[row % MEM_DIM];
                for col in 0..NEURITE_DIM {
                    acc += w[row * NEURITE_DIM + col] * sig[p][col];
                }
                sig[k][row] = acc.tanh();
            }
            pixels.drive[offset + k] = sig[k][0];
        }
    }

    /// Runs the body's own network: the hidden layer seeds the root, and every
    /// component transforms what its parent passes down into what it passes to
    /// its children. Returns the pooled result, which is what the body as a
    /// whole computed.
    ///
    /// Parts are always stored after their parent (a child is appended, and
    /// re-indexing preserves that), so one forward pass in storage order is a
    /// correct traversal of the tree -- no recursion and no scratch ordering.
    pub fn body_signal(
        &self,
        pixels: &PixelArena,
        slot: usize,
        h: &[f32; HIDDEN_DIM],
    ) -> [f32; NEURITE_DIM] {
        let offset = self.pixel_offset[slot] as usize;
        let count = self.pixel_count[slot] as usize;
        if count == 0 {
            return [0.0; NEURITE_DIM];
        }
        // Same stack-first buffer as apply_part_drive, and for the same reason:
        // this runs once per animal per tick, so an allocation here is one for
        // every animal alive on every tick.
        let mut stack = [[0.0f32; NEURITE_DIM]; NEURITE_STACK];
        let mut heap: Vec<[f32; NEURITE_DIM]>;
        let sig: &mut [[f32; NEURITE_DIM]] = if count <= NEURITE_STACK {
            &mut stack[..count]
        } else {
            heap = vec![[0.0f32; NEURITE_DIM]; count];
            &mut heap[..]
        };
        // The root is fed from the individual's own hidden layer: this is
        // where perception enters the body.
        for d in 0..NEURITE_DIM {
            sig[0][d] = h[d % HIDDEN_DIM];
        }
        let mut pooled = [0.0f32; NEURITE_DIM];
        for d in 0..NEURITE_DIM {
            pooled[d] += sig[0][d];
        }
        for k in 1..count {
            let parent = pixels.parent_idx[offset + k];
            if parent < 0 { continue; }
            let p = parent as usize;
            if p >= k { continue; } // defensive: storage order should prevent this
            // A dead component conducts nothing, so everything beyond it is
            // cut off from the brain while still being carried around. Losing
            // the use of a whole limb without losing the limb is exactly the
            // kind of injury that ought to matter.
            if pixels.dead[offset + k] {
                continue;
            }
            let w = &pixels.neurite[offset + k];
            let b = &pixels.memory[offset + k];
            for row in 0..NEURITE_DIM {
                let mut acc = b[row % MEM_DIM];
                for col in 0..NEURITE_DIM {
                    acc += w[row * NEURITE_DIM + col] * sig[p][col];
                }
                sig[k][row] = acc.tanh();
            }
            for d in 0..NEURITE_DIM {
                pooled[d] += sig[k][d];
            }
        }
        // Pooled by sum over sqrt(n), not by mean. A plain average of signed
        // values from many components cancels itself: a thirty-two part animal
        // was measured pooling to about 0.03, so the body's contribution
        // vanished exactly as the body got interesting. Dividing by sqrt(n)
        // keeps the magnitude roughly constant with size -- the standard
        // variance-preserving scaling -- so a large body speaks as loudly as a
        // small one and says something more complicated.
        let inv = 1.0 / (count as f32).sqrt();
        for d in 0..NEURITE_DIM {
            pooled[d] = (pooled[d] * inv).tanh();
        }
        pooled
    }

    pub fn decide(
        &self,
        pixels: &PixelArena,
        slot: usize,
        sense: &[f32; SENSE_DIM],
        enc_w: &[f32],
        enc_b: &[f32],
    ) -> ([f32; ACT_DIM], [f32; HIDDEN_DIM]) {
        let z = Self::encode(sense, enc_w, enc_b);
        let w1 = self.brain_w1(slot);
        let b1 = self.brain_b1(slot);
        let w2 = self.brain_w2(slot);
        let b2 = self.brain_b2(slot);
        let mut h = [0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut acc = b1[j];
            for k in 0..LATENT_DIM {
                acc += w1[j * LATENT_DIM + k] * z[k];
            }
            h[j] = acc.tanh();
        }
        // What the BODY computed, folded into the decision. This is the whole
        // point of putting a unit in every component: an animal's capacity to
        // respond grows as it grows, and a part added by mutation contributes
        // a new signal immediately rather than being dead weight until some
        // separate controller happens to evolve a use for it.
        let body = self.body_signal(pixels, slot, &h);
        let w3 = self.brain_w3(slot);
        let mut out = [0f32; ACT_DIM];
        for j in 0..ACT_DIM {
            let mut acc = b2[j];
            for k in 0..HIDDEN_DIM {
                acc += w2[j * HIDDEN_DIM + k] * h[k];
            }
            for k in 0..NEURITE_DIM {
                acc += w3[j * NEURITE_DIM + k] * body[k];
            }
            out[j] = acc.tanh();
        }
        // The hidden layer comes back too: it seeds the per-component drive,
        // which is applied in the sequential phase where the arena is mutable.
        (out, h)
    }
}

pub const REST_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Rest-pose (un-bent) integer grid position of every pixel, by walking the
/// parent chain -- matches pixel_world.py's `_rest_grid_positions`.
/// How much of this body is actually in contact with the water, as a sum over
/// components of the fraction of each one's perimeter that is exposed.
///
/// This is the number feeding should depend on, and getting it wrong is what
/// made body plans meaningless. Feeding was charged per COMPONENT, so intake
/// scaled as N while upkeep scaled as N^0.6 -- benefit steeper than cost,
/// which means more tissue always pays, anywhere, in any arrangement. Nothing
/// an animal grew could ever be useless, so nothing was ever pruned, and shape
/// carried no information at all.
///
/// Real suspension feeders do not work that way. Filtration rate scales as
/// roughly W^0.66-0.70 across bivalves, ascidians, crustaceans, polychaetes
/// and jellyfish, because the filtering surface is a SURFACE: gill area grows
/// as L^2 while mass grows as L^3. Metabolism meanwhile scales as W^0.75 --
/// steeper than intake -- which is what gives real animals a finite optimum
/// size instead of an unbounded incentive to grow.
///
/// In two dimensions the analogue of surface is perimeter, and this is where
/// it becomes interesting: a solid blob of N parts has a perimeter of about
/// sqrt(N), while a branched or feathery body of the same mass has a perimeter
/// of nearly N. So an animal that wants to eat plankton has to be built like
/// something that eats plankton -- open, branched, high-surface -- and buried
/// interior tissue earns nothing while still costing upkeep. That is exactly
/// the pressure that keeps real anatomy free of useless bulk, and it makes
/// topology something selection can finally see.
pub fn exposed_surface(pixels: &PixelArena, offset: u32, count: u32) -> f32 {
    if count == 0 { return 0.0; }
    let grid = rest_grid_positions(pixels, offset, count);
    // Sorted slice and binary search rather than a hash set: bodies are tens
    // of parts, where the hashing and the allocation both cost more than the
    // search they replace.
    let mut occupied: Vec<(i32, i32)> = grid.clone();
    occupied.sort_unstable();
    let mut total = 0.0;
    for (k, &(x, y)) in grid.iter().enumerate() {
        // Dead tissue strains nothing: it is carried, not used.
        if pixels.dead[offset as usize + k] { continue; }
        let open = REST_DIRECTIONS
            .iter()
            .filter(|(dx, dy)| occupied.binary_search(&(x + dx, y + dy)).is_err())
            .count() as f32;
        // Weighted by how big the component is: a broad frond presents more
        // surface to the water than a small one.
        let g = crate::pixels::girth(pixels, offset as usize + k);
        total += (open / 4.0) * g;
    }
    total
}

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

/// Weighted growth: appends one pixel to an existing individual. Growth is
/// grid-adjacent and prefers extending an existing tip in its own direction;
/// the strength of that morphology pressure lives in lib.rs so it can be
/// measured and tuned explicitly.
/// Recomputes an individual's cached body-part tally. Must be called after
/// anything that changes its pixels: birth, growth, or losing a part in
/// combat. Cheap (bodies are tens of pixels at most) and rare, which is the
/// entire reason the counts are cached rather than derived per tick.
/// Nudges an individual's decoder toward the learned baseline policy.
///
/// This is how GPU learning reaches the population: not by overwriting minds,
/// but by shifting what a newborn STARTS from, the way instinct is inherited.
/// The blend is partial and mutation still applies afterwards, so individual
/// variation -- the raw material selection works on -- is preserved rather
/// than collapsed onto one shared behaviour.
pub fn distill_policy(
    individuals: &mut Individuals,
    slot: usize,
    policy: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    rate: f32,
) {
    let (pw1, pb1, pw2, pb2) = policy;
    let k = rate.clamp(0.0, 1.0);
    for (dst, src) in individuals.brain_w1_mut(slot).iter_mut().zip(pw1.iter()) {
        *dst += (*src - *dst) * k;
    }
    for (dst, src) in individuals.brain_b1_mut(slot).iter_mut().zip(pb1.iter()) {
        *dst += (*src - *dst) * k;
    }
    for (dst, src) in individuals.brain_w2_mut(slot).iter_mut().zip(pw2.iter()) {
        *dst += (*src - *dst) * k;
    }
    for (dst, src) in individuals.brain_b2_mut(slot).iter_mut().zip(pb2.iter()) {
        *dst += (*src - *dst) * k;
    }
}

/// Recomputes the body's anterior axis from its REST pose: the direction from
/// the root to the centroid of all its parts. Must be recomputed whenever the
/// body plan changes, since growing a limb to one side moves the centre of
/// mass and therefore what "forwards" means for this animal.
/// Removes one LEAF part (a part with no children), keeping the body a
/// connected tree. Used by birth anomalies so lineages can shed structure as
/// well as gain it.
/// Two components fuse into one.
///
/// Growth could only ever ADD parts, so a big organ had to be built out of many
/// small ones sitting next to each other -- there was no way to arrive at a
/// single large belly, a single wide jaw, or a claw, only at a cluster of
/// average-sized pieces that happened to share a type. Real development does
/// the other thing constantly: tissue fuses, plates coalesce, paired
/// primordia join on the midline. Fusion is how anything gets a big single
/// structure rather than a heap of small ones.
///
/// Restricted to a LEAF merging into its parent, and deliberately so: fusing a
/// mid-body component would orphan everything beyond it and have to re-parent
/// them, which shifts every descendant one cell inward and can drop two parts
/// onto the same lattice square. A leaf simply disappears into what it was
/// attached to, and nothing else about the body moves.
///
/// The survivor takes the type of whichever contributed more substance, so a
/// mouth absorbing plain flank becomes a bigger mouth, while flank absorbing a
/// small mouth stays flank. Merged parts may exceed the ordinary per-part size
/// ceiling -- that is the entire point, since otherwise fusion buys nothing.
pub fn merge_leaf_into_parent(
    individuals: &mut Individuals,
    pixels: &mut PixelArena,
    slot: usize,
    victim: u32,
) {
    let offset = individuals.pixel_offset[slot];
    let count = individuals.pixel_count[slot];
    if count <= 2 || victim == 0 || victim >= count {
        return;
    }
    let parent = pixels.parent_idx[(offset + victim) as usize];
    if parent < 0 {
        return;
    }
    let (vi, pi) = ((offset + victim) as usize, (offset + parent as u32) as usize);
    let v_size = pixels.size[vi];
    let p_size = pixels.size[pi];
    // Substance is conserved-ish: some is lost in the joining, as it is in any
    // real fusion, so merging is not a free way to manufacture size.
    pixels.size[pi] = (p_size + v_size * crate::MERGE_SIZE_TRANSFER)
        .min(crate::MERGED_PART_SIZE_MAX);
    if v_size > p_size {
        pixels.part_type[pi] = pixels.part_type[vi];
        pixels.neurite[pi] = pixels.neurite[vi];
        pixels.phase_offset[pi] = pixels.phase_offset[vi];
        pixels.freq_mult[pi] = pixels.freq_mult[vi];
    }
    // Storage and flex blend by how much each side brought.
    let total = (p_size + v_size).max(1e-4);
    pixels.storage[pi] = (pixels.storage[pi] * p_size + pixels.storage[vi] * v_size) / total;
    pixels.flex[pi] = (pixels.flex[pi] * p_size + pixels.flex[vi] * v_size) / total;
    pixels.health[pi] = crate::BASE_PIXEL_HEALTH * pixels.size[pi];
    remove_leaf(individuals, pixels, slot, victim);
}

pub fn remove_leaf(individuals: &mut Individuals, pixels: &mut PixelArena, slot: usize, victim: u32) {
    let offset = individuals.pixel_offset[slot];
    let count = individuals.pixel_count[slot];
    if count <= 1 || victim == 0 || victim >= count {
        return;
    }
    let new_offset = pixels.allocate(count - 1);
    let mut w = 0usize;
    let mut remap = vec![-1i32; count as usize];
    for k in 0..count as usize {
        if k as u32 == victim {
            continue;
        }
        remap[k] = w as i32;
        let (src, dst) = (offset as usize + k, new_offset as usize + w);
        pixels.rest_angle[dst] = pixels.rest_angle[src];
        pixels.flex[dst] = pixels.flex[src];
        pixels.memory[dst] = pixels.memory[src];
        pixels.neurite[dst] = pixels.neurite[src];
        pixels.dead[dst] = pixels.dead[src];
        pixels.phase_offset[dst] = pixels.phase_offset[src];
        pixels.freq_mult[dst] = pixels.freq_mult[src];
        pixels.drive[dst] = pixels.drive[src];
        pixels.storage[dst] = pixels.storage[src];
        pixels.size[dst] = pixels.size[src];
        pixels.min_angle[dst] = pixels.min_angle[src];
        pixels.max_angle[dst] = pixels.max_angle[src];
        pixels.health[dst] = pixels.health[src];
        pixels.part_type[dst] = pixels.part_type[src];
        pixels.symmetric[dst] = pixels.symmetric[src];
        pixels.mirror_sign[dst] = pixels.mirror_sign[src];
        w += 1;
    }
    // re-point parents through the remap
    w = 0;
    for k in 0..count as usize {
        if k as u32 == victim {
            continue;
        }
        let old_parent = pixels.parent_idx[offset as usize + k];
        pixels.parent_idx[new_offset as usize + w] =
            if old_parent < 0 { -1 } else { remap[old_parent as usize] };
        w += 1;
    }
    pixels.free(offset, count);
    individuals.pixel_offset[slot] = new_offset;
    individuals.pixel_count[slot] = count - 1;
    recompute_part_counts(individuals, pixels, slot);
    recompute_axis_offset(individuals, pixels, slot);
}

pub fn recompute_axis_offset(individuals: &mut Individuals, pixels: &PixelArena, slot: usize) {
    let offset = individuals.pixel_offset[slot];
    let count = individuals.pixel_count[slot];
    if count < 2 {
        individuals.axis_offset[slot] = 0.0;
        return;
    }
    let grid = rest_grid_positions(pixels, offset, count);
    let (mut cx, mut cy) = (0.0f32, 0.0f32);
    for &(gx, gy) in grid.iter() {
        cx += gx as f32;
        cy += gy as f32;
    }
    cx /= count as f32;
    cy /= count as f32;
    // grid[0] is the root by construction.
    let (rx, ry) = (grid[0].0 as f32, grid[0].1 as f32);
    let (dx, dy) = (cx - rx, cy - ry);
    // Forward is from the body's mass toward the ROOT -- the root leads, the
    // rest trails behind it. Defining it the other way round (root toward
    // centroid) meant "forward" pointed into the body, so every creature swam
    // tail-first, which is exactly what was observed by eye. Measured with the
    // brain frozen out: thrust is perfectly locked to the body (R=1.000) and
    // pointed +162.6 degrees away from the old convention, and across fifteen
    // different body plans every one that produced meaningful thrust clustered
    // at 170-180 degrees. One convention error, one sign.
    individuals.axis_offset[slot] = if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
        0.0
    } else {
        (-dy).atan2(-dx)
    };
}

/// Give an individual entirely fresh heritable traits, owing nothing to any
/// parent. Used only by the heredity control.
pub fn randomize_brain_and_traits(individuals: &mut Individuals, rng: &mut Pcg64, slot: usize) {
    individuals.randomize_brain(slot, rng);
    individuals.bend_amplitude[slot] = rng.random_range(0.1..1.2);
    individuals.bend_frequency[slot] = rng.random_range(0.05..0.5);
    individuals.bend_phase[slot] = rng.random_range(0.0..std::f32::consts::TAU);
    individuals.bite_force[slot] = rng.random_range(0.0..2.0);
    individuals.stickiness[slot] = rng.random_range(0.0..1.0);
    individuals.buoyancy[slot] = rng.random_range(crate::BUOYANCY_MIN..crate::BUOYANCY_MAX);
    individuals.hermaphrodite[slot] = rng.random::<f32>();
    individuals.thrust_offset[slot] = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
    individuals.steer_sign[slot] = if rng.random::<bool>() { 1.0 } else { -1.0 };
}

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
    // Exposed surface is a property of the body PLAN, which only changes when
    // the body does -- and this is the one place every such change already
    // routes through. It was being recomputed from scratch every tick for
    // every animal, building a hash set of lattice coordinates each time, for
    // a number that had not moved since the animal was born.
    individuals.exposed_surface[slot] =
        exposed_surface(pixels, individuals.pixel_offset[slot], individuals.pixel_count[slot]);
    let mut cap_base = crate::ENERGY_CAP_BASE;
    for k in 0..count {
        let g = crate::pixels::girth(pixels, offset + k);
        let area = g * g / crate::PART_AREA_REF;
        let kind = pixels.part_type[offset + k] as usize;
        let per = crate::pixels::PART_STORAGE[kind]
            + pixels.storage[offset + k] * crate::ENERGY_CAP_PER_STORAGE_TRAIT;
        cap_base += per * area * crate::ENERGY_CAP_SCALE;
    }
    individuals.storage_capacity_base[slot] = cap_base;
}

pub fn grow_one_pixel(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, slot: usize) -> bool {
    grow_one_pixel_weighted(individuals, pixels, rng, slot, crate::GROWTH_STRAIGHT_TIP_WEIGHT)
}

/// As `grow_one_pixel`, with the tip-extension bias supplied explicitly so
/// it can be swept without a rebuild.
pub fn grow_one_pixel_weighted(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, slot: usize, tip_weight: f32) -> bool {
    let offset = individuals.pixel_offset[slot];
    let count = individuals.pixel_count[slot];
    let grid_pos = rest_grid_positions(pixels, offset, count);
    // Sorted slice + binary search instead of a hash set. Growth is the single
    // most expensive thing in the tick -- reproduction measured 2.829 ms of a
    // 7.876 ms tick, 36% of the whole simulation -- and a body is tens of
    // parts, a size at which hashing and allocating a set cost more than the
    // searches they are meant to accelerate.
    let mut occupied: Vec<(i32, i32)> = grid_pos.clone();
    occupied.sort_unstable();
    let mut has_child = vec![false; count as usize];
    for k in 0..count as usize {
        let p = pixels.parent_idx[offset as usize + k];
        if p >= 0 { has_child[p as usize] = true; }
    }

    // One buffer of (site, weight) rather than two parallel Vecs, both
    // starting empty and reallocating as they grow.
    let mut candidates: Vec<(usize, (i32, i32))> = Vec::with_capacity(count as usize * 2);
    let mut weights: Vec<f32> = Vec::with_capacity(count as usize * 2);
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
            if occupied.binary_search(&np).is_ok() { continue; }
            let extends_tip = is_tip && own_dir.map_or(false, |od| {
                let od_dir = angle_to_dir(od);
                od_dir == d
            });
            // A symmetric node grows SIDEWAYS by preference. Bilateral
            // symmetry is not just a flag on a part, it is a developmental
            // program that puts paired appendages out along the flanks of a
            // body axis -- and only a lateral growth (dy != 0) can pair at
            // all, since a direction lying on the axis is its own mirror.
            // Without this the flag was almost inert: 35% of founders carried
            // it and it was faithfully inherited, yet only ~12% of bodies
            // ever showed a mirrored pair, because tip extension runs along
            // the axis where pairing is impossible by construction. So
            // symmetric nodes were being handed a trait they could not
            // express.
            // Interior sites are deliberately cheap, and this is what
            // decides whether an animal has a body plan or is a shrub.
            // Every part offers a free neighbouring cell, so the number of
            // interior sites grows with the body while the number of tips
            // stays at two: a twenty-part body presented about thirty
            // interior candidates at weight 1.0 against two tips at weight
            // 2.0, so barely a tenth of all growth extended the animal and
            // the rest packed on sideways. Measured on the live world, the
            // longest chain through a body was only 0.4 of its part count
            // and 37% of nodes carried more than one child -- a radial bush,
            // which is why they looked ineffective and why turning did
            // nothing useful. Weighting tips over interior sites gives a
            // trunk that elongates, with paired appendages branching off it
            // at symmetric nodes: a bilaterian, rather than a lichen.
            let lateral_pair = pixels.symmetric[offset as usize + k] && d.1 != 0;
            let mut w = if extends_tip {
                tip_weight
            } else if lateral_pair {
                crate::SYMMETRY_LATERAL_WEIGHT
            } else {
                crate::INTERIOR_SITE_WEIGHT
            };
            // A node that already carries children is a poor place for yet
            // another one; without this, thick trunks fatten into slabs.
            let existing_children = pixels.parent_idx[offset as usize..(offset + count) as usize]
                .iter()
                .filter(|&&pp| pp == k as i32)
                .count();
            if existing_children >= 2 {
                w *= crate::CROWDED_NODE_PENALTY;
            }
            candidates.push((k, d));
            weights.push(w);
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

    // Bilateral symmetry: growing on a symmetric node emits a mirrored twin
    // on the same node, reflected across the body's long axis (dy -> -dy).
    // A direction lying ON that axis is its own mirror, so it stays single.
    // The twin is only added if its cell is actually free, so symmetry never
    // overwrites existing anatomy.
    let mirror_dir = (dir.0, -dir.1);
    let mirror_pos = {
        let base = grid_pos[parent_local];
        (base.0 + mirror_dir.0, base.1 + mirror_dir.1)
    };
    let make_pair = pixels.symmetric[offset as usize + parent_local]
        && dir.1 != 0
        && occupied.binary_search(&mirror_pos).is_err();
    let added: u32 = if make_pair { 2 } else { 1 };

    // Reallocate for the new part(s), copy, append, free the old block.
    let new_offset = pixels.allocate(count + added);
    for k in 0..count as usize {
        pixels.parent_idx[new_offset as usize + k] = pixels.parent_idx[offset as usize + k];
        pixels.rest_angle[new_offset as usize + k] = pixels.rest_angle[offset as usize + k];
        pixels.flex[new_offset as usize + k] = pixels.flex[offset as usize + k];
        pixels.memory[new_offset as usize + k] = pixels.memory[offset as usize + k];
        pixels.neurite[new_offset as usize + k] = pixels.neurite[offset as usize + k];
        pixels.dead[new_offset as usize + k] = pixels.dead[offset as usize + k];
        pixels.phase_offset[new_offset as usize + k] = pixels.phase_offset[offset as usize + k];
        pixels.freq_mult[new_offset as usize + k] = pixels.freq_mult[offset as usize + k];
        pixels.drive[new_offset as usize + k] = pixels.drive[offset as usize + k];
        pixels.storage[new_offset as usize + k] = pixels.storage[offset as usize + k];
        pixels.size[new_offset as usize + k] = pixels.size[offset as usize + k];
        pixels.min_angle[new_offset as usize + k] = pixels.min_angle[offset as usize + k];
        pixels.max_angle[new_offset as usize + k] = pixels.max_angle[offset as usize + k];
        pixels.health[new_offset as usize + k] = pixels.health[offset as usize + k];
        pixels.part_type[new_offset as usize + k] = pixels.part_type[offset as usize + k];
        pixels.symmetric[new_offset as usize + k] = pixels.symmetric[offset as usize + k];
        pixels.mirror_sign[new_offset as usize + k] = pixels.mirror_sign[offset as usize + k];
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
    // A brand new part brings a brand new random feature -- see random_neurite.
    pixels.neurite[new_offset as usize + count as usize] = random_neurite(rng);
    pixels.dead[new_offset as usize + count as usize] = false;
    // A new appendage gets its own beat. Drawn near the parent's so a limb
    // stays roughly coherent with what it grew from, but free to drift -- a
    // quarter-cycle lag between neighbouring joints is exactly what turns a
    // flat wave into a circular stroke, and it has to be reachable by
    // mutation for anything to find it.
    let par_phase = pixels.phase_offset[new_offset as usize + parent_local];
    let par_freq = pixels.freq_mult[new_offset as usize + parent_local];
    pixels.phase_offset[new_offset as usize + count as usize] =
        par_phase + normal(rng, 0.0, crate::PART_PHASE_MUTATION_STD);
    pixels.freq_mult[new_offset as usize + count as usize] =
        (par_freq + normal(rng, 0.0, crate::PART_FREQ_MUTATION_STD)).clamp(0.25, 4.0);
    pixels.drive[new_offset as usize + count as usize] = 0.0;
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
    // Symmetry is itself heritable: a new part usually matches its parent
    // node, but can flip, so bilateral body plans can both arise and be lost.
    let parent_symmetric = pixels.symmetric[new_offset as usize + parent_local];
    pixels.symmetric[new_offset as usize + count as usize] =
        if rng.random::<f32>() < crate::SYMMETRY_FLIP_CHANCE { !parent_symmetric } else { parent_symmetric };

    if make_pair {
        // The twin is the same KIND of part with the same anatomy, placed at
        // the reflected angle -- a left/right pair, not two random growths.
        let t = new_offset as usize + count as usize;      // the part just written
        let m = new_offset as usize + count as usize + 1;  // its mirror
        pixels.parent_idx[m] = parent_local as i32;
        pixels.rest_angle[m] = dir_to_angle(mirror_dir);
        pixels.flex[m] = pixels.flex[t];
        pixels.storage[m] = pixels.storage[t];
        pixels.memory[m] = [normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1)];
        // A mirrored twin is the same organ, so it computes the same thing.
        pixels.neurite[m] = pixels.neurite[t];
        pixels.dead[m] = false;
        // A mirrored pair beats together, like a real pair of fins.
        pixels.phase_offset[m] = pixels.phase_offset[t];
        pixels.freq_mult[m] = pixels.freq_mult[t];
        pixels.drive[m] = 0.0;
        pixels.part_type[m] = pixels.part_type[t];
        pixels.size[m] = pixels.size[t];
        // Hinge limits mirror too, so the pair bends symmetrically rather
        // than one side flapping while the other is locked.
        pixels.min_angle[m] = -pixels.max_angle[t];
        pixels.max_angle[m] = -pixels.min_angle[t];
        pixels.health[m] = pixels.health[t];
        pixels.symmetric[m] = pixels.symmetric[t];
        // The twin undulates in mirror image, which is what makes the pair's
        // sideways thrust cancel and the body swim straight.
        pixels.mirror_sign[m] = -pixels.mirror_sign[t];
    }

    if count > 0 {
        pixels.free(offset, count);
    }
    individuals.pixel_offset[slot] = new_offset;
    individuals.pixel_count[slot] = count + added;
    recompute_part_counts(individuals, pixels, slot);
    recompute_axis_offset(individuals, pixels, slot);
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
    pixels.neurite[offset as usize] = random_neurite(rng);
    pixels.dead[offset as usize] = false;
    pixels.phase_offset[offset as usize] = 0.0;
    pixels.freq_mult[offset as usize] = 1.0;
    pixels.drive[offset as usize] = 0.0;
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
    // Founders vary, so both strategies are present for selection to work on
    // rather than one having to be invented from nothing.
    individuals.hermaphrodite[slot] = rng.random::<f32>() * 0.6;
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
    pixels.symmetric[offset as usize] = rng.random::<f32>() < crate::SYMMETRY_FOUNDER_CHANCE;
    recompute_part_counts(individuals, pixels, slot);
    recompute_axis_offset(individuals, pixels, slot);
    individuals.randomize_brain(slot, rng);
    individuals.id_to_slot.insert(individuals.id[slot], slot);
    slot
}

/// Full reproduction: copy parent's pixel data (with partial memory
/// transmission), grow one new pixel, mutate all traits -- matches
/// pixel_world.py's `mutated_child_at`.
pub fn reproduce(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, parent: usize, tip_weight: f32) -> usize {
    reproduce_with(individuals, pixels, rng, parent, tip_weight, false)
}

/// As `reproduce`, but `scramble` replaces the child's heritable traits with
/// random ones instead of the parent's -- the control that isolates whether
/// heredity is doing any work. See World::scramble_inheritance.
pub fn reproduce_with(individuals: &mut Individuals, pixels: &mut PixelArena, rng: &mut Pcg64, parent: usize, tip_weight: f32, scramble: bool) -> usize {
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
        pixels.symmetric[new_offset as usize + k] = pixels.symmetric[parent_offset as usize + k];
        pixels.mirror_sign[new_offset as usize + k] = pixels.mirror_sign[parent_offset as usize + k];
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
        pixels.neurite[new_offset as usize + k] =
            inherit_neurite(rng, &pixels.neurite[parent_offset as usize + k]);
        // Offspring are born whole: scars are not inherited.
        pixels.dead[new_offset as usize + k] = false;
        pixels.phase_offset[new_offset as usize + k] =
            pixels.phase_offset[parent_offset as usize + k]
                + normal(rng, 0.0, crate::PART_PHASE_MUTATION_STD);
        pixels.freq_mult[new_offset as usize + k] =
            (pixels.freq_mult[parent_offset as usize + k]
                + normal(rng, 0.0, crate::PART_FREQ_MUTATION_STD))
                .clamp(0.25, 4.0);
        pixels.drive[new_offset as usize + k] = 0.0;
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
    // Sex allocation is biased toward whichever sex is RARE.
    //
    // A fair coin per birth is fine in a large population and lethal in a small
    // one: with a handful of animals left the ratio drifts, and a world of four
    // came out all male -- no mates, no births, certain extinction from a run of
    // coin flips rather than from anything about the animals. Sampling noise
    // should not be the thing that ends a lineage.
    //
    // Biasing toward the rarer sex is not a patch, it is Fisher's principle
    // made mechanical: the rarer sex has higher expected reproductive value, so
    // over-producing it is exactly what selection favours, and plenty of real
    // species implement it directly through environmental or social sex
    // determination. The bias is proportional to the skew, so at parity this is
    // still a fair coin and only a genuinely lopsided population feels it.
    let female_share = individuals.female_share.clamp(0.05, 0.95);
    let p_female = (0.5 + (0.5 - female_share) * crate::SEX_RATIO_CORRECTION).clamp(0.05, 0.95);
    individuals.female[child] = rng.random::<f32>() < p_female;
    individuals.hermaphrodite[child] = clip(
        individuals.hermaphrodite[parent] + normal(rng, 0.0, crate::HERMAPHRODITE_MUTATION_STD),
        0.0,
        1.0,
    );
    for d in 0..crate::KIN_DIM {
        individuals.kin_signature[child][d] = clip(individuals.kin_signature[parent][d] + normal(rng, 0.0, crate::KIN_MUTATION_STD), -2.0, 2.0);
    }
    individuals.parent_id[child] = individuals.id[parent] as i64;
    individuals.pixel_offset[child] = new_offset;
    individuals.pixel_count[child] = parent_count;

    let wt_rate = individuals.weight_transmission_rate[child];
    individuals.inherit_brain(parent, child, wt_rate, rng);

    recompute_part_counts(individuals, pixels, child);
    recompute_axis_offset(individuals, pixels, child);
    let parent_count_before = individuals.pixel_count[child];
    // Birth anomalies: the body plan can gain a part, gain a small burst of
    // them, or LOSE one. Only ever appending meant morphology crept outward in
    // unit steps and could never simplify, so shapes could not really explore.
    // Fusion: two components join into one larger one. Checked before the
    // other anomalies because it is the only one that makes a body SIMPLER
    // while making its parts BIGGER -- every other path either adds pieces or
    // sheds them, and neither of those can produce a single large organ.
    // Fusion is an INDEPENDENT event, not an alternative to growing.
    //
    // Chaining it onto the grow/shed choice meant a birth that fused could not
    // also grow, so every fusion was a net loss of one component and mean body
    // size fell from the forties to thirteen within a few thousand ticks.
    // Fusing and growing are different processes and a real developing body
    // does both in the same generation.
    let will_merge = rng.random::<f32>() < crate::ANOMALY_MERGE_CHANCE;
    if rng.random::<f32>() < crate::ANOMALY_LOSE_PART_CHANCE
        && individuals.pixel_count[child] > 2
    {
        // Shed a leaf part (one with no children), so the body stays a
        // connected tree rather than losing a whole branch it was carrying.
        let off = individuals.pixel_offset[child] as usize;
        let n = individuals.pixel_count[child] as usize;
        let mut has_child = vec![false; n];
        for k in 0..n {
            let par = pixels.parent_idx[off + k];
            if par >= 0 {
                has_child[par as usize] = true;
            }
        }
        let leaves: Vec<usize> = (1..n).filter(|&k| !has_child[k]).collect();
        if !leaves.is_empty() {
            let victim = leaves[rng.random_range(0..leaves.len())];
            remove_leaf(individuals, pixels, child, victim as u32);
        }
    } else {
        let burst = if rng.random::<f32>() < crate::ANOMALY_BURST_CHANCE {
            rng.random_range(2..=crate::ANOMALY_BURST_MAX)
        } else {
            1
        };
        for _ in 0..burst {
            grow_one_pixel_weighted(individuals, pixels, rng, child, tip_weight);
        }
    }
    // A body plan that actually changed gets a grace period; one that merely
    // copied its parent gets nothing. Protection is for INNOVATION, not for
    // being young -- handing it to every newborn would just be a uniform
    // discount and would select for nothing.
    individuals.innovation_protect[child] =
        if individuals.pixel_count[child] != parent_count_before {
            crate::INNOVATION_PROTECT_TICKS
        } else {
            0
        };
    // Fusion, applied after growth so the two are independent.
    if will_merge && individuals.pixel_count[child] > 3 {
        let off = individuals.pixel_offset[child] as usize;
        let n = individuals.pixel_count[child] as usize;
        let mut has_child = vec![false; n];
        for k in 0..n {
            let par = pixels.parent_idx[off + k];
            if par >= 0 {
                has_child[par as usize] = true;
            }
        }
        let leaves: Vec<usize> = (1..n).filter(|&k| !has_child[k]).collect();
        if !leaves.is_empty() {
            let victim = leaves[rng.random_range(0..leaves.len())];
            merge_leaf_into_parent(individuals, pixels, child, victim as u32);
        }
    }
    // Inherit the parent's sense of which way it swims.
    //
    // An offspring's body is its parent's plus a mutation, so the parent's
    // measurement is a good prior and starting from zero throws it away. That
    // matters more than it sounds: the estimate takes on the order of a hundred
    // ticks to converge, so a newborn beginning blind spends a real fraction of
    // its life steering the wrong way -- and at these lifespans a large part of
    // the population is always newborn. This is ordinary inherited motor prior,
    // not inherited skill: the estimate still updates from the animal's own
    // propulsion, and a body plan that mutated enough will correct away from it.
    individuals.thrust_offset[child] = individuals.thrust_offset[parent];
    individuals.steer_sign[child] = individuals.steer_sign[parent];
    // Buoyancy is heritable with drift, so lineages can specialise by depth:
    // neutral for open water, heavy for the bottom, light for the surface.
    individuals.buoyancy[child] = clip(
        individuals.buoyancy[parent] + normal(rng, 0.0, crate::BUOYANCY_MUTATION_STD),
        crate::BUOYANCY_MIN,
        crate::BUOYANCY_MAX,
    );
    if scramble {
        // Break heredity, and ONLY heredity: the child is a viable animal with
        // a viable body, its traits simply owe nothing to its parent. Anything
        // else changed here would confound the comparison.
        randomize_brain_and_traits(individuals, rng, child);
    }
    individuals.birth_size[child] = individuals.pixel_count[child];
    individuals.size_scale[child] = 1.0; // starts at the same baseline size as its birth plan, regardless of how big the parent had inflated to
    individuals.ticks_since_fed[child] = 0;
    individuals.ticks_since_reproduced[child] = u32::MAX;
    individuals.id_to_slot.insert(individuals.id[child], child);
    child
}
