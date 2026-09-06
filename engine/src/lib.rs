//! Full simulation engine, native Rust, ECS-style structure-of-arrays
//! layout: "individuals" and "pixels" are separate component tables (1NF --
//! pixels reference their owning individual by a foreign key, exactly like
//! a one-to-many relational table, instead of each individual owning a
//! Python list). This is what actually removes the bottleneck identified
//! earlier: CPython's per-object/per-attribute overhead in the reproduction/
//! death/collision bookkeeping loop (measured ~25-40ms/tick), which no
//! amount of GPU offload for the NUMERIC kernels (FK, NN) could fix because
//! that bookkeeping was never the numeric part.
//!
//! Variable-length pixel data lives in one shared arena with a free-list
//! allocator (offset+length per individual) -- inspired by, not copied
//! from, a skip-list buddy allocator seen in a reference C++ engine
//! (FreeList.hpp): growth allocates a fresh `count+1` block and frees the
//! old one; death frees the block. Individual "rows" are similarly
//! tombstoned and slot-recycled rather than rebuilding the whole population
//! array every tick (the previous Python `[i for i in individuals if
//! i.alive]` pattern).
mod combat;
mod fields;
mod individuals;
mod physics;
mod pixels;
mod spatial;
mod terrain;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use numpy::ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use fields::Fields;
use individuals::Individuals;
use pixels::PixelArena;
use terrain::{Terrain, TerrainKind};

pub const JOINT_LEN: f32 = 1.0;
// Reproduction is priced by the offspring's actual body, not a flat fee --
// see the reproduction gate in physics.rs for why the old flat cost was the
// mechanism behind runaway population. Tuned so a minimal 2-part body pays
// about what it used to (~6.4) while a 20-part animal pays ~28, which is a
// real investment against a typical energy reserve in the low 30s.
pub const REPRODUCE_BASE_COST: f32 = 4.0;
pub const REPRODUCE_COST_PER_PART: f32 = 1.2;
pub const REPRODUCE_ENERGY_BUFFER: f32 = 7.0;
pub const MATURITY_AGE: f32 = 60.0;
// Measured directly: at typical population density in this world size,
// median nearest-neighbor distance is ~10.5 units -- the old 2.5 radius
// meant under 15% of individuals ever had anyone in range at all, which was
// the real cause of a population-wide reproduction collapse (energy was
// never the limiting factor; individuals were just too spread out to meet).
pub const MATE_RADIUS: f32 = 20.0;
// How much energy a male must carry, relative to the female's own, to be
// accepted. Sexual selection: energy becomes a display of condition rather
// than a private buffer, so merely surviving is no longer enough to breed.
pub const MATE_CHOICE_ENERGY_RATIO: f32 = 0.8;
// Birth anomalies. Reproduction used to do exactly one thing to the body plan
// -- append a single part -- so morphology could only ever creep outward one
// step at a time and could never simplify. Real developmental variation both
// adds and removes structure, and sometimes in more than unit steps. A lineage
// that can shed a useless limb, or gain a small cluster at once, explores a
// far wider space of shapes.
pub const ANOMALY_LOSE_PART_CHANCE: f32 = 0.10;
pub const ANOMALY_BURST_CHANCE: f32 = 0.12;
pub const ANOMALY_BURST_MAX: u32 = 3;
pub const BASE_METABOLISM: f32 = 0.02;
pub const PER_PIXEL_METABOLISM: f32 = 0.015;
pub const EAT_RATE: f32 = 2.0;
// Body mass at which grazing yield is already halved. Small bodies live off
// the food field; large ones have to eat other creatures. This is the single
// mechanism that turns one undifferentiated crowd into trophic levels.
// Swept against fixed seeds: at 6 nearly every creature was too large to feed
// itself and the world collapsed to a single individual; at 30 mean energy
// falls from ~96 to ~45 (reproduction threshold is ~20, so energy finally
// MEANS something), starvation rises from 11% to 16% of deaths, population
// stays healthy at 200-360, and genuinely large predators persist.
pub const GRAZE_MASS_REF: f32 = 30.0;
/// How much each unit of filter-mesh area raises the mass at which grazing
/// stops paying. This is the whole point of the organ: it buys the right to
/// be large AND still live on the food field.
pub const FILTER_GRAZE_BONUS: f32 = 0.9;
/// The speed at which a filter mesh is working at full capacity. Below it the
/// organ delivers proportionally less, and a stationary filter feeder gets
/// nothing from it at all -- a mesh only strains water that passes through it.
pub const FILTER_FLOW_SPEED: f32 = 1.2;
// A drifting resource mosaic. Blooms open at a new place now and then while
// standing capacity everywhere slowly fades, so a patch is a temporary thing
// and a grazer eventually has to go and find the next one.
// How far out shelter is sampled when working out which way cover lies.
pub const SHELTER_SENSE_RANGE: f32 = 6.0;
pub const FOOD_BLOOM_CHANCE: f32 = 0.06;
pub const FOOD_CAPACITY_DECAY: f32 = 0.9995;
pub const MOVE_COST: f32 = 0.01;
pub const MUTATION_STD: f32 = 0.15;
pub const COLLISION_RADIUS: f32 = 1.2;
pub const CAPTURE_BONUS: f32 = 2.0;
// Per-tick chance (scaled by the attacker's own evolved bite_force) that a
// held target loses one pixel outright, on top of the flat energy trickle --
// bounds capture duration by the target's pixel count instead of its
// (potentially huge, lifetime-accumulated) energy total. See the attachment
// loop in physics.rs for why the trickle alone let some captures run for
// hundreds of ticks, visible as creatures "locked in the air".
// Raised from 0.05 after measuring: at the old rate an evenly-matched
// grapple took ~300 ticks to resolve, so captures accumulated until 44% of
// the population was pinned in one at any moment. Predation should be
// punctuated -- a short violent event -- not the world's default state.
pub const CAPTURE_CHEW_CHANCE_BASE: f32 = 0.12;
pub const CAPTURE_CHEW_CHANCE_MAX: f32 = 0.5;
// --- Differentiated body parts (see pixels.rs). The engine's answer to
// "bodies get bigger but never more complex": a plain pixel is structure,
// but a part can specialise into an organ with a real function and a real
// upkeep cost. Every organ is strictly worse than plain body tissue UNLESS
// the animal is in a situation that uses it, which is what makes division of
// labour a discovery rather than a freebie. Nothing scores a "good" body
// plan; only these costs and effects exist.
pub const PART_DIFFERENTIATION_CHANCE: f32 = 0.30;
// Straight-tip persistence in the body-growth sampler. Kept as a named
// constant because this is an important morphology pressure, but the current
// tuned value preserves large-body viability better than lower exploratory
// settings in the complexity diagnostic.
// How strongly growth prefers extending an existing tip in its own direction
// over placing a part anywhere else. Swept against fixed seeds: at 8.0 roughly
// a quarter of all bodies were pure unbranched chains with 1.1 branch points
// each -- which is most of why creatures read as worms. At 2.0 pure chains
// fall to ~12% and branch points rise to ~1.75, with BETTER population than
// intermediate values, and dropping to 1.0 buys nothing further. Some bias is
// kept deliberately: real bodies do have a long axis, and removing it entirely
// would trade one unrealistic shape for another.
pub const GROWTH_STRAIGHT_TIP_WEIGHT: f32 = 2.0;
// Bilateral symmetry (see pixels.rs). Present in a minority of founders and
// able to flip either way when a part grows, so paired body plans are
// something evolution finds and can also lose, not a property of the world.
pub const SYMMETRY_FOUNDER_CHANCE: f32 = 0.35;
pub const SYMMETRY_FLIP_CHANCE: f32 = 0.06; // odds a node's symmetry trait flips when inherited
/// How much a symmetric node prefers to grow laterally, where a mirrored twin
/// is actually possible, over extending along the body axis where it is not.
pub const SYMMETRY_LATERAL_WEIGHT: f32 = 1.0;
/// Weight of a plain interior growth site -- one that neither extends a tip
/// nor places a bilateral pair. Low on purpose: interior sites outnumber tips
/// by an order of magnitude in any large body, so at equal weight they drown
/// out axial growth completely and every animal becomes a shrub.
pub const INTERIOR_SITE_WEIGHT: f32 = 0.12;
/// A little slack on the reach of a bite, so a mouth that is essentially in
/// contact still connects rather than missing by a hair every tick.
pub const BITE_REACH_SLACK: f32 = 0.6;
/// How far a tentacle can reach to sweep smaller animals toward the mouth,
/// how much smaller they must be to be worth gathering, and how hard the
/// sweep pulls.
pub const TENTACLE_REACH: f32 = 6.0;
pub const TENTACLE_PREY_RATIO: f32 = 2.0;
pub const TENTACLE_PULL: f32 = 2.5;
/// Growth is discouraged at a node already carrying two children, so trunks
/// stay trunks instead of fattening into slabs.
pub const CROWDED_NODE_PENALTY: f32 = 0.3;
// Upkeep multiplier applied to PER_PIXEL_METABOLISM, indexed by part kind
// (body, eye, mouth, gut, tentacle, armor, flipper). Plain body is the
// cheapest thing you can be made of.
// Upkeep per part kind. Sensors and fins were priced like armour plate, which
// with no compensating payoff left eyes, mouths and flippers all selected
// AGAINST (2.1-3.7% of tissue against a ~5% random baseline) while the purely
// passive gut thrived at 13.9%. A sense organ is not as expensive to carry as
// a slab of armour.
pub const PART_METABOLISM: [f32; crate::pixels::PART_KIND_COUNT as usize] =
    [1.0, 1.15, 1.3, 1.3, 1.5, 1.8, 1.35, 1.25];
// Each eye extends how far this individual can see (multiplier on
// VISION_RANGE), so a blind lump has to bump into the world while an
// eye-heavy body can track threats and prey at distance -- at a cost.
pub const EYE_VISION_BONUS: f32 = 0.35;
pub const EYE_VISION_MAX: f32 = 3.0;
// Mouths make bites land harder and prey get stripped faster.
pub const MOUTH_BITE_BONUS: f32 = 0.45;
pub const MOUTH_BITE_MAX: f32 = 3.5;
// Tentacles grip: they raise the chance of latching onto something and make
// the hold harder to struggle out of.
pub const TENTACLE_GRIP_BONUS: f32 = 0.5;
pub const TENTACLE_GRIP_MAX: f32 = 4.0;
// Armor plating absorbs damage (effective toughness) but is heavy.
pub const ARMOR_TOUGHNESS_BONUS: f32 = 0.4;
pub const ARMOR_TOUGHNESS_MAX: f32 = 3.0;
// Flippers convert the same swimming effort into more thrust.
pub const FLIPPER_THRUST_BONUS: f32 = 0.3;
pub const FLIPPER_THRUST_MAX: f32 = 2.5;
// A gut extracts more from every meal, which is what makes grazing and
// scavenging viable strategies next to predation.
pub const GUT_DIGESTION_BONUS: f32 = 0.35;
pub const GUT_DIGESTION_MAX: f32 = 3.0;

// How much faster a predator strips prey that is far smaller than itself.
// Consumption used to be size-blind, so a large animal was locked to
// whatever it grabbed first for as long as a tiny meal took to finish, and
// could never work through several small victims in succession.
pub const CHEW_DOMINANCE_MAX: f32 = 8.0;
// Floor on that ratio. Being much smaller than what you are biting must
// reduce how fast you can tear it apart, or size buys no protection through
// the capture path however armoured the victim is.
pub const CHEW_DOMINANCE_MIN: f32 = 0.06;
// One sweep can catch several victims: how much body mass buys each extra
// simultaneous target. Without this a large animal landed a single blow per
// tick while every small creature around it landed its own, which made size
// a liability instead of an advantage.
pub const SWEEP_MASS_PER_TARGET: f32 = 7.0;
// Gape limit. Prey this many times lighter than the predator is swallowed
// whole rather than chewed down part by part; mouths widen the gape, so
// what an animal can engulf is a real consequence of its anatomy. Efficiency
// is below 1 because swallowing whole wastes more than careful feeding does.
pub const ENGULF_SIZE_RATIO: f32 = 4.0;
pub const ENGULF_EFFICIENCY: f32 = 0.75;
pub const ENGULF_GAPE_MIN: f32 = 2.5;
// Prey struggling out of a grip. Per-tick escape odds are this base scaled
// by the victim's size advantage, so something that grabbed prey larger
// than itself loses it quickly while genuinely outmatched prey rarely gets
// free. Measured need: without any escape path, a quarter of the population
// sat permanently immobilised in a grip at any moment, and their captors
// were inert too -- about half the world doing nothing.
pub const STRUGGLE_ESCAPE_BASE: f32 = 0.05;
pub const STRUGGLE_ESCAPE_MAX: f32 = 0.45;
pub const COLOR_MUTATION_RATE: f32 = 0.015;
pub const COLOR_MUTATION_STD: f32 = 40.0;

// Kin recognition: a small heritable "signature" (inherited from the parent
// with the same kind of gaussian drift as every other evolved trait here --
// NOT a discrete species/color check), sensed as a similarity-to-nearest-
// neighbor value fed into the brain. This is deliberately just an ENGINE
// capability, not a "don't attack kin" rule: nothing here reads this value
// and suppresses fight_urge. Whether an individual's evolved brain weights
// actually learn to respond to it (and thus whether protective behavior
// toward close kin emerges at all) is left entirely to selection, which is
// why REPRODUCTION_SUCCESS_REWARD below exists -- a lineage whose members
// happen to fight less with their own actual kin will WIN that competition
// (its parents live to reproduce again, its children live to reproduce in
// turn) far more often than one that can't tell kin from strangers, but
// only if reduced kin-aggression actually pays off in survival, which
// requires the reward to make it pay off.
pub const KIN_DIM: usize = 3;
pub const KIN_MUTATION_STD: f32 = 0.1;

// A direct reward for successful inclusive fitness: when an individual
// reproduces, its own still-living parent (looked up by stable id, not
// slot) gets an energy bonus -- "a reward if the children live until their
// own procreation." Without this, a parent's genes get no extra payoff for
// having produced a child that survives to reproduce beyond the one-time
// cost of making it, so there's no selective pressure favoring a parent
// lineage that behaves protectively toward its own recent offspring instead
// of treating them as just more nearby biomass.
pub const REPRODUCTION_SUCCESS_REWARD: f32 = 4.0;

// How far a newborn spawns from its parent. Was 1.5 -- smaller than
// COLLISION_RADIUS (1.2) means a large fraction of births landed a newborn
// already in its own parent's (or a sibling's) point-blank bite range,
// before it had done anything at all. That's the direct, mechanical reason
// parents and children were seen killing each other seconds after birth --
// not evolved aggression, just spawning inside stabbing distance. Widened
// so a newborn reliably starts outside immediate combat/crowd range while
// still landing well within the same local neighborhood as its parent.
pub const BIRTH_OFFSET_STD: f32 = 4.0;

// --- Anatomy: per-PART size and joint-angle limits, heritable with drift
// exactly like flex/storage -- see pixels.rs's `size`/`min_angle`/`max_angle`.
pub const PART_SIZE_MUTATION_STD: f32 = 0.1;
pub const PART_ANGLE_MUTATION_STD: f32 = 0.12;
pub const PART_SIZE_MIN: f32 = 0.3;
pub const PART_SIZE_MAX: f32 = 2.5;

// --- Combat: see combat.rs. The old rule was a single scalar comparison
// (attacker's speed*bite_force vs. the target's flat evolved toughness) that
// deleted whatever pixel got hit AND every descendant of it in one shot the
// instant it passed -- with no relationship at all to how big the target
// actually was. A small, fast, high-bite_force attacker could sever a huge
// body in a single hit just because that body's `toughness` trait happened
// to be unremarkable, which is what "a small individual one-shots a much
// bigger one" actually was: not an occasional fluke, a coin-flip on one
// unscaled number every single hit. These constants replace the pass/fail
// gate with a graded, multi-mechanism system: hit points that scale with
// real body size, a bonus for striking the head, and poison as a real
// alternative to blunt force.
pub const BASE_PIXEL_HEALTH: f32 = 6.0;
// A body's effective armor scales with its own total size -- the actual
// fix for one-shotting: overwhelming a big body now requires overwhelming
// its actual mass, not just clearing a size-independent number.
pub const TOUGHNESS_SIZE_SCALING: f32 = 0.35;
// A starving attacker bites weakly -- lethality is tied to real, currently-
// held resources, not just a fixed trait value regardless of condition.
pub const ATTACKER_ENERGY_DAMAGE_REF: f32 = 15.0;
// Striking the root ("head") deals bonus damage -- a real, discoverable
// vital point, not a hardcoded species weakness: any evolved brain that
// learns to aim for a target's root (already sensed via position/gradient
// cues) gets a mechanical payoff for it.
pub const HEADSHOT_MULTIPLIER: f32 = 2.5;
// A bite also injects the attacker's own evolved acid_secretion into the
// target's location -- poison as a real, slower alternative kill path next
// to blunt "slicing" damage, reusing the existing acid field/damage system
// rather than adding a second parallel mechanic.
pub const VENOM_INJECTION_SCALE: f32 = 0.6;
// Pixels slowly heal when energy allows -- a body that survives a fight and
// then eats can recover, rather than every wound being permanent.
pub const HEALTH_REGEN_RATE: f32 = 0.03;
pub const HEALTH_REGEN_ENERGY_COST: f32 = 0.02;

pub const PHEROMONE_EMIT_BASE: f32 = 0.5;
pub const PHEROMONE_DECAY: f32 = 0.985;
pub const PHEROMONE_DIFFUSION: f32 = 0.15;
pub const BLOOD_EMIT_ON_HIT: f32 = 0.6;
pub const BLOOD_EMIT_ON_DEATH: f32 = 2.0;
pub const BLOOD_DECAY: f32 = 0.95;
pub const BLOOD_DIFFUSION: f32 = 0.2;
pub const ACID_EMIT_BASE: f32 = 0.35;
pub const ACID_DECAY: f32 = 0.97;
pub const ACID_DIFFUSION: f32 = 0.12;
pub const ACID_DAMAGE_RATE: f32 = 0.6;
pub const LIGHT_EMIT_BASE: f32 = 0.4;
pub const LIGHT_DECAY: f32 = 0.96;
pub const LIGHT_DIFFUSION: f32 = 0.25;
pub const VISION_LOOKAHEAD: f32 = 4.0;
// Real vision: until now, "sensing" another individual only ever happened
// via chemical gradients (blood/pheromone/acid) or by actually colliding
// with them -- there was no way to perceive "something is nearby" before
// it was already close enough to bite or be bitten. This gives every
// individual a genuine sense of the nearest bigger body (a plausible
// threat) and the nearest smaller one (a plausible meal) within
// VISION_RANGE, as a direction + proximity in the sense vector. Nothing
// here decides what to do with that information -- fleeing a threat and
// pursuing prey are both just possible uses of the same two numbers; an
// evolved brain that never learns to use them gets no defaults either.
pub const VISION_RANGE: f32 = 12.0;
// A body that's genuinely away from any food patch used to get a dead-zero
// food gradient (the old gradient only ever compared immediately-adjacent
// cells, so it stayed at zero until the body was already touching a
// patch's edge) -- no signal at all to steer by, indistinguishable from
// "there is no food anywhere". Widened specifically for food (not the
// other fields) so an isolated, exploring individual has a real directional
// cue toward the nearest patch instead of wandering blind until starving.
pub const FOOD_SMELL_RANGE: i32 = 18;
// How far a body can locate food WITHOUT eyes, and how much each eye extends
// that. Smell alone gets you to food you are nearly on top of; sight is what
// lets an animal cross open water toward a patch it can see.
// Metabolic scaling exponent, applied to total body AREA. In two dimensions
// the exchange boundary is a perimeter and the bulk is an area, so the
// surface law gives an exponent near 0.5; 1.0 would be cost strictly
// proportional to area, which measurement shows collapses worlds outright.
pub const METABOLIC_EXPONENT: f32 = 0.6;
/// Area of a typical plain body part, so charging upkeep on area rather than
/// on a part count did not silently rescale the entire energy economy.
pub const PART_AREA_REF: f32 = 0.49;
// Measured: setting the blind range to 4 against a full range of 18 emptied
// the world. The founding population has almost no eyes, so a hard gate
// starves everything long before eyes can evolve -- a bootstrapping cliff, not
// a gradient. Blind foraging has to stay viable while being clearly worse.
pub const FOOD_SMELL_RANGE_BLIND: i32 = 12;
pub const FOOD_SIGHT_RANGE_PER_EYE: i32 = 3;

// Day/night cycle. DAY_LENGTH is in simulated seconds (dt=0.1/tick) -- at
// ~50-300 ticks/sec, that's a real, visibly-cycling rhythm on screen, not
// an imperceptibly slow one.
pub const DAY_LENGTH: f32 = 240.0;
pub const DAY_LIGHT_AMPLITUDE: f32 = 0.5;

// Aposematism (warning coloration): a body whose color stands out sharply
// against its surroundings is normally just easier to notice (the exact
// opposite of camouflage) -- UNLESS it also backs that up with a real
// defense (toughness or acid/venom), in which case standing out can
// instead make a predator hesitate, the same real-world logic as a wasp's
// stripes or a poison frog's color: advertise the defense so predators
// learn to avoid finding out the hard way. This does NOT require any
// species-level "memory" or learning to implement -- it's evaluated as a
// direct, honest correlation between conspicuousness and actual defense
// at the moment of the encounter, so a loud color with nothing backing it
// gets no benefit (and remains easier to notice, per camouflage), while a
// loud color paired with real toughness/venom does. Dishonest mimicry
// (loud color, no real defense, still evolves anyway because SOME
// predators hesitate without checking) is a real, interesting possible
// outcome this doesn't prevent.
pub const APOSEMATISM_TOUGHNESS_REF: f32 = 2.0;
pub const APOSEMATISM_ACID_REF: f32 = 0.5;
pub const APOSEMATISM_DETERRENCE_MAX: f32 = 0.6;

// Sessile anchoring (see individuals.rs's anchor_strength). Only applies
// while actually resting on solid ground -- see the movement block in
// physics.rs for why open water gives zero benefit regardless of the trait.
pub const ANCHOR_METABOLISM_DISCOUNT: f32 = 0.5;

// Quorum sensing: see fields.rs's `quorum` field doc comment. Emission is
// flat and universal (every living individual, no trait involved) --
// that's the actual biological concept (constitutive autoinducer
// production), not a design shortcut. Decays a bit slower/diffuses a bit
// further than pheromone so it reflects sustained local crowding rather
// than an instant snapshot -- quorum sensing in nature is a slow-forming
// consensus signal, not a sharp per-encounter cue like blood or pheromone.
pub const QUORUM_EMIT_BASE: f32 = 0.12;
pub const QUORUM_DECAY: f32 = 0.992;
pub const QUORUM_DIFFUSION: f32 = 0.1;

// Camouflage: color already evolves freely (mutation + drift, no
// selective pressure attached to it before this). This gives it a real
// consequence -- a body whose evolved color happens to match its local
// surroundings is genuinely harder for a predator to notice, exactly the
// textbook industrial-melanism/crypsis story, and nothing here decides
// WHAT color is safe anywhere; that depends entirely on where the
// individual actually lives (open water vs. sand vs. a food-rich patch
// all read as different ambient colors, computed in combat.rs).
pub const CAMOUFLAGE_COLOR_RANGE: f32 = 260.0;
pub const CAMOUFLAGE_DETECTION_PENALTY: f32 = 0.85;

// Stigmergic territoriality, inspired by real home-range-formation models
// (scent-marking random walkers with a central-place-forager return bias --
// e.g. wolf/mammal home-range studies via scent marks). Decays far slower
// and diffuses far less than pheromone/quorum: a real territorial scent
// mark is meant to persist and stay LOCAL (a boundary line, not a cloud),
// not function as a fast-moving trail signal. HOME_RANGE_NORM is the
// distance scale the home-direction sense saturates at (tanh) -- set to a
// meaningful fraction of the (240-cell) world, not the whole map, so
// "far from home" actually means something before an individual has
// wandered edge-to-edge.
pub const TERRITORY_EMIT_BASE: f32 = 0.3;
pub const TERRITORY_DECAY: f32 = 0.997;
pub const TERRITORY_DIFFUSION: f32 = 0.04;
pub const HOME_RANGE_NORM: f32 = 50.0;

// Janzen-Connell / conspecific negative density dependence: the documented
// real-world reason diverse communities don't collapse into a single
// best competitor -- specialist pathogens track their host, so mortality
// rises disproportionately exactly where conspecifics are densest, handing
// locally-rare lineages an advantage. CONSPECIFIC_DENSITY_NORM is the
// kin-weighted neighbor count at which pressure saturates (tanh);
// PATHOGEN_DAMAGE_RATE is the per-tick energy drain at full pressure with
// zero resistance -- deliberately a few times BASE_METABOLISM (0.02), so
// living inside a dense monoculture is a real, survivable-but-costly drag
// rather than an instant death sentence. Resistance divides the damage
// (1/(1+r)) instead of subtracting it, so it always helps but never grants
// full immunity no matter how high it evolves.
// Competition for space. Bodies stacked on each other with no penalty, so a
// creature could sit in a heap and breed without ever needing to do anything
// well. Charged only above a tolerance, so an ordinary family group is free
// and only a genuine crush costs; scaled by body size, because a large animal
// needs proportionally more room than a small one.
pub const CROWDING_TOLERANCE: f32 = 3.0;
pub const CROWDING_ENERGY_COST: f32 = 0.012;
/// Density-dependent mortality, measured against other animals' components
/// pressing on this one's, per component, and blind to kinship. Quadratic
/// because interference competition intensifies faster than linearly with
/// packing. This is what actually regulates numbers -- the energy tax above
/// never did.
/// How far beyond actual contact a neighbouring component still counts as
/// pressing on this one. Contact itself is too strict a test for "crowded" --
/// animals packed shoulder to shoulder are crowded before they interpenetrate.
pub const SPACE_PRESSURE_RANGE: f32 = 2.0;
pub const SPACE_PRESSURE_TOLERANCE: f32 = 1.5;
pub const SPACE_PRESSURE_ENERGY_COST: f32 = 0.05;
pub const SPACE_PRESSURE_MORTALITY: f32 = 0.0009;
pub const SPACE_PRESSURE_MORTALITY_MAX: f32 = 0.02;
pub const CROWDING_SIZE_FACTOR: f32 = 0.02;
// Trespassing on ground someone else has marked. This is what turns territory
// marking from a decorative field into a defended range worth holding.
pub const TRESPASS_MARK_THRESHOLD: f32 = 0.30;
pub const TRESPASS_ENERGY_COST: f32 = 0.05;

pub const CONSPECIFIC_KERNEL_EXPONENT: i32 = 4;
pub const CONSPECIFIC_DENSITY_NORM: f32 = 4.0;
// Response curve, calibrated against a measured distribution rather than
// guessed: members of dominant lineages sit among ~3.5x more of their own
// kind than members of rare ones, but the first attempt read damage off a
// tanh that saturates by density ~3, compressing that real 3.5x signal
// into a 1.4x one -- a near-flat tax that measurably REDUCED lineage
// counts and crashed population in all 3 A/B seeds. Below THRESHOLD (an
// ordinary family group, which is normal and must be free) pressure is
// zero; above it damage scales with the SQUARE of the excess, which is the
// "overcompensating" density dependence the literature actually
// specifies. MAX caps the drain so even the densest core is a heavy,
// survivable cost rather than instant death.
// Keyed on global lineage share (0..1), NOT local crowding: two separate
// measurements showed local conspecific density can't tell an abundant
// lineage from a rare one in this engine (every lineage clumps equally,
// since offspring spawn beside their parent), which made the first version
// a flat population-wide tax that cut lineage counts and crashed
// population in all 3 A/B seeds. A lineage under THRESHOLD of the world
// pays nothing; past it, cost rises with the square of the excess share.
pub const PATHOGEN_SHARE_THRESHOLD: f32 = 0.08;
pub const PATHOGEN_SHARE_SCALE: f32 = 0.22;
pub const PATHOGEN_PRESSURE_MAX: f32 = 2.5;
// Calibrated against a 3-seed A/B using EFFECTIVE species count (inverse
// Simpson), not raw lineage count -- raw count is confounded, since fewer
// individuals mechanically means fewer distinct colors. That metric showed
// the untreated world is effectively a TWO-species world (2.0-2.1) despite
// 143-163 nominal lineages. 0.02 buys real evenness at little population
// cost (seed 22: 5960 -> 5630 population, 143 -> 152 lineages, effective
// 2.0 -> 2.9); 0.035 buys far more (effective 10.5) but costs a third to
// two thirds of the population, which is too steep. This is a genuine
// diversity-vs-population trade-off, not a single correct value.
pub const PATHOGEN_DAMAGE_RATE: f32 = 0.02;
pub const DISEASE_RESISTANCE_METABOLIC_COST: f32 = 0.012;

// Experience logging (see ExperienceRow). Stride 200 means, on average,
// each living individual gets logged once every 200 ticks -- enough to
// eventually accumulate a large, diverse dataset without adding real
// per-tick cost (deciding slots are already iterated for the brain forward
// pass; this just checks a cheap modulo on ones already being visited).
// The cap is a hard ceiling on memory if nothing ever drains the buffer.
pub const EXPERIENCE_SAMPLE_STRIDE: u64 = 200;
/// How many consecutive ticks an individual is logged for once its turn
/// comes round. Consecutive rows are what make (state, action) -> next state
/// learnable; isolated snapshots taken 200 ticks apart cannot express
/// consequences at all.
pub const EXPERIENCE_TRAJECTORY_LEN: u64 = 8;
pub const EXPERIENCE_LOG_CAP: usize = 500_000;

pub const CRAWL_FLOOR_THRESHOLD: f32 = 15.0;
pub const CRAWL_THRUST_SCALE: f32 = 0.9;
pub const SAND_EXTRA_DAMPING: f32 = 0.5;
pub const SAND_DIG_COST: f32 = 0.015;
// Below this evolved dig_strength, the sand seafloor is a real solid
// surface -- you rest on top of it, you don't sink through it just because
// gravity keeps pulling. Getting past it (down toward, or through, the
// bottom) requires a real, evolved capability, not just existing near it.
pub const SAND_DIG_THRESHOLD: f32 = 0.15;

// Measured live: scavenge events outnumbered reproductions by ~15-20x in
// every long-running session this population reached. Corpses accumulate
// wherever predation/cannibalism happens, which is disproportionately
// inside the same dense, stationary clusters that gravity + zero cost for
// staying still already produce -- meaning a static pile can sustain
// itself indefinitely by eating its own dead, entirely independent of the
// (real, already-modeled) local food field it's almost certainly already
// exhausted. That's the actual mechanism behind "swimmers don't
// reproduce, static individuals do" and "mobility isn't rewarded": moving
// away from the pile forfeits a very cheap, abundant energy source for
// the SAME depleted food field a lone forager has to actively relocate to
// escape. Halved so scavenging remains a real, useful behavior (real
// animals do scavenge) without being able to fully substitute for actual
// foraging -- a pile that's eaten its local food field should still need
// to disperse eventually, not sustain itself forever on its own casualties.
pub const CORPSE_ENERGY_PER_PIXEL: f32 = 1.4;
pub const CORPSE_EAT_RADIUS: f32 = 1.5;
pub const CORPSE_EAT_RATE: f32 = 0.5;
pub const CORPSE_SINK_RATE: f32 = 0.05;

pub const DRAG_PARALLEL: f32 = 0.05;
pub const DRAG_PERPENDICULAR: f32 = 0.4;
// Per-part multiplier on perpendicular drag, indexed by part kind (body, eye,
// mouth, gut, tentacle, armor, flipper). A flipper is a paddle and grips the
// water hard when swept; a tentacle is soft and slips through it; armour is a
// broad plate. This is what makes fins an organ that propels rather than a
// number that scales.
// A filter mesh is a broad face held into the flow, so it drags like one --
// which is a real cost of the strategy, not a free bonus: a filter feeder is
// slow, and that is why it is a grazer rather than a hunter.
pub const PART_DRAG_PERP: [f32; crate::pixels::PART_KIND_COUNT as usize] =
    [1.0, 1.0, 1.0, 1.0, 0.6, 1.5, 3.2, 2.0];
pub const LINEAR_DAMPING: f32 = 0.25;
pub const TURN_RATE: f32 = 0.12;
pub const MAX_SPEED: f32 = 3.0;
pub const THERMAL_NOISE: f32 = 0.35;
pub const COLLISION_STIFFNESS: f32 = 1.2;
/// What fraction of a measured overlap is undone by direct displacement each
/// tick, and the most any one body may be displaced in a tick. Removing the
/// whole overlap at once injects energy and makes dense crowds explode;
/// removing a fraction eases bodies apart, which is how Baumgarte
/// stabilisation and position-based dynamics both behave.
pub const CONTACT_CORRECTION: f32 = 0.35;
pub const CONTACT_CORRECTION_MAX: f32 = 0.6;
// Rock repulsion. Move-rejection alone cannot keep bodies out of walls,
// because undulation puts limbs inside stone with no translation at all.
pub const TERRAIN_REPULSION_RANGE: f32 = 1.4;
pub const TERRAIN_REPULSION_STIFFNESS: f32 = 3.0;
/// The seafloor is back, and gravity with it, because the world is a cylinder
/// again: left and right are joined but top and bottom are not, so there is a
/// real surface and a real bottom. Rock stays off -- what was asked for was the
/// floor, not the boulder fields.
pub const TERRAIN_ENABLED: bool = true;
pub const ROCK_ENABLED: bool = false;

// --- Marine snow -----------------------------------------------------------
// Plankton enters at the lit surface and sinks. How deep the source band is,
// how fast the snow falls, and how quickly what reaches the bottom is lost.
// The floor decay matters: without it the seafloor becomes a reservoir and
// the world is back to permanent food patches, just lower down.
pub const SNOW_SOURCE_DEPTH: usize = 6;
pub const SNOW_SINK_RATE: f32 = 0.22;
pub const SNOW_FLOOR_DEPTH: usize = 14;
pub const SNOW_FLOOR_DECAY: f32 = 0.06;
pub const SNOW_PLUMES_PER_TICK: u32 = 3;
/// Bloom cycle. Production is intermittent, and the intervals of NOTHING are
/// the point: a constant drizzle is just the old always-fed world at a lower
/// rate, whereas famine is what makes a reserve worth carrying and a bloom
/// worth finding. Two periods beating against each other so the rhythm does
/// not become something a lineage can simply time.
pub const SNOW_BLOOM_PERIOD_A: f32 = 1900.0;
pub const SNOW_BLOOM_PERIOD_B: f32 = 730.0;
/// Below this the water is barren -- a real famine, not a lull.
pub const SNOW_BLOOM_FLOOR: f32 = 0.34;
/// Peak plankton input per plume. Calibrated against what the old
/// regrow-in-place field actually delivered: roughly 0.001 per cell per tick
/// over 57600 cells, about 58 units of food per tick. Marine snow at 0.075
/// delivered nearer 6, and the world starved to two animals in 5700 ticks.
/// Scarcity is wanted; an empty ocean is not.
pub const SNOW_BLOOM_STRENGTH: f32 = 0.34;

// --- Whale fall ------------------------------------------------------------
// A rare, enormous carcass sinking from above. A different KIND of resource
// from marine snow: snow rewards steady filtering along the drift, a carcass
// rewards noticing one, reaching it fast, and holding it against competitors.
pub const WHALE_FALL_CHANCE: f32 = 0.0012;
pub const WHALE_FALL_MIN_PARTS: u32 = 45;
pub const WHALE_FALL_MAX_PARTS: u32 = 130;
pub const WHALE_FALL_RICHNESS: f32 = 3.5;
pub const GRAVITY: f32 = 0.22;

// Density-dependent cannibalism used to be modeled as a hardcoded PRESSURE
// added straight to fight_urge once local density crossed a threshold --
// removed (see the fight-attempt gate in physics.rs) because it was exactly
// the kind of indiscriminate, context-blind aggression trigger real animals
// don't have: it fired on density alone, with no relationship to hunger,
// territory, or mate competition. Local density is still sensed (quorum_local
// in sense()) -- if crowding-driven aggression is ever actually
// advantageous, evolution can still learn that response on its own, as a
// real behavioral adaptation rather than an engine-enforced one.
//
// What replaced it: a genuine energy cost on every fight ATTEMPT (not just
// landed hits), so reflexive/constant aggression is no longer free --
// hunger-gating becomes an actual selective advantage instead of something
// that has to fight against a hardcoded density trigger to ever show up.
pub const ATTACK_ENERGY_COST: f32 = 0.2;

// Development: individuals now genuinely GROW over their own lifetime
// (previously all growth happened only once, to a newborn child -- an
// individual's own body never changed size after birth at all). This is
// pure INFLATION, not new anatomy: the body PLAN (pixel count, joint
// topology, limbs) is fixed for life at birth -- only `size_scale`, a
// uniform multiplier on every joint's length, grows. A body sprouting new
// parts over its own lifetime read as "evolving while alive", which isn't
// what a single organism's growth is; getting uniformly bigger is. Scale
// growth costs real energy and slows logarithmically as it approaches its
// adult size (size_scale >= ADULT_SIZE_SCALE) -- fast early growth,
// diminishing returns after, without hardcoding a final size (a body can
// still slowly keep inflating past "adult", just far more slowly).
pub const GROWTH_ENERGY_THRESHOLD: f32 = 11.0;  // must have this much spare energy to grow at all
pub const GROWTH_ENERGY_COST: f32 = 0.8;
pub const GROWTH_BASE_CHANCE: f32 = 0.08;   // per-tick chance to attempt growth when eligible, before the slowdown curve
pub const GROWTH_SLOWDOWN: f32 = 0.35;      // higher = growth decelerates faster with size
pub const GROWTH_SCALE_INCREMENT: f32 = 0.08; // how much size_scale increases per successful growth tick
pub const ADULT_SIZE_SCALE: f32 = 1.3;      // inflation-from-birth to count as a real, developed adult -- a low
                                             // bar deliberately: this stacks with MATURITY_AGE and the feed/cooldown
                                             // gates, and a first tuning pass with a much stricter bar collapsed the
                                             // population toward extinction (measured: reproductions dropped from
                                             // thousands to 36 over 2000 ticks) by compounding too many hard
                                             // requirements at once
pub const MAX_SIZE_SCALE: f32 = 3.0;        // hard ceiling on lifetime inflation -- without one, growth chance decays
                                             // but never reaches zero, so a long-lived body can inflate unboundedly;
                                             // since metabolism scales with size, that runs the upkeep cost past
                                             // what any amount of foraging can sustain (measured: this alone was
                                             // enough to collapse the whole population toward extinction)

// Reproduction previously required only a static energy threshold, which
// scavenging (abundant, easy) made trivially easy to sit at indefinitely --
// energy was never actually scarce. Now it also requires having actually
// fed (food, a landed bite, or scavenging) recently: a real "you have to
// go find/catch food before you can have offspring" requirement.
pub const RECENT_FEED_WINDOW: u32 = 150;

// A real recovery period between births, for females only (males have no
// equivalent biological constraint here -- gestation/nursing is asymmetric
// in real biology, and this makes reproduction genuinely bounded by more
// than "find a mate + have energy" for the sex that actually bears young).
// Base recovery after birth, before the size term. See gestation_ticks():
// total gestation grows with the offspring's part count, so a 20-part animal
// waits roughly 25 + 60 ticks against a small one's 25 + 6.
pub const FEMALE_REPRODUCTION_COOLDOWN: u32 = 25;
pub const GESTATION_TICKS_PER_PART: f32 = 3.0;
// Space and safety as preconditions for breeding. Together these make
// reproduction density-dependent and locally regulated: a crowded or
// recently-violent patch simply does not produce offspring, so population is
// checked where it is dense rather than by a global ceiling, and there is
// real pressure to disperse into quieter water or into the reef.
pub const BREEDING_SPACE_RADIUS: f32 = 9.0;
pub const BREEDING_SPACE_MAX_NEIGHBORS: f32 = 6.0;
pub const BREEDING_SPACE_SIZE_PENALTY: f32 = 0.06; // bigger bodies need proportionally more room
pub const BREEDING_SAFETY_BLOOD_MAX: f32 = 0.35;
// Credited to an individual the moment it successfully reproduces, for
// the experience log the future replay training will consume.
pub const REWARD_REPRODUCE: f32 = 1.0;
/// Keeping a full larder is worth a little every tick. Deliberately far below
/// the reproduction reward: rewarding energy directly was previously measured
/// making animals hoard rather than breed (mean energy tripled, births fell
/// 73%), so this nudges toward reserves without making hoarding the goal.
pub const REWARD_ENERGY_STOCK: f32 = 0.004;
/// What a body can hold. Energy used to be unbounded -- an animal was
/// observed sitting on 898 units of free, organ-less buffer. Capacity is now
/// something built out of storage tissue and gut, weighted by part area.
pub const ENERGY_CAP_BASE: f32 = 12.0;
/// Energy banked per unit of tissue-area, on top of what the tissue type
/// itself stores (see `pixels::PART_STORAGE`).
pub const ENERGY_CAP_PER_STORAGE_TRAIT: f32 = 0.8;
pub const ENERGY_CAP_SCALE: f32 = 16.0;
// How far a newborn's decoder is pulled toward the learned baseline policy.
// Deliberately partial: at 1.0 every creature would start identical and the
// variation selection needs would be gone, which would trade evolution away
// for learning instead of combining them.
pub const POLICY_DISTILL_RATE: f32 = 0.25;
// Range of brain-controlled swimming effort. The floor is above zero because
// a body still drifts and flexes when idling; the ceiling is a genuine
// sprint. Cost rises with the square of the gain, so maxing it out is only
// worth it when something is actually at stake.
pub const SWIM_GAIN_MIN: f32 = 0.15;
pub const SWIM_GAIN_MAX: f32 = 1.9;
// Turning as physics rather than assignment. The brain holds a body
// curvature; fluid torque on that curved body rotates it, damped by water.
// Phase advance per unit of body depth: how many radians the bending wave
// shifts between one part and the next one further from the head. Sets the
// wavelength of the travelling wave running down the body.
pub const BODY_WAVE_NUMBER: f32 = 0.7;
pub const TURN_CURVATURE_SCALE: f32 = 0.9;
// How far a joint may be held bent as a deliberate postural command, on top
// of whatever it is oscillating through. Bounded so a body cannot fold itself
// into a knot, but generous enough that steering actually has authority.
pub const MAX_POSTURE_BEND: f32 = 0.9;
pub const TURN_POSTURE_BIAS_SCALE: f32 = 0.35;
pub const ROTATIONAL_INERTIA: f32 = 2.5;
pub const ANGULAR_DAMPING: f32 = 2.0;
pub const MAX_ANGULAR_SPEED: f32 = 2.5;

pub const WEATHER_TRIGGER_CHANCE: f64 = 0.0006;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Weather {
    None,
    FoodBloom,
    ColdSnap,
    FertileSurge,
}

#[derive(Clone)]
pub struct Corpse {
    pub root_pos: [f32; 2],
    pub local_shape: Vec<[f32; 2]>,
    pub color: [u8; 3],
    pub energy: f32,
}

/// First concrete step toward the "shared brain trained asynchronously on
/// the GPU from experience replay" idea: raw (state, action, energy)
/// transitions, sampled cheaply and deterministically (no RNG call inside
/// the hot per-tick parallel loop -- see physics.rs), for a background
/// Python/PyTorch process to eventually consume. Deliberately NOT trying to
/// compute a reward or do any training here -- reward shaping (energy
/// delta, death penalty, reproduction bonus) is offline-computable from
/// consecutive rows sharing the same `id`, and belongs in the training
/// pipeline, not the simulation engine.
pub struct ExperienceRow {
    pub id: u64,
    pub tick: u64,
    pub sense: Vec<f32>,
    pub action: Vec<f32>,
    pub energy: f32,
    /// Reward credited since this individual was last sampled. Currently
    /// reproduction events; energy deltas remain derivable offline from
    /// consecutive rows sharing an id.
    pub reward: f32,
}

#[pyclass]
pub struct World {
    pub size: u32,
    pub pop_cap: usize,
    pub rng: Pcg64,
    pub tick_count: u64,
    pub sim_time: f32,
    pub dt: f32,
    pub food_regrow_rate: f32,
    pub food_cap: f32,

    pub individuals: Individuals,
    pub pixels: PixelArena,
    pub corpses: Vec<Corpse>,
    pub terrain: Terrain,
    pub fields: Fields,

    pub weather: Weather,
    pub weather_ticks_left: i32,
    pub food_regrow_multiplier: f32,
    pub metabolism_multiplier: f32,
    pub maturity_multiplier: f32,
    // A real day/night cycle: 0 at deep night, 1 at high noon, a smooth sine
    // of sim_time -- not cosmetic. It shifts the ambient color camouflage is
    // judged against (see combat.rs) and is itself a sense input, so
    // evolution can respond to time of day (nocturnal vs. diurnal foraging,
    // hiding by day if predation risk tracks light) if that ever pays off.
    pub day_light: f32,

    pub reproductions: u64,
    pub fights: u64,
    pub deaths: u64,
    pub scavenged: u64,
    // What creatures actually die OF. Without this, "is there real
    // survivability pressure?" is guesswork -- a world where nearly everything
    // dies of old age or starvation is one where behaviour barely matters,
    // and no amount of brain capacity will make intelligence impactful.
    pub deaths_starved: u64,
    pub deaths_predation: u64,
    pub deaths_popcap: u64,
    /// Deaths from density-dependent mortality, reported separately so it is
    /// visible whether crowding is actually regulating the population or just
    /// adding noise to the other causes.
    pub deaths_crowding: u64,
    pub whale_falls: u64,

    pub timings: Vec<(&'static str, f64)>, // (phase, milliseconds) for the most recent tick -- diagnostic only

    // See ExperienceRow's doc comment. Sampled at EXPERIENCE_SAMPLE_STRIDE
    // and capped at EXPERIENCE_LOG_CAP so a Python side that forgets to
    // drain this can't leak memory unboundedly -- old samples are simply
    // dropped (not shifted, to keep this O(1) per tick) once full.
    pub experience_log: Vec<ExperienceRow>,

    // Runtime-overridable copy of PATHOGEN_DAMAGE_RATE, so an A/B test can
    // hold the seed fixed and vary ONLY this (a compile-time constant
    // can't be compared against itself in one run). Production never
    // changes it.
    pub pathogen_damage_rate: f32,
    // Runtime-overridable copy of REPRODUCE_COST_PER_PART, for the same
    // reason as pathogen_damage_rate: sweeping a compile-time constant
    // against a fixed seed is impossible in a single run.
    pub repro_cost_per_part: f32,
    /// Runtime-overridable GRAZE_MASS_REF, so the trophic threshold can be
    /// swept against fixed seeds instead of guessed at.
    pub graze_mass_ref: f32,
    /// Runtime-overridable sensory economics, so the question "does gating
    /// long-range food detection behind eyes make eyes pay for themselves?"
    /// can be answered by an A/B on one build rather than two.
    pub blind_smell_range: i32,
    pub sight_range_per_eye: i32,
    pub part_metabolism: [f32; crate::pixels::PART_KIND_COUNT as usize],
    /// Runtime-overridable METABOLIC_EXPONENT, so the strength of the
    /// large-body energy discount can be swept. 1.0 is the old linear cost.
    pub metabolic_exponent: f32,
    /// Runtime-overridable COLLISION_STIFFNESS. Detecting an overlap is not
    /// the same as resolving one: with detection fixed, 70% of components
    /// were still measured sitting inside another animal's component, so how
    /// hard contact actually pushes has to be swept rather than guessed.
    pub collision_stiffness: f32,
    /// Runtime-overridable CONTACT_CORRECTION, so the positional half of
    /// contact resolution can be turned off and compared against the pure
    /// force solver it replaced. 0.0 restores the old behaviour exactly.
    pub contact_correction: f32,
    /// Test-only: replaces every brain's output with deterministic noise.
    /// The point is to answer, before spending any more effort on making
    /// brains bigger or better trained, whether the brain is doing ANYTHING
    /// measurable -- if a world of animals deciding at random performs the
    /// same as a world of evolved ones, then behaviour is not on the critical
    /// path to fitness and no amount of network capacity will change that.
    pub brain_noise: f32,
    /// Runtime-overridable density-dependent mortality, so how hard crowding
    /// bites can be swept rather than guessed.
    pub space_pressure_tolerance: f32,
    pub space_pressure_mortality: f32,
    /// Runtime-overridable marine snow, so how much plankton the ocean
    /// actually produces can be calibrated rather than guessed.
    pub snow_strength: f32,
    pub snow_plumes: u32,
    /// Runtime-overridable THERMAL_NOISE, so the noise floor can be swept
    /// against fixed seeds rather than guessed at.
    pub thermal_noise: f32,
    /// Runtime-overridable GROWTH_STRAIGHT_TIP_WEIGHT, so the morphology
    /// pressure that biases bodies toward elongation can be swept.
    pub growth_tip_weight: f32,
    /// Test-only: when set, forces (turn_curvature, swim_gain) every tick
    /// so the brain cannot reshape the body. Without this, measuring
    /// propulsion is confounded -- the brain senses different things at
    /// different orientations and curves the body differently, so an
    /// IDENTICAL body produced thrust varying from 0 to 13.7 across
    /// headings, which looks like broken physics but is just behaviour.
    pub freeze_locomotion: Option<(f32, f32)>,
    /// Runtime-overridable ANGULAR_DAMPING, so the balance between
    /// intentional turning and involuntary self-spin can be swept.
    pub angular_damping: f32,
    /// Runtime-overridable POLICY_DISTILL_RATE, so how hard learned
    /// instinct is pressed into newborns can be swept. At 1.0 a newborn
    /// starts as an exact copy of the learned policy, which isolates
    /// 'is the policy any good' from 'is the distillation strong enough'.
    pub policy_distill_rate: f32,
    /// Runtime-overridable CORPSE_ENERGY_PER_PIXEL. This is the price of a
    /// meal, and therefore whether being a predator can pay for a large
    /// body at all.
    pub meal_energy_per_part: f32,
    /// Runtime-overridable GRAVITY, so locomotion can be probed in
    /// isolation without sinking confounding the measurement.
    pub gravity: f32,

    // The world's shared perception encoder (see individuals::encode). One
    // matrix for every creature alive, initialised randomly. A random
    // projection already preserves enough structure to be a usable feature
    // basis, and it removes the impossible job each individual previously
    // had of evolving its own feature extractor from raw senses. This is
    // also precisely the object a GPU replay-training process would improve:
    // swap better weights in here and every creature perceives better
    // without any of them losing their own evolved decision-making.
    pub shared_enc_w: Vec<f32>,
    pub shared_enc_b: Vec<f32>,

    // A baseline decision policy learned on the GPU from the population's own
    // successful behaviour, in exactly the shape of an individual's decoder.
    // Newborns are nudged toward it at birth and then evolve away from it, so
    // learning reaches the population the way instinct does -- through births
    // -- while every individual still owns and mutates its own decisions.
    // None until the trainer has produced one; the world runs fine without.
    pub shared_policy: Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
}

#[pymethods]
impl World {
    #[new]
    #[pyo3(signature = (size, food_regrow_rate=0.03, food_cap=1.0, pop_cap=1000, seed=None, n_food_patches=25))]
    fn new(size: u32, food_regrow_rate: f32, food_cap: f32, pop_cap: usize, seed: Option<u64>, n_food_patches: u32) -> Self {
        let seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = Pcg64::seed_from_u64(seed);
        let terrain = Terrain::generate(size, &mut rng);
        let fields = Fields::new(size, food_cap, &terrain, &mut rng, n_food_patches);
        let enc_len = individuals::LATENT_DIM * individuals::SENSE_DIM;
        // Xavier-ish scale so the latent starts well inside tanh's useful
        // range instead of saturating on the first forward pass.
        let enc_scale = (1.0 / individuals::SENSE_DIM as f32).sqrt();
        let shared_enc_w: Vec<f32> = (0..enc_len)
            .map(|_| rng.random_range(-1.0f32..1.0f32) * enc_scale * 2.0)
            .collect();
        let shared_enc_b: Vec<f32> = (0..individuals::LATENT_DIM).map(|_| 0.0f32).collect();
        World {
            size,
            pop_cap,
            rng,
            tick_count: 0,
            sim_time: 0.0,
            dt: 0.1,
            individuals: Individuals::with_capacity(pop_cap),
            pixels: PixelArena::new(),
            corpses: Vec::new(),
            terrain,
            fields,
            weather: Weather::None,
            weather_ticks_left: 0,
            food_regrow_multiplier: 1.0,
            metabolism_multiplier: 1.0,
            maturity_multiplier: 1.0,
            day_light: 0.5,
            reproductions: 0,
            fights: 0,
            deaths: 0,
            scavenged: 0,
            deaths_starved: 0,
            deaths_predation: 0,
            deaths_popcap: 0,
            deaths_crowding: 0,
            whale_falls: 0,
            food_regrow_rate,
            food_cap,
            timings: Vec::new(),
            experience_log: Vec::new(),
            pathogen_damage_rate: PATHOGEN_DAMAGE_RATE,
            repro_cost_per_part: REPRODUCE_COST_PER_PART,
            graze_mass_ref: GRAZE_MASS_REF,
            blind_smell_range: FOOD_SMELL_RANGE_BLIND,
            sight_range_per_eye: FOOD_SIGHT_RANGE_PER_EYE,
            part_metabolism: PART_METABOLISM,
            metabolic_exponent: METABOLIC_EXPONENT,
            collision_stiffness: COLLISION_STIFFNESS,
            contact_correction: CONTACT_CORRECTION,
            brain_noise: 0.0,
            space_pressure_tolerance: SPACE_PRESSURE_TOLERANCE,
            space_pressure_mortality: SPACE_PRESSURE_MORTALITY,
            snow_strength: SNOW_BLOOM_STRENGTH,
            snow_plumes: SNOW_PLUMES_PER_TICK,
            thermal_noise: THERMAL_NOISE,
            growth_tip_weight: GROWTH_STRAIGHT_TIP_WEIGHT,
            freeze_locomotion: None,
            angular_damping: ANGULAR_DAMPING,
            policy_distill_rate: POLICY_DISTILL_RATE,
            meal_energy_per_part: CORPSE_ENERGY_PER_PIXEL,
            gravity: GRAVITY,
            shared_enc_w,
            shared_enc_b,
            shared_policy: None,
        }
    }

    fn spawn_random(&mut self, n: u32) {
        for _ in 0..n {
            let mut pos = [
                self.rng.random_range(0.0..self.size as f32),
                self.rng.random_range(0.0..self.size as f32),
            ];
            for _ in 0..20 {
                // Avoid Rock (impassable) and Sand (now a real solid surface
                // most founders can't dig through) alike -- a founder should
                // start able to move freely, not spawn already resting
                // inside a medium it has no way to have earned entry into.
                let k = self.terrain.at(pos[0] as u32, pos[1] as u32);
                if k != TerrainKind::Rock && k != TerrainKind::Sand {
                    break;
                }
                pos = [
                    self.rng.random_range(0.0..self.size as f32),
                    self.rng.random_range(0.0..self.size as f32),
                ];
            }
            let color = [
                self.rng.random_range(80..255) as u8,
                self.rng.random_range(80..255) as u8,
                self.rng.random_range(80..255) as u8,
            ];
            let slot = individuals::spawn_founder(&mut self.individuals, &mut self.pixels, &mut self.rng, pos, color);
            let extra = self.rng.random_range(2..4);
            for _ in 0..extra {
                individuals::grow_one_pixel(&mut self.individuals, &mut self.pixels, &mut self.rng, slot);
            }
            self.individuals.birth_size[slot] = self.individuals.pixel_count[slot];
        }
    }

    fn tick(&mut self) {
        physics::tick(self);
    }

    fn timings<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        for (k, v) in &self.timings {
            d.set_item(*k, *v).unwrap();
        }
        d
    }

    /// Test-only hook: spawn one individual with a specific bend_amplitude
    /// (bypassing spawn_random's randomization) and no gravity/other
    /// individuals, so thrust-vs-amplitude can be checked without the
    /// population-level confounds (reproduction, collision, terrain) that
    /// make a live-population correlation test noisy.
    fn debug_spawn_controlled(&mut self, pos: [f32; 2], bend_amplitude: f32) -> u64 {
        let color = [200u8, 200, 200];
        let slot = individuals::spawn_founder(&mut self.individuals, &mut self.pixels, &mut self.rng, pos, color);
        self.individuals.bend_amplitude[slot] = bend_amplitude;
        self.individuals.heading[slot] = 0.0;
        for _ in 0..5 {
            individuals::grow_one_pixel(&mut self.individuals, &mut self.pixels, &mut self.rng, slot);
        }
        self.individuals.id[slot]
    }

    fn debug_thrust_force(&self, id: u64) -> Option<[f32; 2]> {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                let pos = physics::world_positions(self, slot, self.sim_time);
                let vel = physics::pixel_velocities(self, slot, self.sim_time, 0.02);
                return Some(physics::fluid_thrust_force(self, slot, &pos, &vel));
            }
        }
        None
    }

    fn debug_reproduction_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let list = PyList::empty(py);
        for slot in 0..self.individuals.len() {
            if !self.individuals.alive[slot] { continue; }
            let d = PyDict::new(py);
            d.set_item("id", self.individuals.id[slot]).unwrap();
            d.set_item("energy", self.individuals.energy[slot]).unwrap();
            d.set_item("age", self.individuals.age[slot]).unwrap();
            d.set_item("size", self.individuals.pixel_count[slot]).unwrap();
            d.set_item("birth_size", self.individuals.birth_size[slot]).unwrap();
            d.set_item("size_scale", self.individuals.size_scale[slot]).unwrap();
            d.set_item("ticks_since_fed", self.individuals.ticks_since_fed[slot]).unwrap();
            d.set_item("ticks_since_reproduced", self.individuals.ticks_since_reproduced[slot]).unwrap();
            d.set_item("female", self.individuals.female[slot]).unwrap();
            list.append(d).unwrap();
        }
        list
    }

    fn debug_nearby_mate_count(&self, id: u64) -> Option<usize> {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                let alive_slots: Vec<usize> = (0..self.individuals.len()).filter(|&s| self.individuals.alive[s]).collect();
                let grid = crate::spatial::SpatialGrid::build(self.size as f32, alive_slots.iter().map(|&s| (s as u32, self.individuals.root_pos[s])));
                let pos = self.individuals.root_pos[slot];
                let count = grid.nearby(pos).into_iter()
                    .filter(|&o| o as usize != slot && ((self.individuals.root_pos[o as usize][0]-pos[0]).powi(2) + (self.individuals.root_pos[o as usize][1]-pos[1]).powi(2)).sqrt() < crate::MATE_RADIUS)
                    .count();
                return Some(count);
            }
        }
        None
    }

    /// Test-only hook: force `attacker_id` to be latched onto `target_id`,
    /// bypassing the RNG-gated bite-and-stickiness roll in resolve_collision,
    /// so the attachment-starvation deadlock (individuals "fighting forever,
    /// stuck in place") can be reproduced and its fix verified deterministically.
    fn debug_force_attach(&mut self, attacker_id: u64, target_id: u64) {
        let mut attacker_slot = None;
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == attacker_id { attacker_slot = Some(slot); }
        }
        if let Some(a) = attacker_slot {
            self.individuals.attached_to[a] = target_id as i64;
        }
    }

    /// Test-only hook: (threat_dx, threat_dy, threat_proximity, prey_dx,
    /// prey_dy, prey_proximity, mate_dx, mate_dy, mate_proximity) as
    /// actually computed for `id`'s own sense vector this instant.
    fn debug_vision(&self, id: u64) -> Option<[f32; 9]> {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                let alive_slots: Vec<usize> = (0..self.individuals.len()).filter(|&s| self.individuals.alive[s]).collect();
                let grid = crate::spatial::SpatialGrid::build(self.size as f32, alive_slots.iter().map(|&s| (s as u32, self.individuals.root_pos[s])));
                return Some(physics::vision(self, slot, &grid));
            }
        }
        None
    }

    fn debug_set_position(&mut self, id: u64, x: f32, y: f32) {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                self.individuals.root_pos[slot] = [x, y];
            }
        }
    }

    fn debug_set_anchor_strength(&mut self, id: u64, value: f32) {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                self.individuals.anchor_strength[slot] = value;
            }
        }
    }

    fn debug_set_color(&mut self, id: u64, r: u8, g: u8, b: u8) {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                self.individuals.color[slot] = [r, g, b];
            }
        }
    }

    fn debug_camouflage_effectiveness(&self, id: u64) -> Option<f32> {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                return Some(combat::camouflage_effectiveness(self, slot));
            }
        }
        None
    }

    /// Test-only hook: is `id` currently held by someone else's attachment?
    fn debug_is_captured(&self, id: u64) -> bool {
        for slot in 0..self.individuals.len() {
            if !self.individuals.alive[slot] { continue; }
            if self.individuals.attached_to[slot] == id as i64 { return true; }
        }
        false
    }

    fn debug_is_alive(&self, id: u64) -> bool {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id { return true; }
        }
        false
    }

    fn debug_set_energy(&mut self, id: u64, energy: f32) {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                self.individuals.energy[slot] = energy;
            }
        }
    }

    /// Test-only hook: what the new graded combat math (combat.rs) would
    /// produce for a hit between these two individuals at a given attacker
    /// pixel speed, without needing to stage real positions/collision
    /// timing -- isolates whether "a small attacker can no longer one-shot
    /// a much bigger target" actually holds as body size grows, independent
    /// of everything else in resolve_collision.
    fn debug_combat_damage(&self, attacker_id: u64, target_id: u64, speed: f32) -> Option<(f32, f32, f32, f32)> {
        let mut attacker_slot = None;
        let mut target_slot = None;
        for slot in 0..self.individuals.len() {
            if !self.individuals.alive[slot] { continue; }
            if self.individuals.id[slot] == attacker_id { attacker_slot = Some(slot); }
            if self.individuals.id[slot] == target_id { target_slot = Some(slot); }
        }
        let (a, t) = (attacker_slot?, target_slot?);
        let power = combat::attacker_power(self, a, speed);
        let armor = combat::effective_toughness(self, t);
        let body_dmg = combat::kinetic_damage(power, armor, false);
        let head_dmg = combat::kinetic_damage(power, armor, true);
        Some((power, armor, body_dmg, head_dmg))
    }

    fn debug_root_pos(&self, id: u64) -> Option<[f32; 2]> {
        for slot in 0..self.individuals.len() {
            if self.individuals.alive[slot] && self.individuals.id[slot] == id {
                return Some(self.individuals.root_pos[slot]);
            }
        }
        None
    }

    // --- state accessors for the Python web server ---

    fn tick_count(&self) -> u64 { self.tick_count }
    /// Continuous simulated clock, in simulated seconds -- what the bend-
    /// wave animation is actually a function of. Exposed so the frontend
    /// can reconstruct exact intermediate poses at any real point in time
    /// via the same formula the engine uses (see world_positions_at),
    /// instead of linearly interpolating raw joint positions between two
    /// published snapshots, which visibly aliases a fast periodic motion
    /// once more than a couple of ticks separate those snapshots.
    fn sim_time(&self) -> f32 { self.sim_time }
    fn day_light(&self) -> f32 { self.day_light }

    /// User interaction: drop food at a world position (e.g. a click on
    /// the frontend canvas). A real, lasting change to that patch (see
    /// Fields::add_food_at), not just a scripted spawn -- individuals find
    /// and eat it through the exact same food-sensing/eating path as any
    /// other food.
    fn add_food_at(&mut self, x: f32, y: f32, amount: f32) {
        let food_cap = self.food_cap;
        self.fields.add_food_at(self.size, x, y, amount, 4.0, food_cap);
    }
    fn population(&self) -> usize { self.individuals.alive.iter().filter(|a| **a).count() }
    fn events<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("reproductions", self.reproductions).unwrap();
        d.set_item("fights", self.fights).unwrap();
        d.set_item("deaths", self.deaths).unwrap();
        d.set_item("scavenged", self.scavenged).unwrap();
        d.set_item("starved", self.deaths_starved).unwrap();
        d.set_item("eaten", self.deaths_predation).unwrap();
        d.set_item("culled", self.deaths_popcap).unwrap();
        d.set_item("crowded", self.deaths_crowding).unwrap();
        d.set_item("whale_falls", self.whale_falls).unwrap();
        d
    }
    fn weather_name(&self) -> Option<&'static str> {
        match self.weather {
            Weather::None => None,
            Weather::FoodBloom => Some("food_bloom"),
            Weather::ColdSnap => Some("cold_snap"),
            Weather::FertileSurge => Some("fertile_surge"),
        }
    }

    fn food<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.food_2d(self.size).into_pyarray(py) }
    fn pheromone<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.pheromone_2d(self.size).into_pyarray(py) }
    fn blood<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.blood_2d(self.size).into_pyarray(py) }
    fn acid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.acid_2d(self.size).into_pyarray(py) }
    fn light<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.light_2d(self.size).into_pyarray(py) }
    fn quorum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.quorum_2d(self.size).into_pyarray(py) }
    fn territory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> { self.fields.territory_2d(self.size).into_pyarray(py) }

    /// Pops every buffered ExperienceRow and hands it back as numpy arrays
    /// (ids, ticks, sense [N x SENSE_DIM], action [N x ACT_DIM], energy) --
    /// the raw material for the not-yet-built async GPU training process.
    /// Returns None if the buffer is empty (avoids allocating an empty
    /// Array2, whose shape would be ambiguous at N=0 anyway).
    fn drain_experience_log<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        if self.experience_log.is_empty() {
            return None;
        }
        let rows = std::mem::take(&mut self.experience_log);
        let n = rows.len();
        let sense_dim = individuals::SENSE_DIM;
        let act_dim = individuals::ACT_DIM;
        let mut ids = Vec::with_capacity(n);
        let mut ticks = Vec::with_capacity(n);
        let mut sense_flat = Vec::with_capacity(n * sense_dim);
        let mut action_flat = Vec::with_capacity(n * act_dim);
        let mut energy = Vec::with_capacity(n);
        let mut reward = Vec::with_capacity(n);
        for row in rows {
            ids.push(row.id);
            ticks.push(row.tick);
            sense_flat.extend_from_slice(&row.sense);
            action_flat.extend_from_slice(&row.action);
            energy.push(row.energy);
            reward.push(row.reward);
        }
        let sense_arr = Array2::from_shape_vec((n, sense_dim), sense_flat).unwrap();
        let action_arr = Array2::from_shape_vec((n, act_dim), action_flat).unwrap();
        let d = PyDict::new(py);
        d.set_item("ids", ids.into_pyarray(py)).unwrap();
        d.set_item("ticks", ticks.into_pyarray(py)).unwrap();
        d.set_item("sense", sense_arr.into_pyarray(py)).unwrap();
        d.set_item("action", action_arr.into_pyarray(py)).unwrap();
        d.set_item("energy", PyArray1::from_vec(py, energy)).unwrap();
        d.set_item("reward", PyArray1::from_vec(py, reward)).unwrap();
        Some(d)
    }
    fn terrain_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> { self.terrain.as_2d().into_pyarray(py) }

    /// Full per-individual state for rendering + species/chronicle logic on
    /// the Python side: one dict per alive individual.
    /// Everything one animal's brain is computing right now: the labelled
    /// sense vector going in, the shared perception latent, its own hidden
    /// layer, and the actions coming out -- plus the anatomy those senses and
    /// actions belong to. For looking at a specimen rather than guessing what
    /// it is doing.
    fn brain_state<'py>(&self, py: Python<'py>, id: u64) -> Option<Bound<'py, PyDict>> {
        let slot = *self.individuals.id_to_slot.get(&id)?;
        if !self.individuals.alive[slot] {
            return None;
        }
        let grid = crate::spatial::SpatialGrid::build(
            self.size as f32,
            (0..self.individuals.alive.len())
                .filter(|&s| self.individuals.alive[s])
                .map(|s| (s as u32, self.individuals.root_pos[s])),
        );
        let (sense, _) = crate::physics::sense(self, slot, &grid);
        let (latent, hidden, act) =
            self.individuals.decide_traced(slot, &sense, &self.shared_enc_w, &self.shared_enc_b);

        let d = PyDict::new(py);
        d.set_item("id", id).unwrap();
        d.set_item("sense", sense.to_vec()).unwrap();
        d.set_item("sense_labels", individuals::SENSE_LABELS.to_vec()).unwrap();
        d.set_item("latent", latent).unwrap();
        d.set_item("hidden", hidden).unwrap();
        d.set_item("act", act).unwrap();
        d.set_item("act_labels", individuals::ACT_LABELS.to_vec()).unwrap();
        let offset = self.individuals.pixel_offset[slot] as usize;
        let count = self.individuals.pixel_count[slot] as usize;
        let part_type: Vec<u32> =
            (0..count).map(|k| self.pixels.part_type[offset + k] as u32).collect();
        d.set_item("part_type", part_type).unwrap();
        d.set_item("energy", self.individuals.energy[slot]).unwrap();
        d.set_item("age", self.individuals.age[slot]).unwrap();
        d.set_item("heading", self.individuals.heading[slot]).unwrap();
        d.set_item("velocity", self.individuals.velocity[slot].to_vec()).unwrap();
        Some(d)
    }

    fn individuals_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let list = PyList::empty(py);
        // Debugging aid for a reported "teleports across the map, then
        // dies" bug: attachment drag sets a captured target's root_pos
        // directly every tick without ever touching its velocity, which
        // this state dump otherwise can't distinguish from a real physics
        // teleport. Built once per call (O(n)), not per-individual.
        let captured_slots: std::collections::HashSet<usize> = (0..self.individuals.alive.len())
            .filter(|&s| self.individuals.alive[s])
            .filter_map(|s| self.individuals.resolve_attached_target(s))
            .collect();
        for slot in 0..self.individuals.alive.len() {
            if !self.individuals.alive[slot] {
                continue;
            }
            let positions = physics::world_positions(self, slot, self.sim_time);
            let count = self.individuals.pixel_count[slot] as usize;
            let offset = self.individuals.pixel_offset[slot] as usize;
            let parents: Vec<i64> = (0..count).map(|k| self.pixels.parent_idx[offset + k] as i64).collect();
            let rest_angle: Vec<f32> = (0..count).map(|k| self.pixels.rest_angle[offset + k]).collect();
            let flex: Vec<f32> = (0..count).map(|k| self.pixels.flex[offset + k]).collect();
            let storage: Vec<f32> = (0..count).map(|k| self.pixels.storage[offset + k]).collect();
            let part_size: Vec<f32> = (0..count).map(|k| self.pixels.size[offset + k]).collect();
            // Girth is published SEPARATELY from size and must stay that way:
            // the client's forward-kinematics port uses `part_size` for
            // segment length and has to mirror the engine exactly, while the
            // organ thickness profile applies only to how wide a part is.
            // Folding the profile into `part_size` would silently shorten
            // every tentacle on screen relative to where the engine actually
            // put it.
            let part_girth: Vec<f32> = (0..count).map(|k| crate::pixels::girth(&self.pixels, offset + k)).collect();
            // u32, not u8: PyO3 maps Vec<u8> to Python `bytes`, which orjson
            // refuses to serialize -- every publish then failed and the live
            // page froze at tick 0 while the simulation itself ran on fine.
            let part_type: Vec<u32> = (0..count).map(|k| self.pixels.part_type[offset + k] as u32).collect();
            let mirror_sign: Vec<f32> = (0..count).map(|k| self.pixels.mirror_sign[offset + k]).collect();
            let health_frac: Vec<f32> = (0..count).map(|k| {
                let max_health = BASE_PIXEL_HEALTH * self.pixels.size[offset + k];
                if max_health > 0.0 { (self.pixels.health[offset + k] / max_health).clamp(0.0, 1.0) } else { 1.0 }
            }).collect();
            let pos_list: Vec<[f32; 2]> = positions;

            let d = PyDict::new(py);
            d.set_item("id", self.individuals.id[slot]).unwrap();
            d.set_item("positions", pos_list).unwrap();
            d.set_item("parents", parents).unwrap();
            d.set_item("rest_angle", rest_angle).unwrap();
            d.set_item("flex", flex).unwrap();
            d.set_item("storage", storage).unwrap();
            d.set_item("part_size", part_size).unwrap();
            d.set_item("part_girth", part_girth).unwrap();
            d.set_item("part_type", part_type).unwrap();
            d.set_item("mirror_sign", mirror_sign).unwrap();
            d.set_item("health_frac", health_frac).unwrap();
            d.set_item("captured", captured_slots.contains(&slot)).unwrap();
            let c = self.individuals.color[slot];
            d.set_item("color", [c[0] as u32, c[1] as u32, c[2] as u32]).unwrap();
            d.set_item("energy", self.individuals.energy[slot]).unwrap();
            let v = self.individuals.velocity[slot];
            d.set_item("speed", (v[0] * v[0] + v[1] * v[1]).sqrt()).unwrap();
            d.set_item("size", count).unwrap();
            d.set_item("size_scale", self.individuals.size_scale[slot]).unwrap();
            d.set_item("age", self.individuals.age[slot]).unwrap();
            let crawling = self.individuals.root_pos[slot][1] < CRAWL_FLOOR_THRESHOLD
                && self.individuals.crawl_affinity[slot] > 0.05;
            d.set_item("crawling", crawling).unwrap();
            d.set_item("bend_amplitude", self.individuals.bend_amplitude[slot]).unwrap();
            d.set_item("bend_frequency", self.individuals.bend_frequency[slot]).unwrap();
            d.set_item("bend_phase", self.individuals.bend_phase[slot]).unwrap();
            d.set_item("heading", self.individuals.heading[slot]).unwrap();
            // The frontend reconstructs bodies with the same FK formula, so it
            // needs the same effort value or its animation desynchronises.
            d.set_item("swim_gain", self.individuals.swim_gain[slot]).unwrap();
            d.set_item("turn_curvature", self.individuals.turn_curvature[slot]).unwrap();
            d.set_item("axis_offset", self.individuals.axis_offset[slot]).unwrap();
            // bite_force/toughness/stickiness/weight_transmission_rate/
            // crawl_affinity/acid_secretion/light_emission deliberately are
            // NOT here: the frontend never reads them per-individual (only
            // as the species table's per-lineage averages), so carrying
            // them in every dict was ~42k pointless key insertions and
            // float boxings per publish. They're aggregated natively now --
            // see `species_summary` above.
            d.set_item("female", self.individuals.female[slot]).unwrap();
            list.append(d).unwrap();
        }
        list
    }

    /// Test-only: the raw kin-weighted conspecific density for every alive
    /// individual, in the same order as `individuals_state()`, so the
    /// pathogen response curve can be calibrated against the distribution
    /// that actually occurs instead of a guessed scale.
    fn debug_conspecific_densities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let vals = physics::conspecific_densities(self);
        PyArray1::from_vec(py, vals)
    }

    /// Test-only: wipes every living individual's learned weights back to
    /// random. Used to measure whether evolved behaviour actually beats
    /// random behaviour -- the control condition for "is intelligence
    /// contributing anything at all".
    fn debug_randomize_all_brains(&mut self) {
        for slot in 0..self.individuals.alive.len() {
            if self.individuals.alive[slot] {
                let mut rng = std::mem::replace(&mut self.rng, Pcg64::seed_from_u64(0));
                self.individuals.randomize_brain(slot, &mut rng);
                self.rng = rng;
            }
        }
    }

    /// Installs new shared-encoder weights, as produced by the asynchronous
    /// training process. Every creature immediately perceives through the
    /// improved representation while keeping its own evolved decision layer
    /// untouched -- which is the whole point of separating perception from
    /// decision. Rejected (rather than panicking) if the shapes are wrong,
    /// since this is fed from a file written by another process.
    fn set_shared_encoder(&mut self, w: Vec<f32>, b: Vec<f32>) -> bool {
        let want_w = individuals::LATENT_DIM * individuals::SENSE_DIM;
        if w.len() != want_w || b.len() != individuals::LATENT_DIM {
            return false;
        }
        self.shared_enc_w = w;
        self.shared_enc_b = b;
        true
    }

    /// Installs a learned baseline policy. Shapes must match an individual's
    /// decoder exactly, since that is what it gets blended into.
    fn set_shared_policy(&mut self, w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>) -> bool {
        let (h, l, a) = (individuals::HIDDEN_DIM, individuals::LATENT_DIM, individuals::ACT_DIM);
        if w1.len() != h * l || b1.len() != h || w2.len() != a * h || b2.len() != a {
            return false;
        }
        self.shared_policy = Some((w1, b1, w2, b2));
        true
    }

    /// The current shared-encoder weights, so the trainer can warm-start
    /// from what the world is actually using.
    fn shared_encoder<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("w", PyArray1::from_slice(py, &self.shared_enc_w)).unwrap();
        d.set_item("b", PyArray1::from_slice(py, &self.shared_enc_b)).unwrap();
        d.set_item("latent_dim", individuals::LATENT_DIM).unwrap();
        d.set_item("sense_dim", individuals::SENSE_DIM).unwrap();
        d
    }

    /// Test-only: overrides the energy a meal yields per part of prey.
    fn debug_set_meal_energy(&mut self, v: f32) { self.meal_energy_per_part = v; }

    /// Test-only: overrides how hard learned instinct is pressed into newborns.
    fn debug_set_policy_distill_rate(&mut self, v: f32) { self.policy_distill_rate = v; }

    /// Test-only: overrides angular damping.
    fn debug_set_angular_damping(&mut self, v: f32) { self.angular_damping = v; }

    /// Test-only: forces every part of a body to be symmetric (or not), so a
    /// bilateral and a lopsided body plan can be compared directly.
    fn debug_set_symmetry(&mut self, id: u64, symmetric: bool) -> bool {
        match self.individuals.id_to_slot.get(&id) {
            Some(&slot) if self.individuals.alive[slot] => {
                let off = self.individuals.pixel_offset[slot] as usize;
                let n = self.individuals.pixel_count[slot] as usize;
                for k in 0..n {
                    self.pixels.symmetric[off + k] = symmetric;
                }
                true
            }
            _ => false,
        }
    }

    /// Test-only: pins body curvature and swim effort, taking the brain out
    /// of the loop so propulsion can be measured on its own.
    fn debug_freeze_locomotion(&mut self, curvature: f32, swim_gain: f32) {
        self.freeze_locomotion = Some((curvature, swim_gain));
    }

    /// Test-only: overrides how strongly growth prefers extending a tip.
    fn debug_set_growth_tip_weight(&mut self, v: f32) { self.growth_tip_weight = v; }

    /// Test-only: overrides gravity.
    fn debug_set_gravity(&mut self, v: f32) { self.gravity = v; }

    /// Test-only: forces an individual's heading, to hold a course while its
    /// propulsion is measured.
    fn debug_set_heading(&mut self, id: u64, h: f32) -> bool {
        match self.individuals.id_to_slot.get(&id) {
            Some(&slot) if self.individuals.alive[slot] => {
                self.individuals.heading[slot] = h;
                self.individuals.angular_velocity[slot] = 0.0;
                true
            }
            _ => false,
        }
    }

    /// Test-only: grows one part, so a probe body can be built deterministically.
    fn debug_grow(&mut self, id: u64) -> bool {
        match self.individuals.id_to_slot.get(&id) {
            Some(&slot) if self.individuals.alive[slot] => {
                individuals::grow_one_pixel(&mut self.individuals, &mut self.pixels, &mut self.rng, slot)
            }
            _ => false,
        }
    }

    /// Test-only: the body's own anterior axis offset.
    fn debug_axis_offset(&self, id: u64) -> Option<f32> {
        self.individuals.id_to_slot.get(&id).map(|&s| self.individuals.axis_offset[s])
    }

    /// Test-only: overrides how far a creature can locate food with no eyes,
    /// and how much each eye extends that. Setting blind range back up to the
    /// sight range restores the old behaviour where foraging was ungated.
    fn debug_set_smell_ranges(&mut self, blind: i32, per_eye: i32) {
        self.blind_smell_range = blind;
        self.sight_range_per_eye = per_eye;
    }

    /// Test-only: overrides per-part upkeep, indexed body, eye, mouth, gut,
    /// tentacle, armor, flipper.
    /// Test-only: scales the whole metabolic economy. This is the lever on
    /// POPULATION rather than on body size: with upkeep sublinear in size, a
    /// higher multiplier hurts small animals disproportionately, because they
    /// pay near the full per-part rate while a large body pays a fraction of
    /// it. So it should thin the world out without flattening the animals.
    fn debug_set_metabolism_multiplier(&mut self, v: f32) {
        self.metabolism_multiplier = v;
    }

    fn debug_set_part_metabolism(&mut self, v: Vec<f32>) {
        for (i, x) in v.iter().take(crate::pixels::PART_KIND_COUNT as usize).enumerate() {
            self.part_metabolism[i] = *x;
        }
    }

    /// Test-only: the mean of the food field. The priority-effect hypothesis
    /// for why identical settings give either a worm world or a large-bodied
    /// one predicts the food field separates BEFORE body size does, so it has
    /// to be observable.
    fn debug_mean_food(&self) -> f32 {
        let f = &self.fields.food;
        if f.is_empty() { return 0.0; }
        f.iter().sum::<f32>() / f.len() as f32
    }

    /// Test-only: overrides plankton production.
    fn debug_set_snow(&mut self, strength: f32, plumes: u32) {
        self.snow_strength = strength;
        self.snow_plumes = plumes;
    }

    /// Test-only: overrides density-dependent mortality.
    fn debug_set_space_pressure(&mut self, tolerance: f32, mortality: f32) {
        self.space_pressure_tolerance = tolerance;
        self.space_pressure_mortality = mortality;
    }

    /// Test-only: 0.0 leaves brains alone, 1.0 replaces every decision with
    /// deterministic noise. See `brain_noise`.
    fn debug_set_brain_noise(&mut self, v: f32) {
        self.brain_noise = v;
    }

    /// Test-only: overrides the positional half of contact resolution.
    /// 0.0 is the old force-only solver.
    fn debug_set_contact_correction(&mut self, v: f32) {
        self.contact_correction = v;
    }

    /// Test-only: overrides how hard overlapping bodies push apart.
    fn debug_set_collision_stiffness(&mut self, v: f32) {
        self.collision_stiffness = v;
    }

    /// Test-only: overrides the Kleiber metabolic scaling exponent.
    fn debug_set_metabolic_exponent(&mut self, v: f32) {
        self.metabolic_exponent = v;
    }

    /// Test-only: overrides the thermal noise floor.
    fn debug_set_thermal_noise(&mut self, v: f32) {
        self.thermal_noise = v;
    }

    /// Test-only: overrides the mass at which grazing yield halves.
    fn debug_set_graze_mass_ref(&mut self, v: f32) {
        self.graze_mass_ref = v;
    }

    /// Test-only: overrides the per-part reproduction cost for a sweep.
    fn debug_set_repro_cost_per_part(&mut self, v: f32) {
        self.repro_cost_per_part = v;
    }

    /// Test-only: overrides the pathogen damage rate for an A/B run.
    fn debug_set_pathogen_rate(&mut self, rate: f32) {
        self.pathogen_damage_rate = rate;
    }

    /// Per-lineage (color-keyed) trait averages, aggregated natively over
    /// the SoA arrays. This used to run in Python over the full
    /// `individuals_state()` list-of-dicts, which forced SEVEN traits to be
    /// carried in every one of those ~6000 per-individual dicts purely so
    /// Python could sum them back up -- ~42k wasted key insertions and
    /// float boxings per publish, for numbers the engine can accumulate in
    /// one cheap pass over contiguous arrays. Output shape deliberately
    /// matches what the Python `species_summary` produced (same keys, same
    /// 2-decimal rounding) so the frontend's species table is unchanged;
    /// naming and the chronicle's lineage bookkeeping stay in Python, where
    /// the stateful/textual logic lives.
    fn species_summary<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        const N_TRAITS: usize = 11;
        // color -> (count, size_sum, max_size, trait_sums)
        let mut acc: std::collections::HashMap<[u8; 3], (u32, u64, u32, [f64; N_TRAITS])> =
            std::collections::HashMap::new();
        for slot in 0..self.individuals.alive.len() {
            if !self.individuals.alive[slot] {
                continue;
            }
            let count = self.individuals.pixel_count[slot];
            let e = acc
                .entry(self.individuals.color[slot])
                .or_insert((0, 0, 0, [0.0; N_TRAITS]));
            e.0 += 1;
            e.1 += count as u64;
            if count > e.2 {
                e.2 = count;
            }
            let t = &mut e.3;
            t[0] += self.individuals.size_scale[slot] as f64;
            t[1] += self.individuals.energy[slot] as f64;
            t[2] += self.individuals.bend_amplitude[slot] as f64;
            t[3] += self.individuals.bite_force[slot] as f64;
            t[4] += self.individuals.toughness[slot] as f64;
            t[5] += self.individuals.stickiness[slot] as f64;
            t[6] += self.individuals.weight_transmission_rate[slot] as f64;
            t[7] += self.individuals.crawl_affinity[slot] as f64;
            t[8] += self.individuals.acid_secretion[slot] as f64;
            t[9] += self.individuals.light_emission[slot] as f64;
            t[10] += self.individuals.disease_resistance[slot] as f64;
        }

        const TRAIT_KEYS: [&str; N_TRAITS] = [
            "avg_size_scale", "avg_energy", "avg_bend_amplitude", "avg_bite_force",
            "avg_toughness", "avg_stickiness", "avg_weight_transmission",
            "avg_crawl_affinity", "avg_acid_secretion", "avg_light_emission",
            "avg_disease_resistance",
        ];
        let round2 = |v: f64| (v * 100.0).round() / 100.0;

        let mut rows: Vec<([u8; 3], (u32, u64, u32, [f64; N_TRAITS]))> = acc.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        let list = PyList::empty(py);
        for (color, (count, size_sum, max_size, traits)) in rows {
            let n = count as f64;
            let d = PyDict::new(py);
            d.set_item("color", [color[0] as u32, color[1] as u32, color[2] as u32]).unwrap();
            d.set_item("count", count).unwrap();
            d.set_item("avg_size", round2(size_sum as f64 / n)).unwrap();
            d.set_item("max_size", max_size).unwrap();
            for (key, sum) in TRAIT_KEYS.iter().zip(traits.iter()) {
                d.set_item(*key, round2(sum / n)).unwrap();
            }
            list.append(d).unwrap();
        }
        list
    }

    fn corpses_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let list = PyList::empty(py);
        for c in &self.corpses {
            let positions: Vec<[f32; 2]> = c.local_shape.iter().map(|o| [o[0] + c.root_pos[0], o[1] + c.root_pos[1]]).collect();
            let d = PyDict::new(py);
            d.set_item("positions", positions).unwrap();
            d.set_item("color", [c.color[0] as u32, c.color[1] as u32, c.color[2] as u32]).unwrap();
            d.set_item("energy", c.energy).unwrap();
            list.append(d).unwrap();
        }
        list
    }
}

#[pymodule]
fn rust_world(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
