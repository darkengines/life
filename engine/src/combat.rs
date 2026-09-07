//! Combat resolution as a graded, multi-mechanism system, not a single
//! scalar pass/fail gate. The old rule compared one number (attacker
//! speed*bite_force) against another (the target's flat evolved
//! `toughness`) and, if it passed, deleted whatever pixel got hit AND
//! every descendant of it in the joint tree -- in one hit, with zero
//! relationship to how much body the target actually had. That was the
//! literal mechanism behind "a small individual one-shots a much bigger
//! one": not a rare fluke, a coin-flip on one size-independent number on
//! every single swing.
//!
//! This module is the seam for adding more of these mechanisms later
//! (frostbite, electric shock, whatever else earns its own function) --
//! each is a small, independently-testable function of evolved traits and
//! world state, not a branch buried inline in the tick loop. physics.rs's
//! resolve_collision composes them; it doesn't implement any of them.
use crate::terrain::TerrainKind;
use crate::World;

/// A body's real resistance to being cut down: its own evolved `toughness`
/// trait PLUS a term scaling with how much actual body it has (summed
/// per-part `size`, inflated by `size_scale`). This is the actual fix for
/// one-shotting: overwhelming a big body now requires overwhelming its
/// mass, not a number that happens to be the same whether the body has 3
/// pixels or 30.
pub fn effective_toughness(world: &World, target: usize) -> f32 {
    let mass = crate::physics::body_size_sum(world, target) * world.individuals.size_scale[target];
    // Armor plating is real protection on top of the evolved trait and raw
    // mass -- the anatomical route to being hard to kill, as opposed to the
    // simply-being-enormous route.
    (world.individuals.toughness[target] + crate::TOUGHNESS_SIZE_SCALING * mass)
        * crate::physics::armor_multiplier(world, target)
}

/// How hard `attacker` is hitting right now, via one of its own pixels
/// moving at `speed`: evolved bite_force, scaled down when starving.
/// Lethality is tied to a real, currently-held resource (energy), not a
/// fixed trait value regardless of condition -- a starving attacker's bite
/// is genuinely weaker, not just as dangerous as when it's flush.
pub fn attacker_power(world: &World, attacker: usize, speed: f32) -> f32 {
    // A starving animal is weaker, but not harmless. The floor used to be
    // 0.15, which meant a hungry predator hit at a seventh of its strength and
    // so could not fight its way out of starvation at all -- hunger made
    // hunting impossible exactly when hunting was the only option left. Real
    // starving predators are dangerous; desperation is not the same as
    // helplessness. Nothing here tells an animal to attack when hungry -- it
    // simply removes the engine's guarantee that trying would fail.
    let energy_factor = (world.individuals.energy[attacker] / crate::ATTACKER_ENERGY_DAMAGE_REF)
        .clamp(crate::STARVING_ATTACK_FLOOR, 1.5);
    // A strike carries the MASS behind it, not just the speed of the part
    // that lands. Momentum is mass times velocity, and leaving mass out meant
    // a three-part animal moving quickly hit exactly as hard as a forty-part
    // one moving the same way -- so being big bought nothing in a fight, which
    // is most of why size never translated into dominance. Scaled by a root so
    // a heavyweight is a serious opponent without being untouchable.
    let mass = (crate::physics::body_size_sum(world, attacker)
        * world.individuals.size_scale[attacker])
        .max(0.01);
    let heft = (mass / crate::ATTACK_MASS_REF).sqrt().clamp(0.35, 3.5);
    speed * world.individuals.bite_force[attacker] * energy_factor * heft
}

/// Kinetic ("slicing") damage from one landed hit against one target pixel:
/// the excess power over that pixel-owner's effective armor, multiplied up
/// if the hit landed on the head (local pixel index 0 -- the root is
/// always index 0 by construction, a real, discoverable vital point rather
/// than a hardcoded species weakness: any evolved brain that learns to aim
/// there, using senses it already has, gets a mechanical payoff).
pub fn kinetic_damage(power: f32, armor: f32, is_head: bool) -> f32 {
    let raw = (power - armor).max(0.0);
    if is_head { raw * crate::HEADSHOT_MULTIPLIER } else { raw }
}

/// A landed bite also injects the attacker's own evolved acid_secretion
/// into the world's acid field at the bite location -- poison as a
/// genuinely different, slower kill path that does NOT require ever
/// beating the target's toughness, reusing the existing acid
/// damage-over-time + diffusion system instead of a second parallel toxin
/// mechanic. A weak, non-lethal-on-contact attacker with strong venom can
/// still eventually bring down something it could never out-muscle.
pub fn inject_venom(world: &mut World, attacker: usize, world_x: u32, world_y: u32) {
    let potency = world.individuals.acid_secretion[attacker];
    if potency <= 0.01 {
        return;
    }
    let idx = (world_x * world.size + world_y) as usize;
    world.fields.acid[idx] += crate::VENOM_INJECTION_SCALE * potency;
}

/// The world-space hit radius of one specific target pixel: bigger evolved
/// parts (pixels.rs's `size`) are a bigger target, exactly like a bigger
/// animal's flank is easier to land a hit on than a small one's.
pub fn hit_radius(world: &World, target_offset: usize, local_idx: usize) -> f32 {
    crate::COLLISION_RADIUS * (0.5 + 0.5 * crate::pixels::girth(&world.pixels, target_offset + local_idx))
}

/// Slow passive healing: pixels below their max health regenerate a little
/// each tick, paid for in energy -- a body that survives a fight and then
/// finds food can recover, rather than every wound being permanent. Real
/// animals heal; this is that, gated by the same resource (energy) that
/// gates growth and reproduction, not a free ability.
pub fn regenerate(world: &mut World, slot: usize) {
    let offset = world.individuals.pixel_offset[slot] as usize;
    let count = world.individuals.pixel_count[slot] as usize;
    if world.individuals.energy[slot] <= crate::GROWTH_ENERGY_THRESHOLD {
        return; // healing is a luxury, same threshold as growth -- not for a body that's already struggling to eat
    }
    let mut healed_any = false;
    for k in offset..offset + count {
        // Dead tissue does not come back. Healing a wound is one thing;
        // reviving a component that was killed outright would make the
        // killed-in-place outcome a brief inconvenience rather than a real
        // injury, and there would be no reason to fear losing a limb.
        if world.pixels.dead[k] { continue; }
        let max_health = crate::BASE_PIXEL_HEALTH * world.pixels.size[k];
        if world.pixels.health[k] < max_health {
            world.pixels.health[k] = (world.pixels.health[k] + crate::HEALTH_REGEN_RATE).min(max_health);
            healed_any = true;
        }
    }
    if healed_any {
        world.individuals.energy[slot] -= crate::HEALTH_REGEN_ENERGY_COST;
    }
}

/// The local ambient color a predator would see behind `slot`: sand and
/// rock have their own fixed tones (matching the frontend's own terrain
/// rendering), open water's tone shifts with local food/light density
/// exactly like the background the frontend actually draws there. Not a
/// hardcoded "safe color" -- just what's actually behind the body right
/// now, which differs by location, so the same evolved color can be
/// well- or badly-camouflaged depending on where an individual ends up.
fn ambient_color(world: &World, slot: usize) -> [f32; 3] {
    let pos = world.individuals.root_pos[slot];
    let (x, y) = crate::physics::grid_xy(world, pos);
    // Day/night shifts the whole ambient scene brighter/darker -- the same
    // color is genuinely easier to blend into at night and harder at noon,
    // exactly the real reason nocturnal camouflage differs from daytime
    // camouflage. `day` is 0..1; scaled so night doesn't go fully black.
    let day = world.day_light;
    let brightness = 0.4 + 0.6 * day;
    let base = match world.terrain.at(x, y) {
        TerrainKind::Sand => [150.0, 125.0, 85.0],
        TerrainKind::Rock => [58.0, 60.0, 68.0],
        TerrainKind::Empty => {
            let idx = (x * world.size + y) as usize;
            let f = world.fields.food[idx];
            let li = world.fields.light[idx];
            [10.0 + f * 20.0, 15.0 + f * 60.0 + li * 60.0, 20.0 + f * 20.0 + li * 30.0]
        }
    };
    [base[0] * brightness, base[1] * brightness, base[2] * brightness]
}

/// Warning-coloration deterrence: how much a predator hesitates to press
/// an attack once it's already noticed `slot`, as a function of how
/// conspicuous its color is against its surroundings AND how much real
/// defense (toughness or acid/venom) it actually has. Neither loud color
/// alone (just easier to find, via camouflage_effectiveness already being
/// low) nor high toughness alone (armor is already its own thing in
/// kinetic_damage) does anything here -- only the CORRELATION between the
/// two, the same honest-signal logic real aposematic warning coloration
/// relies on. Nothing prevents a "loud color, no real defense" bluff from
/// evolving too; that would just be an interesting, real form of mimicry
/// if it does.
pub fn aposematism_deterrence(world: &World, slot: usize) -> f32 {
    let conspicuousness = 1.0 - camouflage_effectiveness(world, slot);
    let toughness_component = (world.individuals.toughness[slot] / crate::APOSEMATISM_TOUGHNESS_REF).min(1.0);
    let acid_component = (world.individuals.acid_secretion[slot] / crate::APOSEMATISM_ACID_REF).min(1.0);
    let defense = toughness_component.max(acid_component);
    conspicuousness * defense * crate::APOSEMATISM_DETERRENCE_MAX
}

/// How likely `slot` is to go unnoticed by a predator that's otherwise
/// looking right at it: 1.0 for a perfect color match to its surroundings,
/// 0.0 once its color is different enough that blending in stops helping
/// at all (CAMOUFLAGE_COLOR_RANGE). Multiplied by CAMOUFLAGE_DETECTION_
/// PENALTY in resolve_collision so even flawless camouflage doesn't make
/// an individual permanently unfindable -- movement, smell, and everything
/// else evolution didn't spend on color still gives it away sometimes.
pub fn camouflage_effectiveness(world: &World, slot: usize) -> f32 {
    let ambient = ambient_color(world, slot);
    let c = world.individuals.color[slot];
    let dr = c[0] as f32 - ambient[0];
    let dg = c[1] as f32 - ambient[1];
    let db = c[2] as f32 - ambient[2];
    let dist = (dr * dr + dg * dg + db * db).sqrt();
    (1.0 - dist / crate::CAMOUFLAGE_COLOR_RANGE).clamp(0.0, 1.0)
}
