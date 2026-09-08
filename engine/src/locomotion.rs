//! Interchangeable body-physics backends.
//!
//! There are two ways to make these animals move, and which one is right
//! depends on what is being asked of the simulation.
//!
//! The **analytic** backend is what this project has always used: the body is a
//! kinematic chain, its shape is regenerated every tick by a travelling wave
//! passing down the joints, and thrust comes from resolving that motion against
//! resistive-force-theory drag. It is fast, unconditionally stable, and every
//! component's position is a closed-form function of the body plan and the
//! clock -- there is no solver to diverge. What it cannot represent is a body
//! that is genuinely pushed around: joints cannot be forced out of their
//! commanded pose by an impact, a limb cannot be bent by the water it is
//! pushing, and two animals cannot lever against each other.
//!
//! The **rigid-body** backend hands the same body to a real constraint solver:
//! each component is a rigid body, each joint a motorised revolute constraint,
//! and the wave becomes target angles the motors chase rather than positions
//! that are simply assigned. Everything the analytic model cannot express falls
//! out for free -- momentum, recoil, impacts that deform a body mid-stroke,
//! contact forces that resolve properly instead of being approximated by a
//! penalty push. It costs a great deal more, and a solver can be driven
//! unstable in ways a closed-form expression cannot.
//!
//! Neither is simply better, which is exactly why this is a choice made at
//! runtime rather than a rewrite. The interface below is deliberately narrow:
//! a backend is asked for component positions and velocities, and is told to
//! advance the world by a tick. Everything else in the engine -- feeding,
//! metabolism, reproduction, senses -- is written against those two things and
//! does not know or care which backend produced them.
use crate::World;

/// What every backend must be able to do.
///
/// Kept small on purpose. A wider interface would let backends leak their own
/// assumptions into the rest of the engine, and the entire value of having two
/// is that the rest of the engine cannot tell them apart.
pub trait BodyPhysics: Send {
    /// Name, for reporting which engine a measurement came from -- comparing
    /// two backends is worthless if you cannot tell which one you were running.
    fn name(&self) -> &'static str;

    /// Component world positions for one individual at the given time.
    fn positions(&self, world: &World, slot: usize, t: f32) -> Vec<[f32; 2]>;

    /// Component velocities, however this backend knows them: derived from the
    /// kinematics for the analytic model, read from the solver for a rigid-body
    /// one.
    fn velocities(&self, world: &World, slot: usize, positions: &[[f32; 2]], t: f32)
        -> Vec<[f32; 2]>;

    /// Called when a body's plan changes -- birth, growth, fusion, losing a
    /// part -- so a backend holding its own representation can rebuild it. The
    /// analytic model holds nothing and ignores this.
    fn body_changed(&mut self, _world: &World, _slot: usize) {}

    /// Called when an individual dies and its slot is freed.
    fn body_removed(&mut self, _slot: usize) {}

    /// True if this backend integrates motion itself, in which case the
    /// engine's own force integration is skipped for it.
    fn owns_integration(&self) -> bool {
        false
    }
}

/// The kinematic-chain model: a travelling wave down the joints, resolved
/// against fluid drag. Holds no state of its own -- every position is a
/// function of the body plan and the clock, which is precisely why it cannot
/// diverge and why it is so cheap.
pub struct AnalyticBodies;

impl BodyPhysics for AnalyticBodies {
    fn name(&self) -> &'static str {
        "analytic"
    }

    fn positions(&self, world: &World, slot: usize, t: f32) -> Vec<[f32; 2]> {
        crate::physics::world_positions(world, slot, t)
    }

    fn velocities(
        &self,
        world: &World,
        slot: usize,
        positions: &[[f32; 2]],
        t: f32,
    ) -> Vec<[f32; 2]> {
        crate::physics::pixel_velocities_from(world, slot, positions, t, 0.02)
    }
}

/// Which backend a world is running.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Analytic,
    Rigid,
}

impl Backend {
    pub fn from_name(s: &str) -> Option<Backend> {
        match s {
            "analytic" => Some(Backend::Analytic),
            "rigid" | "rapier" => Some(Backend::Rigid),
            _ => None,
        }
    }
}
