//! Rigid-body backend: the same animals, handed to a real constraint solver.
//!
//! Every component becomes a rigid body, every parent-child link a motorised
//! revolute joint, and the undulation wave stops being a set of positions that
//! are assigned and becomes a set of target angles the motors chase. That one
//! change is the whole point of this backend. In the analytic model a body is
//! wherever the wave says it is, so nothing can push it out of shape: a limb
//! cannot be bent by the water it is pushing, an impact cannot deform a body
//! mid-stroke, and two animals cannot lever against one another. Here the pose
//! is the OUTCOME of motor torques fighting drag, contact and momentum, and all
//! of that follows without being modelled specially.
//!
//! Drag is applied per component, as it is in the analytic model and for the
//! same reason: a body in water is not a projectile, and resistive force theory
//! -- separate coefficients along and across each segment -- is what turns an
//! undulation into forward motion rather than a wiggle in place.
//!
//! The honest trade is cost and stability. A solver has to be stepped, its
//! constraints iterated, and it can be driven unstable by torques a closed-form
//! expression would simply have absorbed. It is here to be chosen when the
//! richness is worth that, and not otherwise.
use std::collections::HashMap;

use rapier2d::prelude::*;

use crate::locomotion::BodyPhysics;
use crate::World;

/// One animal's representation inside the solver.
struct RigidBody {
    /// Component index -> rigid body handle, in the same order the arena
    /// stores them, so translating back and forth needs no lookup table.
    parts: Vec<RigidBodyHandle>,
    joints: Vec<ImpulseJointHandle>,
    /// The body plan this was built from. If the animal grows, loses a part or
    /// fuses two, the plan no longer matches and the whole thing is rebuilt --
    /// cheaper and far less error-prone than patching a live constraint graph.
    plan_len: usize,
}

pub struct RigidBodies {
    physics: PhysicsPipeline,
    islands: IslandManager,
    broad: DefaultBroadPhase,
    narrow: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    params: IntegrationParameters,
    gravity: Vector<Real>,
    built: HashMap<usize, RigidBody>,
}

impl RigidBodies {
    pub fn new(dt: f32, gravity_y: f32) -> Self {
        let mut params = IntegrationParameters::default();
        params.dt = dt;
        RigidBodies {
            physics: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad: DefaultBroadPhase::new(),
            narrow: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            params,
            gravity: vector![0.0, -gravity_y],
            built: HashMap::new(),
        }
    }

    /// Build (or rebuild) one animal inside the solver from its body plan.
    ///
    /// Rebuilding wholesale on any change is deliberate. A body here changes by
    /// growing a component, fusing two, or having one bitten off, and each of
    /// those reshapes the constraint graph; patching a live graph for that is a
    /// well-known source of subtle, hard-to-reproduce solver bugs, and these
    /// bodies are tens of parts, so rebuilding one is cheap.
    fn build(&mut self, world: &World, slot: usize) {
        self.destroy(slot);
        let offset = world.individuals.pixel_offset[slot] as usize;
        let count = world.individuals.pixel_count[slot] as usize;
        if count == 0 {
            return;
        }
        // Seed the solver from the analytic pose, so an animal entering the
        // rigid world starts in the shape it already had rather than snapping
        // to some default and exploding as the constraints resolve.
        let pose = crate::physics::analytic_positions(world, slot, world.sim_time);
        let scale = world.individuals.size_scale[slot];

        let mut parts = Vec::with_capacity(count);
        for k in 0..count.min(pose.len()) {
            let girth = crate::pixels::girth(&world.pixels, offset + k) * scale;
            let radius = (girth * 0.5).max(0.05);
            let rb = RigidBodyBuilder::dynamic()
                .translation(vector![pose[k][0], pose[k][1]])
                // Damping stands in for the part of fluid resistance that is
                // not direction-dependent. The direction-dependent part -- the
                // reason an undulation moves an animal forward at all -- is
                // applied as explicit per-segment forces in `apply_drag`.
                .linear_damping(crate::RIGID_LINEAR_DAMPING)
                .angular_damping(crate::RIGID_ANGULAR_DAMPING)
                .build();
            let handle = self.bodies.insert(rb);
            let collider = ColliderBuilder::ball(radius)
                .density(crate::RIGID_DENSITY)
                .friction(0.4)
                .restitution(0.05)
                // Components of the SAME animal must not collide with each
                // other: a body plan is a tree of touching parts, and letting
                // neighbours push each other apart would make every animal
                // detonate on the first step.
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_1,
                    Group::GROUP_1,
                ))
                .build();
            self.colliders
                .insert_with_parent(collider, handle, &mut self.bodies);
            parts.push(handle);
        }

        let mut joints = Vec::new();
        for k in 1..parts.len() {
            let parent = world.pixels.parent_idx[offset + k];
            if parent < 0 {
                continue;
            }
            let p = parent as usize;
            if p >= parts.len() {
                continue;
            }
            // The anchor is the offset from parent to child in the parent's
            // frame; using the actual current separation keeps the joint from
            // having to yank the pair together on the first step.
            let dx = pose[k][0] - pose[p][0];
            let dy = pose[k][1] - pose[p][1];
            let joint = RevoluteJointBuilder::new()
                .local_anchor1(point![dx * 0.5, dy * 0.5])
                .local_anchor2(point![-dx * 0.5, -dy * 0.5])
                .motor_model(MotorModel::ForceBased)
                .build();
            let h = self
                .impulse_joints
                .insert(parts[p], parts[k], joint, true);
            joints.push(h);
        }

        self.built.insert(
            slot,
            RigidBody {
                parts,
                joints,
                plan_len: count,
            },
        );
    }

    fn destroy(&mut self, slot: usize) {
        if let Some(b) = self.built.remove(&slot) {
            for h in b.joints {
                self.impulse_joints.remove(h, true);
            }
            for h in b.parts {
                self.bodies.remove(
                    h,
                    &mut self.islands,
                    &mut self.colliders,
                    &mut self.impulse_joints,
                    &mut self.multibody_joints,
                    true,
                );
            }
        }
    }

    /// Drive each joint toward the angle the wave is currently asking for.
    ///
    /// This is where the two backends genuinely differ. The analytic model
    /// SETS the joint angle; here the motor is asked for it and may not get it,
    /// because the water, a collision, or the animal's own momentum can all
    /// disagree. An animal that commands a hard turn while being shoved will
    /// not complete it, which is the behaviour this backend exists to provide.
    fn drive_motors(&mut self, world: &World, slot: usize, target: &[f32]) {
        let Some(b) = self.built.get(&slot) else { return };
        let offset = world.individuals.pixel_offset[slot] as usize;
        for (j, &h) in b.joints.iter().enumerate() {
            let k = j + 1;
            if k >= b.plan_len {
                break;
            }
            let stiffness = crate::RIGID_MOTOR_STIFFNESS
                * world.pixels.flex[offset + k].max(0.05);
            if let Some(joint) = self.impulse_joints.get_mut(h) {
                joint.data.set_motor_position(
                    JointAxis::AngX,
                    target.get(k).copied().unwrap_or(0.0),
                    stiffness,
                    crate::RIGID_MOTOR_DAMPING,
                );
            }
        }
    }

    /// Resistive force theory, applied per component.
    ///
    /// Isotropic damping alone cannot produce swimming: a body would simply
    /// slow down wherever it went. Thrust exists because a segment moving
    /// SIDEWAYS through water meets far more resistance than one sliding along
    /// its own length, so a travelling wave pushes more water backwards than
    /// forwards. That anisotropy is the entire mechanism, and it has to be
    /// applied explicitly whichever backend is running.
    fn apply_drag(&mut self, world: &World, slot: usize) {
        let Some(b) = self.built.get(&slot) else { return };
        let offset = world.individuals.pixel_offset[slot] as usize;
        for k in 1..b.parts.len() {
            let parent = world.pixels.parent_idx[offset + k];
            if parent < 0 {
                continue;
            }
            let p = parent as usize;
            if p >= b.parts.len() {
                continue;
            }
            let (pos_k, vel_k) = {
                let rb = &self.bodies[b.parts[k]];
                (*rb.translation(), *rb.linvel())
            };
            let pos_p = *self.bodies[b.parts[p]].translation();
            let seg = pos_k - pos_p;
            let len = seg.norm();
            if len < 1e-6 {
                continue;
            }
            let tangent = seg / len;
            let v_par = vel_k.dot(&tangent);
            let par = tangent * v_par;
            let perp = vel_k - par;
            let perp_coeff = crate::DRAG_PERPENDICULAR
                * crate::PART_DRAG_PERP[world.pixels.part_type[offset + k] as usize];
            let force = -(par * crate::DRAG_PARALLEL + perp * perp_coeff) * len;
            self.bodies[b.parts[k]].add_force(force, true);
        }
    }
}

impl BodyPhysics for RigidBodies {
    fn name(&self) -> &'static str {
        "rigid"
    }

    fn positions(&self, world: &World, slot: usize, t: f32) -> Vec<[f32; 2]> {
        match self.built.get(&slot) {
            Some(b) if b.plan_len == world.individuals.pixel_count[slot] as usize => b
                .parts
                .iter()
                .map(|&h| {
                    let p = self.bodies[h].translation();
                    [p.x, p.y]
                })
                .collect(),
            // Not built yet, or the plan has changed and the rebuild has not
            // happened: fall back to the analytic pose rather than returning
            // something stale. Every consumer expects a body of exactly the
            // current component count.
            _ => crate::physics::analytic_positions(world, slot, t),
        }
    }

    fn velocities(
        &self,
        world: &World,
        slot: usize,
        positions: &[[f32; 2]],
        t: f32,
    ) -> Vec<[f32; 2]> {
        match self.built.get(&slot) {
            Some(b) if b.plan_len == world.individuals.pixel_count[slot] as usize => b
                .parts
                .iter()
                .map(|&h| {
                    let v = self.bodies[h].linvel();
                    [v.x, v.y]
                })
                .collect(),
            _ => crate::physics::pixel_velocities_from(world, slot, positions, t, 0.02),
        }
    }

    fn body_changed(&mut self, world: &World, slot: usize) {
        self.build(world, slot);
    }

    fn body_removed(&mut self, slot: usize) {
        self.destroy(slot);
    }

    fn owns_integration(&self) -> bool {
        true
    }
}

impl RigidBodies {
    /// Advance the solver by one tick, after driving motors and applying drag
    /// for every animal it holds.
    pub fn step_world(&mut self, world: &mut World, targets: &HashMap<usize, Vec<f32>>) {
        let slots: Vec<usize> = self.built.keys().copied().collect();
        for slot in &slots {
            if !world.individuals.alive[*slot] {
                self.destroy(*slot);
                continue;
            }
            if let Some(t) = targets.get(slot) {
                self.drive_motors(world, *slot, t);
            }
            self.apply_drag(world, *slot);
        }
        self.physics.step(
            &self.gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad,
            &mut self.narrow,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            None,
            &(),
            &(),
        );
        // Write the solver's answer back into the individual table, which is
        // what the rest of the engine reads. The root component's body is the
        // animal's position, and its heading follows the root-to-next-part
        // direction rather than being a number the brain assigns.
        for slot in slots {
            let Some(b) = self.built.get(&slot) else { continue };
            if b.parts.is_empty() || !world.individuals.alive[slot] {
                continue;
            }
            let root = &self.bodies[b.parts[0]];
            let p = root.translation();
            let v = root.linvel();
            world.individuals.root_pos[slot] = [p.x, p.y];
            world.individuals.velocity[slot] = [v.x, v.y];
            if b.parts.len() > 1 {
                let n = self.bodies[b.parts[1]].translation();
                let dx = n.x - p.x;
                let dy = n.y - p.y;
                if dx * dx + dy * dy > 1e-8 {
                    world.individuals.heading[slot] = dy.atan2(dx);
                }
            }
        }
    }
}
