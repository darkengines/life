//! Chemical/resource fields: food (clustered patches, not uniform), and the
//! diffusing signals (pheromone, blood, acid, light). Flat row-major Vec<f32>
//! per field, matching the layout the earlier benchmarking already showed is
//! plenty fast on CPU at this grid size (diffusion was never the bottleneck).
use numpy::ndarray::Array2;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::terrain::{Terrain, TerrainKind};

pub struct Fields {
    pub food: Vec<f32>,
    pub food_capacity: Vec<f32>,
    pub pheromone: Vec<f32>,
    pub blood: Vec<f32>,
    pub acid: Vec<f32>,
    pub light: Vec<f32>,
    // Quorum sensing (bacteria-inspired): unlike every other field here,
    // this isn't an evolved trait's deliberate output -- every living
    // individual emits a small, fixed, non-evolvable amount into it just
    // by existing, the same way real quorum-sensing autoinducers are
    // constitutively produced rather than switched on. What (if anything)
    // an evolved brain DOES in response to sensing local concentration --
    // scatter, get more aggressive, hunker down -- is entirely something
    // selection has to find; nothing here decides that. See physics.rs's
    // emission site and individuals.rs's SENSE_DIM for where it's read.
    pub quorum: Vec<f32>,
    // Stigmergic territory marking: individuals with a nonzero
    // `territoriality` trait passively deposit into this field wherever
    // they go (see TERRITORY_EMIT_BASE in lib.rs), and it diffuses/decays
    // slowly -- real scent marks linger far longer than a fast-decaying
    // trail pheromone. There is deliberately no per-cell "owner" recorded
    // (that would need a much heavier per-cell data structure); the real
    // home-range literature models this the same way -- any individual far
    // from its own home_pos treats high local concentration as "someone
    // else's, back off", while high concentration NEAR home reads as
    // familiar/reinforcing. The brain has to learn that conjunction itself
    // from the two separate sense inputs; nothing here decides it.
    pub territory: Vec<f32>,
}

impl Fields {
    pub fn new(size: u32, food_cap: f32, terrain: &Terrain, rng: &mut Pcg64, n_patches: u32) -> Self {
        let n = (size * size) as usize;
        let mut food_capacity = vec![0.0f32; n];
        for _ in 0..n_patches {
            let cx = rng.random_range(0.0..size as f32);
            let cy = rng.random_range(0.0..size as f32);
            let radius = rng.random_range(size as f32 * 0.03..size as f32 * 0.09);
            let intensity = rng.random_range(0.6..1.0);
            let inv_two_r2 = 1.0 / (2.0 * radius * radius);
            for x in 0..size {
                for y in 0..size {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    food_capacity[(x * size + y) as usize] += intensity * (-(dx * dx + dy * dy) * inv_two_r2).exp();
                }
            }
        }
        for v in food_capacity.iter_mut() {
            *v = v.clamp(0.05, food_cap);
        }
        for (i, k) in terrain.kind.iter().enumerate() {
            if *k == TerrainKind::Rock {
                food_capacity[i] = 0.0;
            }
        }
        let food: Vec<f32> = food_capacity.iter().map(|c| c * 0.5).collect();
        Fields {
            food,
            food_capacity,
            pheromone: vec![0.0; n],
            blood: vec![0.0; n],
            acid: vec![0.0; n],
            light: vec![0.0; n],
            quorum: vec![0.0; n],
            territory: vec![0.0; n],
        }
    }

    pub fn idx(size: u32, x: i32, y: i32) -> usize {
        let xc = x.clamp(0, size as i32 - 1) as u32;
        let yc = y.clamp(0, size as i32 - 1) as u32;
        (xc * size + yc) as usize
    }

    pub fn gradient(field: &[f32], size: u32, pos: [f32; 2]) -> (f32, f32) {
        Self::gradient_at_range(field, size, pos, 1)
    }

    /// Same idea as `gradient`, but sampled `range` cells out instead of
    /// just the immediately-adjacent one. A 1-cell gradient is only
    /// nonzero once a body is ALREADY touching the edge of a patch -- for
    /// something with real spatial extent, like a food patch tens of
    /// units across, that's not "smell", it's "bump into it and notice
    /// too late". Used for food specifically (see FOOD_SMELL_RANGE) so a
    /// body genuinely away from any patch still gets a real directional
    /// cue toward the nearest one, instead of zero signal until it
    /// happens to wander close by chance.
    pub fn gradient_at_range(field: &[f32], size: u32, pos: [f32; 2], range: i32) -> (f32, f32) {
        let x = (pos[0] as i32).clamp(0, size as i32 - 1);
        let y = (pos[1] as i32).clamp(0, size as i32 - 1);
        let x0 = (x - range).max(0);
        let x1 = (x + range).min(size as i32 - 1);
        let y0 = (y - range).max(0);
        let y1 = (y + range).min(size as i32 - 1);
        let gx = field[Self::idx(size, x1, y)] - field[Self::idx(size, x0, y)];
        let gy = field[Self::idx(size, x, y1)] - field[Self::idx(size, x, y0)];
        (gx, gy)
    }

    pub fn sample(field: &[f32], size: u32, pos: [f32; 2]) -> f32 {
        let x = (pos[0] as i32).clamp(0, size as i32 - 1);
        let y = (pos[1] as i32).clamp(0, size as i32 - 1);
        field[Self::idx(size, x, y)]
    }

    fn diffuse_and_decay(field: &[f32], size: u32, diffusion_rate: f32, decay: f32) -> Vec<f32> {
        // Reflecting boundary (edge cells treat the out-of-bounds neighbor as
        // themselves), NOT wraparound. Individuals live in a bounded world
        // with real walls -- a chemical signal must not be able to leak from
        // the bottom edge to the top edge just because diffusion happened to
        // be implemented as a torus. That mismatch was a real, visible bug:
        // chemicals "generated at the bottom" were showing up at the top.
        let n = size as usize;
        let mut out = vec![0.0f32; field.len()];
        for x in 0..n {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x == n - 1 { n - 1 } else { x + 1 };
            for y in 0..n {
                let ym = if y == 0 { 0 } else { y - 1 };
                let yp = if y == n - 1 { n - 1 } else { y + 1 };
                let center = field[x * n + y];
                let neighbor_avg = (field[xm * n + y] + field[xp * n + y] + field[x * n + ym] + field[x * n + yp]) * 0.25;
                let updated = center + diffusion_rate * (neighbor_avg - center);
                out[x * n + y] = updated * decay;
            }
        }
        out
    }

    pub fn step_diffusion(&mut self, size: u32) {
        self.pheromone = Self::diffuse_and_decay(&self.pheromone, size, crate::PHEROMONE_DIFFUSION, crate::PHEROMONE_DECAY);
        self.blood = Self::diffuse_and_decay(&self.blood, size, crate::BLOOD_DIFFUSION, crate::BLOOD_DECAY);
        self.acid = Self::diffuse_and_decay(&self.acid, size, crate::ACID_DIFFUSION, crate::ACID_DECAY);
        self.light = Self::diffuse_and_decay(&self.light, size, crate::LIGHT_DIFFUSION, crate::LIGHT_DECAY);
        self.quorum = Self::diffuse_and_decay(&self.quorum, size, crate::QUORUM_DIFFUSION, crate::QUORUM_DECAY);
        self.territory = Self::diffuse_and_decay(&self.territory, size, crate::TERRITORY_DIFFUSION, crate::TERRITORY_DECAY);
    }

    /// User-triggered feeding: a Gaussian bump of food centered on (x, y),
    /// added both to the immediate food level (so it's usable right away)
    /// and to food_capacity (so the spot keeps regrowing richer afterward
    /// instead of the bump just draining back to whatever it was before --
    /// a real, lasting change to that patch of the world, not a one-tick
    /// freebie).
    pub fn add_food_at(&mut self, size: u32, x: f32, y: f32, amount: f32, radius: f32, cap: f32) {
        let n = size as usize;
        let inv_two_r2 = 1.0 / (2.0 * radius * radius).max(0.001);
        let ir = radius.ceil() as i32 * 3;
        let cx = x.round() as i32;
        let cy = y.round() as i32;
        for dx in -ir..=ir {
            for dy in -ir..=ir {
                let px = cx + dx;
                let py = cy + dy;
                if px < 0 || py < 0 || px >= size as i32 || py >= size as i32 { continue; }
                let d2 = (dx * dx + dy * dy) as f32;
                let w = (-d2 * inv_two_r2).exp();
                if w < 0.001 { continue; }
                let idx = (px as usize) * n + (py as usize);
                self.food[idx] = (self.food[idx] + amount * w).clamp(0.0, cap);
                self.food_capacity[idx] = (self.food_capacity[idx] + amount * w * 0.3).clamp(0.0, cap);
            }
        }
    }

    pub fn step_food_regrow(&mut self, regrow_rate: f32, cap: f32) {
        for i in 0..self.food.len() {
            self.food[i] = (self.food[i] + regrow_rate * (self.food_capacity[i] - self.food[i])).clamp(0.0, cap);
        }
    }

    fn to_2d(field: &[f32], size: u32) -> Array2<f32> {
        let n = size as usize;
        Array2::from_shape_vec((n, n), field.to_vec()).unwrap()
    }
    pub fn food_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.food, size) }
    pub fn pheromone_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.pheromone, size) }
    pub fn blood_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.blood, size) }
    pub fn acid_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.acid, size) }
    pub fn light_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.light, size) }
    pub fn quorum_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.quorum, size) }
    pub fn territory_2d(&self, size: u32) -> Array2<f32> { Self::to_2d(&self.territory, size) }
}
