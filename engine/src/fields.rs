//! Chemical/resource fields: food (clustered patches, not uniform), and the
//! diffusing signals (pheromone, blood, acid, light). Flat row-major Vec<f32>
//! per field. That note about diffusion never being the bottleneck was true
//! once and is not any more: measured, the field pass was 41% of a tick, and
//! unlike everything else it costs the same whether the world holds three
//! animals or three thousand, because it is pure grid work.
use numpy::ndarray::Array2;
use rayon::prelude::*;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::terrain::{Terrain, TerrainKind};

pub struct Fields {
    pub food: Vec<f32>,
    pub food_capacity: Vec<f32>,
    /// Reused diffusion workspace, so the field pass allocates nothing.
    scratch: Vec<f32>,
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
            scratch: Vec::new(),
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

    /// In-place diffusion into a caller-supplied scratch buffer, then swapped.
    ///
    /// The previous version allocated a fresh 230 KB Vec per field per tick --
    /// six of them, sixty times a second -- and touched every one of 57600
    /// cells serially. Diffusion was the single largest phase in the tick at
    /// 41% of measured time, and it costs the same whether the world holds
    /// three animals or three thousand, because it is pure grid work.
    ///
    /// Two things fixed: the scratch buffer is reused rather than reallocated,
    /// and the row loop is parallel. Rows are independent -- each reads the
    /// old field and writes only its own row of the new one -- so this is a
    /// clean split with no sharing.
    fn diffuse_into(field: &[f32], out: &mut [f32], size: u32, diffusion_rate: f32, decay: f32) {
        let n = size as usize;
        out.par_chunks_mut(n).enumerate().for_each(|(x, row)| {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x == n - 1 { n - 1 } else { x + 1 };
            let base = x * n;
            let base_m = xm * n;
            let base_p = xp * n;
            for y in 0..n {
                let ym = if y == 0 { 0 } else { y - 1 };
                let yp = if y == n - 1 { n - 1 } else { y + 1 };
                let center = field[base + y];
                let neighbor_avg = (field[base_m + y]
                    + field[base_p + y]
                    + field[base + ym]
                    + field[base + yp])
                    * 0.25;
                row[y] = (center + diffusion_rate * (neighbor_avg - center)) * decay;
            }
        });
    }

    #[allow(dead_code)]
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

    /// Diffuse the signal fields, two per tick on a rotation.
    ///
    /// Parallelising and de-allocating the pass only bought 12%, because the
    /// work is not compute-bound: six fields of 57600 floats is under three
    /// megabytes of traffic and six separate parallel regions whose per-region
    /// overhead is comparable to the work inside them.
    ///
    /// The real saving is not doing it every tick. Diffusion is a smoothing
    /// operator, and running it every third tick at three times the rate is
    /// very nearly the same operator -- the fields are slow, continuous
    /// quantities and nothing samples them faster than they change. Decay is
    /// compounded over the interval rather than tripled, since decay is
    /// multiplicative. Two fields per tick keeps the cost even instead of
    /// spiking every third tick.
    pub fn step_diffusion(&mut self, size: u32, tick: u64) {
        let n = (size as usize) * (size as usize);
        if self.scratch.len() != n {
            self.scratch = vec![0.0; n];
        }
        let stride = crate::FIELD_DIFFUSION_STRIDE;
        let slot = (tick % stride as u64) as usize;
        macro_rules! diffuse {
            ($f:ident, $d:expr, $k:expr) => {{
                // Rate scaled for the longer interval, clamped below the
                // stability limit of an explicit diffusion step.
                let rate = ($d * stride as f32).min(0.9);
                let decay = ($k as f32).powi(stride as i32);
                Self::diffuse_into(&self.$f, &mut self.scratch, size, rate, decay);
                std::mem::swap(&mut self.$f, &mut self.scratch);
            }};
        }
        match slot {
            0 => {
                diffuse!(pheromone, crate::PHEROMONE_DIFFUSION, crate::PHEROMONE_DECAY);
                diffuse!(blood, crate::BLOOD_DIFFUSION, crate::BLOOD_DECAY);
            }
            1 => {
                diffuse!(acid, crate::ACID_DIFFUSION, crate::ACID_DECAY);
                diffuse!(light, crate::LIGHT_DIFFUSION, crate::LIGHT_DECAY);
            }
            _ => {
                diffuse!(quorum, crate::QUORUM_DIFFUSION, crate::QUORUM_DECAY);
                diffuse!(territory, crate::TERRITORY_DIFFUSION, crate::TERRITORY_DECAY);
            }
        }
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

    /// Food blooms drift: new patches open somewhere else while old ones fade.
    ///
    /// Measured over 9000 ticks, mean posture curvature never moved (0.49,
    /// flat) and only ~20% of creatures ever swam straight -- because food
    /// regrew where it stood, so staying inside a patch beat leaving it and
    /// WHERE a creature went had no fitness consequence at all. Nothing
    /// selected for holding a course, so brains never learned to, and vision
    /// and navigation had nothing to earn. A shifting resource mosaic is what
    /// makes location matter: a patch you are sitting in will fade, and the
    /// next one is somewhere you have to travel to.
    pub fn step_food_blooms(&mut self, size: u32, rng: &mut Pcg64, cap: f32) {
        // Old ground slowly goes barren, so no patch is permanent.
        for c in self.food_capacity.iter_mut() {
            *c *= crate::FOOD_CAPACITY_DECAY;
        }
        if rng.random::<f32>() < crate::FOOD_BLOOM_CHANCE {
            let cx = rng.random_range(0.0..size as f32);
            let cy = rng.random_range(0.0..size as f32);
            let radius = rng.random_range(size as f32 * 0.03..size as f32 * 0.08);
            let inv_two_r2 = 1.0 / (2.0 * radius * radius);
            let intensity = rng.random_range(0.6..1.0);
            let n = size as usize;
            let ir = (radius * 2.5) as i32;
            let (ix, iy) = (cx as i32, cy as i32);
            for dx in -ir..=ir {
                for dy in -ir..=ir {
                    let px = ix + dx;
                    let py = iy + dy;
                    if px < 0 || py < 0 || px >= size as i32 || py >= size as i32 {
                        continue;
                    }
                    let d2 = (dx * dx + dy * dy) as f32;
                    let w = intensity * (-d2 * inv_two_r2).exp();
                    if w < 0.01 {
                        continue;
                    }
                    let idx = px as usize * n + py as usize;
                    self.food_capacity[idx] = (self.food_capacity[idx] + w).min(cap);
                }
            }
        }
    }

    pub fn step_food_regrow(&mut self, regrow_rate: f32, cap: f32) {
        for i in 0..self.food.len() {
            self.food[i] = (self.food[i] + regrow_rate * (self.food_capacity[i] - self.food[i])).clamp(0.0, cap);
        }
    }

    /// Marine snow: plankton enters at the surface, drifts down, and is gone
    /// if nothing eats it.
    ///
    /// Food used to regrow in place, everywhere, toward a fixed capacity map.
    /// That made a patch a permanent address: an animal could sit on one and
    /// be fed forever, which is why so many of them simply stopped moving, and
    /// it meant location had almost no consequence -- the good places were the
    /// same good places for the whole run.
    ///
    /// Real open water does not work like that. Production happens at the lit
    /// surface, and what is not eaten sinks continuously into the dark as
    /// marine snow. That single fact makes depth a gradient with a rich end
    /// and a poor end, makes food a moving target that has to be swum to, and
    /// makes a patch something that passes rather than somewhere to sit.
    ///
    /// Blooms are intermittent on purpose, including intervals of nothing at
    /// all. A constant drizzle would just be the old world at a lower rate; it
    /// is the famine between blooms that makes reserves worth carrying and
    /// makes finding a bloom worth doing.
    pub fn step_marine_snow(
        &mut self,
        size: u32,
        rng: &mut Pcg64,
        cap: f32,
        sink_rate: f32,
        bloom_intensity: f32,
        n_plumes: u32,
        production_rows: f32,
        phase: f32,
    ) {
        let n = size as usize;

        // Sink the whole field by `sink_rate` cells, mixing between the two
        // rows it falls between so slow drift is smooth rather than stepped.
        let whole = sink_rate.floor() as usize;
        let frac = sink_rate - whole as f32;
        if sink_rate > 0.0 {
            for x in 0..n {
                let col = x * n;
                // Bottom-up, so each cell reads rows that have not moved yet.
                for y in 0..n {
                    let src = y + whole;
                    let a = if src < n { self.food[col + src] } else { 0.0 };
                    let b = if src + 1 < n { self.food[col + src + 1] } else { 0.0 };
                    self.food[col + y] = a * (1.0 - frac) + b * frac;
                }
            }
        }

        // Whatever reaches the seafloor lingers briefly and then is gone --
        // this is a flux, not a reservoir, and if it accumulated the bottom
        // would simply become the old permanent food patch again.
        for x in 0..n {
            let col = x * n;
            for y in 0..crate::SNOW_FLOOR_DEPTH.min(n) {
                self.food[col + y] *= 1.0 - crate::SNOW_FLOOR_DECAY;
            }
        }

        if bloom_intensity <= 0.0 {
            return;
        }

        // New plankton enters in patches at the surface, never as an even
        // sheet: a uniform ceiling of food would give no reason to prefer one
        // stretch of water over another.
        // Production happens through a PHOTIC ZONE, not on a line.
        //
        // Plankton used to be created in a six-row band at the very top of a
        // 240-row world, which made one row of water the best place in the
        // entire ocean: sit on the ceiling and intercept everything before it
        // sinks past anyone below. Measured, 92% of the population was above
        // y=160 and 68% in the top band, jammed against the surface. That was
        // not stupidity or broken swimming -- it was the correct answer to a
        // badly shaped world, and no amount of intelligence would have chosen
        // otherwise.
        //
        // A real photic zone is tens of metres deep with production falling
        // off gradually through it. A gradient gives a reason to be at many
        // depths; a cliff gives one reason to be at exactly one.
        let zone = crate::SNOW_SOURCE_DEPTH.min(n).max(1);
        let top = n.saturating_sub(zone);
        // Total production is a property of the OCEAN; the photic zone decides
        // how that production is DISTRIBUTED through the column, not how much
        // of it there is. Normalising by the zone's total light keeps those two
        // things separate.
        //
        // Getting this wrong is exactly what happened: widening the zone from
        // six rows to ninety, to stop everything crowding the surface, silently
        // multiplied total plankton by about nine, because every row in the
        // zone received a full dose. The world then looked MORE crowded after a
        // change meant to decongest it -- not a paradox, just an ocean that had
        // quietly become nine times richer.
        // A DEEP CHLOROPHYLL MAXIMUM: production peaks BELOW the surface.
        //
        // Production used to rise monotonically toward the surface, so the top
        // row was strictly the richest water in the world and animals squatted
        // on it -- the same failure as the original six-row band, just with a
        // gentler slope. A monotonic gradient always has its best point at one
        // end, and that end is where everything goes.
        //
        // Real oceans do not work that way. Light comes from above but
        // nutrients come from below, and the two together put the chlorophyll
        // maximum at a depth, not at the surface -- it is one of the most
        // reliable features of open water. A peak in the middle means the best
        // place to be is somewhere IN the column, that being there means
        // giving up the surface, and that the animals above and below you are
        // making a different living.
        let peak = top as f32 + zone as f32 * (1.0 - crate::SNOW_PEAK_DEPTH_FRAC);
        let spread_rows = (zone as f32 * crate::SNOW_PEAK_WIDTH).max(1.0);
        let lit_of = |y: usize| {
            let d = (y as f32 - peak) / spread_rows;
            (-(d * d)).exp().max(0.02)
        };
        let lit_total: f32 = (top..n).map(lit_of).sum::<f32>().max(1e-6);
        let spread = production_rows / lit_total;
        // Horizontal structure: where the water is productive, and where it is
        // not.
        //
        // Plumes were dropped at uniformly random x, which over any span of
        // ticks averages out to an evenly productive ocean -- the randomness
        // was per-plume rather than in the water itself, so there was nothing
        // for an animal to find, remember, or return to. Real oceans are
        // nothing like uniform horizontally: fronts, eddies and upwelling
        // zones make some stretches richly productive and others close to
        // desert, and those features persist for a long time and drift.
        //
        // Three sinusoids at unrelated wavelengths and drift speeds, wrapped
        // to the cylinder so the seam is invisible. Because the periods do not
        // divide one another the pattern never repeats, and because they drift
        // at different rates the rich stretches move and slowly reorganise --
        // a productive patch is worth finding and worth following, but not
        // worth settling on forever.
        let band = |x: f32| -> f32 {
            let u = x / n as f32;
            let a = (std::f32::consts::TAU * (u * 1.0 + phase * 0.00013)).sin();
            let b = (std::f32::consts::TAU * (u * 2.0 - phase * 0.00021)).sin();
            let c = (std::f32::consts::TAU * (u * 5.0 + phase * 0.00047)).sin();
            let raw = 0.5 + 0.5 * (a * 0.5 + b * 0.32 + c * 0.18);
            // Contrast, so there are genuinely barren stretches rather than a
            // gentle ripple in an otherwise even ocean.
            raw.clamp(0.0, 1.0).powf(crate::SNOW_BAND_CONTRAST)
        };

        for _ in 0..n_plumes {
            let cx = rng.random_range(0..n) as f32;
            let width = rng.random_range(size as f32 * 0.02..size as f32 * 0.10);
            let inv_two_r2 = 1.0 / (2.0 * width * width);
            let strength = bloom_intensity * rng.random_range(0.5..1.5);
            let span = (width * 2.5).ceil() as i32;
            for dx in -span..=span {
                // The world is a cylinder, so a plume near one edge spills
                // round onto the other rather than being cut off.
                let x = (((cx as i32 + dx) % n as i32) + n as i32) as usize % n;
                let w = strength * (-(dx * dx) as f32 * inv_two_r2).exp() * band(x as f32);
                if w < 1e-4 { continue; }
                for y in top..n {
                    // Light falls off with depth, so production does too --
                    // richest near the surface, tapering away with depth,
                    // rather than a uniform slab. Normalised so the column's
                    // TOTAL output does not depend on how deep the zone is.
                    self.food[x * n + y] =
                        (self.food[x * n + y] + w * lit_of(y) * spread).min(cap);
                }
            }
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
