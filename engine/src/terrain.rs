//! Static world structure: rocks (hard obstacles) and a sand seafloor
//! (passable but costly), ported from pixel_world.py's `_generate_terrain`.
use numpy::ndarray::Array2;
use rand::Rng;
use rand_pcg::Pcg64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TerrainKind {
    Empty,
    Rock,
    Sand,
}

pub const SAND_FLOOR_FRACTION: f32 = 0.09;
const N_ROCK_CLUSTERS: u32 = 26;
// Reef structure. The world used to be a single open arena -- a sand floor
// and a handful of solid rock blobs -- which meant there was nowhere a small
// animal could go that a large one could not follow.
//
// Sized up after measuring: refuge worked in the right direction (small
// creatures occupied positions with ~66% more surrounding rock than large
// ones) but rock covered only 3.6% of the world, so it was a curiosity
// rather than a habitat. A reef has to be a real fraction of the world for
// living in it to be a viable strategy rather than a lucky hiding spot. Size-selective refuge
// is the textbook mechanism that lets predator and prey coexist instead of
// the predator simply eating everything, and it is also what produces
// habitat specialisation for free: a body that fits in the reef lives a
// different life from one that cannot. Nothing here grants small creatures
// protection directly; the passages are just narrow, and per-pixel terrain
// collision does the rest.
const N_TUNNELS_PER_CLUSTER: u32 = 3;
const TUNNEL_STEPS: u32 = 90;
const N_CREVICE_POCKETS: u32 = 70;
// Corridor half-width. Small bodies fit; large ones cannot manoeuvre.
const TUNNEL_RADIUS: i32 = 2;
const CREVICE_MOUTH_RADIUS: i32 = 1;
// How far up the water column reef structure can reach, as a fraction of
// world height. Everything rock-like lives inside this band above the floor.
const REEF_MAX_RISE: f32 = 0.30;
/// Radius over which enclosure is measured -- roughly a body length.
const ENCLOSURE_RADIUS: usize = 4;

pub struct Terrain {
    pub size: u32,
    pub kind: Vec<TerrainKind>, // row-major: kind[x*size+y]
    /// Fraction of nearby cells that are solid, per cell.
    ///
    /// Creatures had no way to perceive terrain at all -- they discovered rock
    /// only by colliding with it -- so even though sheltering measurably pays
    /// (small bodies survive 42% inside the deep reef against 20% for large
    /// ones), nothing could navigate toward it. Precomputed once because
    /// terrain never changes, which makes sensing shelter a field lookup
    /// rather than a per-tick scan.
    pub enclosure: Vec<f32>,
}

impl Terrain {
    pub fn at(&self, x: u32, y: u32) -> TerrainKind {
        self.kind[(x * self.size + y) as usize]
    }

    pub fn generate(size: u32, rng: &mut Pcg64) -> Self {
        let mut kind = vec![TerrainKind::Empty; (size * size) as usize];
        // A sand seafloor, and nothing else. The world is a cylinder: left
        // and right are joined, top and bottom are not, so there is a real
        // surface where plankton enters the water and a real bottom where
        // whatever is not eaten finally settles. Depth is the one axis here
        // that means something.
        //
        // Rock is off. What the boulder fields and reef used to provide was a
        // size-selective refuge, and it genuinely worked -- small bodies were
        // measured surviving at 42% inside the deep reef against 20% for
        // large ones. Without it, a small animal's only refuge is behavioural,
        // so if size structure is wanted it now has to be earned rather than
        // handed out by the terrain.
        let floor_height = ((size as f32) * SAND_FLOOR_FRACTION).max(4.0) as u32;
        if crate::TERRAIN_ENABLED {
            for x in 0..size {
                for y in 0..floor_height.min(size) {
                    kind[(x * size + y) as usize] = TerrainKind::Sand;
                }
            }
        }
        if !crate::ROCK_ENABLED {
            let _ = rng;
            return Self::with_enclosure(size, kind);
        }
        // Rock belongs to the seafloor. Scattering clusters up through the
        // water column left boulders hanging in mid-water with nothing holding
        // them up, which reads as broken rather than as an environment. Real
        // reef and rock formations sit ON the bottom and rise from it, so
        // cluster centres are drawn near the floor and fall off sharply with
        // height -- the occasional tall outcrop still reaches up, but nothing
        // floats free.
        for _ in 0..N_ROCK_CLUSTERS {
            let cx = rng.random_range(0.0..size as f32);
            // Squaring a 0..1 draw biases strongly toward the floor while
            // still allowing the occasional tall formation.
            let h = rng.random::<f32>();
            let rise = h * h * size as f32 * REEF_MAX_RISE;
            let cy = floor_height as f32 + rise;
            let radius = rng.random_range(size as f32 * 0.02..size as f32 * 0.065);
            let r2 = radius * radius;
            for x in 0..size {
                for y in 0..size {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    if dx * dx + dy * dy < r2 {
                        kind[(x * size + y) as usize] = TerrainKind::Rock;
                    }
                }
            }
        }
        // Carve cave networks through the rock. A random walk cleared at
        // radius 1 leaves a passage roughly two to three cells wide -- wide
        // enough for a small body to swim through, too tight for a large one
        // to manoeuvre in, which is the whole point.
        let idx = |x: i32, y: i32| -> Option<usize> {
            if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
                None
            } else {
                Some((x as u32 * size + y as u32) as usize)
            }
        };
        // Passages have to be wide enough that a SMALL body actually fits,
        // while still excluding large ones. Carving at radius 1 gave ~2-3
        // cell corridors, which was fine while bodies could illegally embed
        // in stone but became impassable to everything once rock was made
        // genuinely solid -- at which point the reef stopped being a refuge
        // and became a wall. A component collides by its own radius plus the
        // repulsion range, so corridors need real clearance to admit anyone.
        let carve = |kind: &mut Vec<TerrainKind>, cx: i32, cy: i32, r: i32| {
            for dx in -r..=r {
                for dy in -r..=r {
                    if dx * dx + dy * dy <= r * r {
                        if let Some(i) = idx(cx + dx, cy + dy) {
                            kind[i] = TerrainKind::Empty;
                        }
                    }
                }
            }
        };

        let rock_cells: Vec<usize> = (0..kind.len()).filter(|&i| kind[i] == TerrainKind::Rock).collect();
        if !rock_cells.is_empty() {
            for _ in 0..(N_ROCK_CLUSTERS * N_TUNNELS_PER_CLUSTER) {
                let seed_cell = rock_cells[rng.random_range(0..rock_cells.len())];
                let mut x = (seed_cell as u32 / size) as i32;
                let mut y = (seed_cell as u32 % size) as i32;
                // A drifting walk rather than pure noise, so tunnels run
                // somewhere instead of dissolving the whole mass into gravel.
                let mut dir = rng.random_range(0.0..std::f32::consts::TAU);
                for _ in 0..TUNNEL_STEPS {
                    carve(&mut kind, x, y, TUNNEL_RADIUS);
                    dir += rng.random_range(-0.5..0.5);
                    x += (dir.cos() * 1.5).round() as i32;
                    y += (dir.sin() * 1.5).round() as i32;
                    if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
                        break;
                    }
                }
            }
        }

        // Free-standing crevices: a small rock ring with a hollow middle and
        // a narrow mouth. These sit out in open water and on the floor, so
        // refuge isn't confined to the big reef masses.
        // Crevices belong in the rock, not floating in open water. Placed
        // within the reef band just above the floor, so they read as pockets
        // eroded into the bottom structure.
        for _ in 0..N_CREVICE_POCKETS {
            let cx = rng.random_range(2..size as i32 - 2);
            let band = ((size as f32 * REEF_MAX_RISE) as i32).max(6);
            let cy = (floor_height as i32 + rng.random_range(0..band)).min(size as i32 - 3);
            let outer = rng.random_range(5..9);
            for dx in -outer..=outer {
                for dy in -outer..=outer {
                    let d2 = dx * dx + dy * dy;
                    if d2 <= outer * outer {
                        if let Some(i) = idx(cx + dx, cy + dy) {
                            kind[i] = TerrainKind::Rock;
                        }
                    }
                }
            }
            // hollow it out, then open one narrow mouth
            carve(&mut kind, cx, cy, outer - 2);
            let mouth = rng.random_range(0.0..std::f32::consts::TAU);
            for step in 0..(outer + 2) {
                let mx = cx + (mouth.cos() * step as f32).round() as i32;
                let my = cy + (mouth.sin() * step as f32).round() as i32;
                carve(&mut kind, mx, my, CREVICE_MOUTH_RADIUS);
            }
        }

        Self::with_enclosure(size, kind)
    }

    /// Builds the enclosure field from a solid mask and returns the terrain.
    fn with_enclosure(size: u32, kind: Vec<TerrainKind>) -> Self {
        // Box-blur the solid mask into an enclosure field. Prefix sums keep
        // this linear, and it runs once per world.
        let n = size as usize;
        let r = ENCLOSURE_RADIUS;
        let mut pref = vec![0u32; (n + 1) * (n + 1)];
        for x in 0..n {
            let mut row = 0u32;
            for y in 0..n {
                if kind[x * n + y] != TerrainKind::Empty {
                    row += 1;
                }
                pref[(x + 1) * (n + 1) + (y + 1)] = pref[x * (n + 1) + (y + 1)] + row;
            }
        }
        let mut enclosure = vec![0.0f32; n * n];
        for x in 0..n {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(n);
            for y in 0..n {
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(n);
                let total = pref[x1 * (n + 1) + y1] + pref[x0 * (n + 1) + y0]
                    - pref[x0 * (n + 1) + y1] - pref[x1 * (n + 1) + y0];
                enclosure[x * n + y] = total as f32 / ((x1 - x0) * (y1 - y0)) as f32;
            }
        }
        Terrain { size, kind, enclosure }
    }

    pub fn as_2d(&self) -> Array2<i32> {
        let n = self.size as usize;
        let mut out = Array2::<i32>::zeros((n, n));
        for x in 0..n {
            for y in 0..n {
                out[[x, y]] = match self.kind[x * n + y] {
                    TerrainKind::Empty => 0,
                    TerrainKind::Rock => 1,
                    TerrainKind::Sand => 2,
                };
            }
        }
        out
    }
}
