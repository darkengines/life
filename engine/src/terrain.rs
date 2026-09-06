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

pub struct Terrain {
    pub size: u32,
    pub kind: Vec<TerrainKind>, // row-major: kind[x*size+y]
}

impl Terrain {
    pub fn at(&self, x: u32, y: u32) -> TerrainKind {
        self.kind[(x * self.size + y) as usize]
    }

    pub fn generate(size: u32, rng: &mut Pcg64) -> Self {
        let mut kind = vec![TerrainKind::Empty; (size * size) as usize];
        let floor_height = ((size as f32) * SAND_FLOOR_FRACTION).max(4.0) as u32;
        for x in 0..size {
            for y in 0..floor_height.min(size) {
                kind[(x * size + y) as usize] = TerrainKind::Sand;
            }
        }
        for _ in 0..N_ROCK_CLUSTERS {
            let cx = rng.random_range(0.0..size as f32);
            let on_floor = rng.random::<f32>() < 0.5;
            let cy = if on_floor {
                rng.random_range(0.0..(floor_height as f32 * 1.3).max(1.0))
            } else {
                rng.random_range(floor_height as f32..(size as f32 * 0.7).max(floor_height as f32 + 1.0))
            };
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
                    carve(&mut kind, x, y, 1);
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
        for _ in 0..N_CREVICE_POCKETS {
            let cx = rng.random_range(2..size as i32 - 2);
            let cy = rng.random_range(2..size as i32 - 2);
            let outer = rng.random_range(3..6);
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
                carve(&mut kind, mx, my, 0);
            }
        }

        Terrain { size, kind }
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
