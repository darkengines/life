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
const N_ROCK_CLUSTERS: u32 = 7;

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
            let radius = rng.random_range(size as f32 * 0.015..size as f32 * 0.04);
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
