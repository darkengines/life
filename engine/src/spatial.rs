//! Uniform spatial grid for O(local) neighbor lookups -- same algorithmic
//! idea as pixel_world.py's dict-based `_spatial`/`_nearby`, but with no
//! per-lookup Python dict/object overhead.
use std::collections::HashMap;

const CELL_SIZE: f32 = 3.0;

pub struct SpatialGrid {
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl SpatialGrid {
    pub fn build(positions: impl Iterator<Item = (u32, [f32; 2])>) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (slot, pos) in positions {
            let cell = Self::cell_of(pos);
            cells.entry(cell).or_default().push(slot);
        }
        SpatialGrid { cells }
    }

    fn cell_of(pos: [f32; 2]) -> (i32, i32) {
        ((pos[0] / CELL_SIZE).floor() as i32, (pos[1] / CELL_SIZE).floor() as i32)
    }

    pub fn nearby(&self, pos: [f32; 2]) -> Vec<u32> {
        self.nearby_radius(pos, CELL_SIZE)
    }

    /// Same idea as `nearby`, but for an arbitrary radius instead of the
    /// fixed one-cell-in-every-direction search `nearby` does. That fixed
    /// search was fine for collision/crowd checks (both genuinely
    /// short-range), but silently capped anything that used it at an
    /// effective range of about one CELL_SIZE regardless of what radius
    /// the CALLER actually wanted -- found via direct testing when
    /// VISION_RANGE (12.0) turned out to only ever detect something within
    /// about 3-4 units, never the full 12, because `nearby` never searched
    /// far enough to find it.
    pub fn nearby_radius(&self, pos: [f32; 2], radius: f32) -> Vec<u32> {
        let (cx, cy) = Self::cell_of(pos);
        let reach = (radius / CELL_SIZE).ceil() as i32;
        let mut out = Vec::new();
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                if let Some(v) = self.cells.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }
}
