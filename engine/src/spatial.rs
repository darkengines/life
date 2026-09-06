//! Uniform spatial grid for O(local) neighbor lookups -- same algorithmic
//! idea as pixel_world.py's dict-based `_spatial`/`_nearby`, but with no
//! per-lookup Python dict/object overhead.
use std::collections::HashMap;

const CELL_SIZE: f32 = 3.0;

pub struct SpatialGrid {
    cells: HashMap<(i32, i32), Vec<u32>>,
    n_cells: i32,
}

/// A grid over individual COMPONENTS rather than over whole animals.
///
/// Body-to-body contact was broad-phased through the body grid, which indexes
/// each animal by its root alone. That forces a query wide enough to reach the
/// largest animal in the WORLD -- one forty-part giant makes every other body,
/// however small, sweep a radius-twenty neighbourhood, hundreds of cells,
/// almost all of it empty. It then tested every component of this body against
/// every component of each candidate.
///
/// Indexing the components themselves removes both problems: a component only
/// ever queries its own contact reach, about a unit, and finds precisely the
/// components that could touch it. The cell is deliberately much smaller than
/// the body grid's, because component reach is roughly a unit rather than the
/// span of an animal.
pub struct PartGrid {
    cells: HashMap<(i32, i32), Vec<(u32, u32)>>,
    /// Cells per side. The world is a torus, so a query near one edge has to
    /// reach round into the cells on the opposite edge; without this the seam
    /// behaves as a wall that nothing can see across, which is the exact
    /// artefact wrapping exists to remove.
    n_cells: i32,
}

const PART_CELL_SIZE: f32 = 1.5;

impl PartGrid {
    pub fn build(world_size: f32, parts: impl Iterator<Item = (u32, u32, [f32; 2])>) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<(u32, u32)>> = HashMap::new();
        for (slot, idx, pos) in parts {
            cells.entry(Self::cell_of(pos)).or_default().push((slot, idx));
        }
        PartGrid { cells, n_cells: (world_size / PART_CELL_SIZE).ceil() as i32 }
    }

    fn cell_of(pos: [f32; 2]) -> (i32, i32) {
        ((pos[0] / PART_CELL_SIZE).floor() as i32, (pos[1] / PART_CELL_SIZE).floor() as i32)
    }

    /// Visits every component within `radius` of `pos`. Takes a closure rather
    /// than returning a Vec because this runs once per component per tick, in
    /// the hot parallel path, and allocating there would cost more than the
    /// search itself.
    pub fn for_each_near(&self, pos: [f32; 2], radius: f32, mut f: impl FnMut(u32, u32)) {
        let (cx, cy) = Self::cell_of(pos);
        let reach = (radius / PART_CELL_SIZE).ceil() as i32;
        let n = self.n_cells.max(1);
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let key = (wrap_cell(cx + dx, n), wrap_cell(cy + dy, n));
                if let Some(v) = self.cells.get(&key) {
                    for &(slot, idx) in v {
                        f(slot, idx);
                    }
                }
            }
        }
    }
}

/// Folds a cell coordinate back into range, so the grid is a torus like the
/// world it indexes.
#[inline]
fn wrap_cell(c: i32, n: i32) -> i32 {
    let m = c % n;
    if m < 0 { m + n } else { m }
}

impl SpatialGrid {
    pub fn build(world_size: f32, positions: impl Iterator<Item = (u32, [f32; 2])>) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (slot, pos) in positions {
            let cell = Self::cell_of(pos);
            cells.entry(cell).or_default().push(slot);
        }
        SpatialGrid { cells, n_cells: (world_size / CELL_SIZE).ceil() as i32 }
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
        let n = self.n_cells.max(1);
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let key = (wrap_cell(cx + dx, n), wrap_cell(cy + dy, n));
                if let Some(v) = self.cells.get(&key) {
                    out.extend_from_slice(v);
                }
            }
        }
        // A query wider than the world would otherwise visit the same cell
        // several times and count its occupants more than once.
        if reach * 2 + 1 >= n {
            out.sort_unstable();
            out.dedup();
        }
        out
    }
}
