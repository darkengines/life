//! The pixel arena: every individual's variable-length pixel list lives in
//! one shared set of flat arrays (parent_idx/rest_angle/flex/memory), with
//! each individual owning a contiguous (offset, length) region -- a "pixels"
//! table with an implicit foreign key to its owning individual, in 1NF
//! terms. Growth allocates a fresh region one pixel larger and copies the
//! old data over (pixel counts are small -- tens at most -- so this is
//! cheap); the old region returns to a free list. This is a simplified,
//! from-scratch first-fit allocator with adjacent-block coalescing,
//! inspired by (not copied from) a skip-list buddy allocator seen in a
//! reference C++ engine's FreeList.hpp -- that one optimizes for many
//! large, long-lived allocations; this one is tuned for this domain's
//! actual shape (many tiny, short-lived regions, churning constantly).
pub struct PixelArena {
    pub parent_idx: Vec<i32>, // -1 = root, else LOCAL index within the owning individual's region
    pub rest_angle: Vec<f32>,
    pub flex: Vec<f32>,
    pub memory: Vec<[f32; 4]>,
    // Storage affinity: an evolvable per-PIXEL trait, mechanically identical
    // to flex -- not a "belly" concept imposed by design, just a value each
    // pixel can carry that the capture/digestion engine consults (see
    // physics::resolve_collision). Nothing hardcodes what shape or count of
    // high-storage pixels is good; if a body benefits from concentrating
    // storage in one place (a real evolved belly) or spreading it out, or
    // not having any, evolution finds that on its own the same way it finds
    // rigid vs. flexible body plans via `flex`.
    pub storage: Vec<f32>,
    // Anatomy, heritable per body PART (not per individual): every part can
    // independently evolve how big it is and how far its joint can swing.
    // `size` scales this pixel's mass/collision footprint/hit-point pool and
    // its drawn radius -- a body plan can put a big armored plate here and a
    // tiny fast flipper there. `min_angle`/`max_angle` bound how far this
    // joint's animated bend can deviate from its rest pose -- a real hinge
    // limit (like an elbow that can't bend backward), not the previously
    // unbounded oscillation every joint had regardless of what it was.
    pub size: Vec<f32>,
    pub min_angle: Vec<f32>,
    pub max_angle: Vec<f32>,
    // Combat is a resource pool now, not a binary pass/fail: a hit drains
    // this instead of always deleting the pixel outright. Max health scales
    // with `size`, so a bigger part can absorb more before it's actually
    // severed -- see combat.rs.
    pub health: Vec<f32>,
    // Differentiated body parts. Until now every pixel was mechanically the
    // same lump with continuous modifiers, which is why bodies could get
    // BIGGER but never more COMPLEX -- there was nothing for a part to
    // specialise INTO, so a 20-pixel animal was just a longer worm, not an
    // animal with organs. Each type carries a real function and a real
    // metabolic cost (see PART_* in lib.rs), so division of labour becomes
    // something selection can discover: an eye-heavy scout, a mouth-and-
    // tentacle ambusher, a gut-heavy grazer, an armoured tank. Nothing
    // rewards any particular combination -- only the costs and effects
    // exist, exactly like every other trait here.
    pub part_type: Vec<u8>,
    // Bilateral symmetry, heritable per NODE. When a new component grows on a
    // node carrying this, a mirrored counterpart appears on the same node at
    // the reflected angle -- so organs arrive in left/right pairs. This is
    // one of the genuine major body-plan innovations in animal evolution
    // (bilateria), and it is what makes a shape read as an animal rather than
    // a lump: paired eyes, paired flippers, paired tentacles. It stays
    // evolvable rather than universal, because a pair costs twice the upkeep
    // and is only worth it where balanced propulsion or stereo sensing pays.
    pub symmetric: Vec<bool>,
    free_blocks: Vec<(u32, u32)>, // (offset, length), kept sorted by offset
}

/// Body-part kinds. Kept as a plain u8 in the arena (cheap to copy, cheap to
/// publish to the frontend) with these constants as the vocabulary.
pub const PART_BODY: u8 = 0;
pub const PART_EYE: u8 = 1;
pub const PART_MOUTH: u8 = 2;
pub const PART_GUT: u8 = 3;
pub const PART_TENTACLE: u8 = 4;
pub const PART_ARMOR: u8 = 5;
pub const PART_FLIPPER: u8 = 6;
pub const PART_KIND_COUNT: u8 = 7;

impl PixelArena {
    pub fn new() -> Self {
        PixelArena {
            parent_idx: Vec::new(), rest_angle: Vec::new(), flex: Vec::new(), memory: Vec::new(), storage: Vec::new(),
            size: Vec::new(), min_angle: Vec::new(), max_angle: Vec::new(), health: Vec::new(),
            part_type: Vec::new(),
            symmetric: Vec::new(),
            free_blocks: Vec::new(),
        }
    }

    pub fn allocate(&mut self, length: u32) -> u32 {
        // first-fit: smallest-offset free block that's big enough
        if let Some(i) = self.free_blocks.iter().position(|(_, len)| *len >= length) {
            let (offset, len) = self.free_blocks[i];
            if len == length {
                self.free_blocks.remove(i);
            } else {
                self.free_blocks[i] = (offset + length, len - length);
            }
            return offset;
        }
        let offset = self.parent_idx.len() as u32;
        let new_len = offset + length;
        self.parent_idx.resize(new_len as usize, -1);
        self.rest_angle.resize(new_len as usize, 0.0);
        self.flex.resize(new_len as usize, 1.0);
        self.memory.resize(new_len as usize, [0.0; 4]);
        self.storage.resize(new_len as usize, 0.0);
        self.size.resize(new_len as usize, 1.0);
        self.min_angle.resize(new_len as usize, -1.0);
        self.max_angle.resize(new_len as usize, 1.0);
        self.health.resize(new_len as usize, crate::BASE_PIXEL_HEALTH);
        self.part_type.resize(new_len as usize, PART_BODY);
        self.symmetric.resize(new_len as usize, false);
        offset
    }

    pub fn free(&mut self, offset: u32, length: u32) {
        if length == 0 {
            return;
        }
        let pos = self.free_blocks.partition_point(|(o, _)| *o < offset);
        self.free_blocks.insert(pos, (offset, length));
        // coalesce with the following block if adjacent
        if pos + 1 < self.free_blocks.len() {
            let (next_off, next_len) = self.free_blocks[pos + 1];
            let (this_off, this_len) = self.free_blocks[pos];
            if this_off + this_len == next_off {
                self.free_blocks[pos] = (this_off, this_len + next_len);
                self.free_blocks.remove(pos + 1);
            }
        }
        // coalesce with the preceding block if adjacent
        if pos > 0 {
            let (prev_off, prev_len) = self.free_blocks[pos - 1];
            let (this_off, this_len) = self.free_blocks[pos];
            if prev_off + prev_len == this_off {
                self.free_blocks[pos - 1] = (prev_off, prev_len + this_len);
                self.free_blocks.remove(pos);
            }
        }
    }
}
