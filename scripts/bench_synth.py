"""Engine cost per unit of world, which is the only figure that compares.

Two benchmark attempts failed for instructive reasons. Growing a world to a
component count gives a DIFFERENT world for each engine version, because the
engine decides how it evolves -- 9.9 components per animal one run, 12.9 the
next, which swamped the effect being measured. Freezing demographics fixed the
composition but let the population drain, so successive runs got faster and the
numbers rose monotonically regardless of the engine.

The measurement that survives both problems is cost per unit of world:
milliseconds of tick per thousand components, sampled alongside the component
count so the two always correspond. It is comparable across versions, across
world sizes, and across runs of different length.
"""
import statistics
import time

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
N_ANIMALS, PARTS_EACH = 400, 12
BLOCKS, BLOCK = 6, 150

w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
w.debug_set_life_history(1000.0, 0.95)   # no births during measurement
w.spawn_random(N_ANIMALS)
for ind in w.individuals_state():
    for _ in range(PARTS_EACH):
        w.debug_grow(ind["id"])
for _ in range(40):
    w.tick()

norm, raw, sizes = [], [], []
for _ in range(BLOCKS):
    parts = sum(len(i["positions"]) for i in w.individuals_state())
    if parts == 0:
        break
    t0 = time.perf_counter()
    for _ in range(BLOCK):
        w.tick()
    ms = 1000 * (time.perf_counter() - t0) / BLOCK
    norm.append(ms / (parts / 1000.0))
    raw.append(ms)
    sizes.append(parts)

print(f"component counts across blocks: {sizes}")
print(f"raw ms/tick:        {' '.join(f'{m:.2f}' for m in raw)}")
print(f"ms per 1k components: {' '.join(f'{m:.3f}' for m in norm)}")
print(f"\nMEDIAN {statistics.median(norm):.3f} ms per 1000 components per tick")
