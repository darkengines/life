"""Throughput at a FIXED world size, with nothing else running.

Comparing optimisation runs has been confounded twice: worlds of different
sizes, and background processes competing for the same cores. Both make a
faster engine look slower. This pins the population by seeding a fresh world
and growing it to a set component count, then measures.
"""
import statistics
import sys
import time

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TARGET_PARTS = 4000
SAMPLE = 600

w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
w.spawn_random(300)
# Grow to a comparable size rather than a comparable tick count.
for t in range(40000):
    w.tick()
    if t % 200 == 0:
        inds = w.individuals_state()
        parts = sum(len(i["positions"]) for i in inds)
        if parts >= TARGET_PARTS or not inds:
            break
inds = w.individuals_state()
parts = sum(len(i["positions"]) for i in inds)
print(f"world: pop {len(inds)}, {parts} components, {len(w.corpses_state())} corpses")

t0 = time.perf_counter()
for _ in range(SAMPLE):
    w.tick()
el = time.perf_counter() - t0
print(f"{SAMPLE/el:.1f} ticks/s   {1000*el/SAMPLE:.3f} ms/tick")
acc = {}
for _ in range(200):
    w.tick()
    for k, v in w.timings().items():
        acc.setdefault(k, []).append(v)
for k, v in sorted(acc.items(), key=lambda t: -statistics.mean(t[1]))[:6]:
    print(f"   {k:<22} {statistics.mean(v):7.3f} ms")
