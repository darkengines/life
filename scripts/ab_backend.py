"""Analytic vs rigid-body physics, on the same world.

The point of the split is that either engine can be chosen for the context, so
what matters is not which is "better" but what each costs and what each buys.
Both are given identical intent -- the rigid backend's joint motors chase the
same angles the analytic model assigns outright -- so any difference in what
the animals do is a difference in the physics rather than in what they were
asked to do.

Reported: throughput, and whether propulsion still works. A backend that is
beautiful and cannot swim is not an option.
"""
import math
import statistics
import sys
import time

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
N, PARTS = 150, 10
SAMPLE = 200


def build(backend):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
    got = w.set_physics_backend(backend)
    w.debug_set_life_history(1000.0, 0.95)
    w.spawn_random(N)
    for ind in w.individuals_state():
        for _ in range(PARTS):
            w.debug_grow(ind["id"])
    for _ in range(30):
        w.tick()
    return w, got


for backend in ("analytic", "rigid"):
    w, got = build(backend)
    if got != backend:
        print(f"{backend:>9}: NOT AVAILABLE in this build (running {got})")
        continue
    inds = w.individuals_state()
    parts = sum(len(i["positions"]) for i in inds)
    before = {i["id"]: i["positions"][0] for i in inds}
    t0 = time.perf_counter()
    for _ in range(SAMPLE):
        w.tick()
    el = time.perf_counter() - t0
    inds = w.individuals_state()
    after = {i["id"]: i["positions"][0] for i in inds}
    common = set(before) & set(after)
    moved = [math.dist(before[i], after[i]) for i in common] if common else [0]
    alive = len(inds)
    print(f"{got:>9}: {SAMPLE/el:7.1f} ticks/s  "
          f"({1000*el/SAMPLE:6.2f} ms/tick, {parts} components)  "
          f"pop {alive:>4}  mean travel {statistics.mean(moved):6.2f} over {SAMPLE} ticks")
