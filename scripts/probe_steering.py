"""Can these animals steer, separately from whether they decide to?

Population alignment between intent and motion sits near 0.09, and that single
number is consistent with two opposite situations: a steering loop that cannot
track a target, or a steering loop that tracks perfectly while the brain asks
for a different direction every tick. They need opposite fixes, so guessing
between them is worse than useless.

Forcing every animal's intent to one fixed direction removes the brain from the
loop. Whatever alignment remains is what the BODY and its steering can actually
achieve; the gap between that and the live figure is what the brain is losing.
"""
import math
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
N, PARTS = 120, 10
SETTLE, MEASURE = 1500, 500


def run(dx, dy, label, noise=None, n=None):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
    w.debug_set_life_history(1000.0, 0.95)   # no births: measure the same animals throughout
    if noise is not None:
        w.debug_set_thermal_noise(noise)
    w.spawn_random(n if n is not None else N)
    for ind in w.individuals_state():
        for _ in range(PARTS):
            w.debug_grow(ind["id"])
    if dx or dy:
        w.debug_force_intent(dx, dy)
    for _ in range(SETTLE):
        w.tick()
    before = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    for _ in range(MEASURE):
        w.tick()
    after = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    common = set(before) & set(after)
    if not common:
        print(f"{label:<28} nothing survived")
        return
    aligns, dists = [], []
    for i in common:
        mx = after[i][0] - before[i][0]
        my = after[i][1] - before[i][1]
        d = math.hypot(mx, my)
        if d < 0.5:
            continue          # went nowhere; direction is meaningless
        aligns.append((mx * dx + my * dy) / (d * math.hypot(dx, dy)))
        dists.append(d)
    if not aligns:
        print(f"{label:<28} n={len(common)} but nothing travelled far enough to score")
        return
    good = sum(1 for a in aligns if a > 0.7) / len(aligns)
    # The DISTRIBUTION, not just the mean. A mean of zero is produced equally by
    # a population that cannot steer and by one where half steer perfectly and
    # half steer exactly backwards -- and those need opposite fixes.
    bad = sum(1 for a in aligns if a < -0.7) / len(aligns)
    mid = 1.0 - good - bad
    print(f"{label:<24} n={len(aligns):<4} mean {statistics.mean(aligns):+.3f}  "
          f"travelled {statistics.mean(dists):5.1f}  |  on-target {good*100:3.0f}%  "
          f"backwards {bad*100:3.0f}%  scattered {mid*100:3.0f}%")


print("forced intent -- does crowding destroy steering?")
for n, tag in ((120, "120 animals (crowded)"), (12, "12 animals (sparse)")):
    print("--- " + tag + " ---")
    run(1.0, 0.0, "commanded +x (east)", 0.0, n)
    run(0.0, 1.0, "commanded +y (up)", 0.0, n)
    print("")
