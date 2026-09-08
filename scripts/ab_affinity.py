"""Does social affinity actually produce structure, or is it food patches?

Animals aggregate around food whether or not they attract one another, so a
clustering figure measured on its own attributes nothing. This turns the
mechanism off and leaves everything else identical.

Clustering is scored against a null model of the same animals at random
positions, so it is not confounded by population density either -- a denser
world has closer neighbours regardless of why.
"""
import math
import random
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SEEDS = [11, 22, 33]


def nnd(pts):
    out, cell = [], {}
    for i, (x, y) in enumerate(pts):
        cell.setdefault((int(x // 12), int(y // 12)), []).append(i)
    for i, (x, y) in enumerate(pts):
        best = 1e9
        cx, cy = int(x // 12), int(y // 12)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((cx + dx, cy + dy), ()):
                    if j == i:
                        continue
                    ddx = abs(pts[j][0] - x)
                    ddx = min(ddx, W - ddx)
                    d = math.hypot(ddx, pts[j][1] - y)
                    if d < best:
                        best = d
        if best < 1e8:
            out.append(best)
    return out


print(f"{TICKS} ticks | seeds {SEEDS}")
print("clustering < 1 means animals are closer together than random placement\n")
print(f"{'arm':>10} {'seed':>5} {'pop':>6} {'parts':>7} {'clustering':>11}")
for scale, label in ((0.0, "off"), (1.0, "on")):
    rows = []
    for seed in SEEDS:
        w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
        w.debug_set_affinity(scale)
        w.spawn_random(300)
        for _ in range(TICKS):
            w.tick()
        inds = w.individuals_state()
        if len(inds) < 20:
            print(f"{label:>10} {seed:>5}   too few ({len(inds)})")
            continue
        pts = [i["positions"][0] for i in inds]
        obs = statistics.mean(nnd(pts))
        rnd = []
        rng = random.Random(seed)
        for _ in range(3):
            rnd += nnd([(rng.uniform(0, W), rng.uniform(0, W)) for _ in pts])
        ratio = obs / statistics.mean(rnd)
        parts = statistics.mean(len(i["positions"]) for i in inds)
        print(f"{label:>10} {seed:>5} {len(inds):>6} {parts:>7.1f} {ratio:>11.3f}")
        sys.stdout.flush()
        rows.append(ratio)
    if rows:
        print(f"{label:>10} {'MEAN':>5} {'':>6} {'':>7} {statistics.mean(rows):>11.3f}\n")
