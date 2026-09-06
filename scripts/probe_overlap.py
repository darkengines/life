"""Do bodies actually stop overlapping, and what does checking properly cost?

Creatures were observed piling on top of one another for a long time, and the
crowding energy cost added to punish it never fixed it. The reason turned out
to be mechanical rather than behavioural: collision broad-phased through the
spatial grid's default one-cell search, about three world units around the
ROOT, which was fine when animals were three-part blobs. Bodies now reach
forty parts and span twenty units or more, so two animals lying completely
across one another were never even tested for contact unless their roots
nearly touched. No penalty can discourage an overlap the engine cannot see.

This measures the overlap directly -- what fraction of components are inside
another animal's component -- and times the tick, because testing every
component pair over a much wider neighbourhood is not free and performance is
a standing requirement.
"""
import statistics
import sys
import time

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [11, 22]
STIFFNESS = [float(a) for a in sys.argv[2:]] or [1.2, 5.0, 15.0, 40.0]

COLLISION_RADIUS = 1.2  # mirrors the engine constant


def overlap_stats(inds):
    """Fraction of components sitting inside a component of another animal."""
    cells = {}
    pts = []
    for ind in inds:
        girth = ind.get("part_girth") or ind.get("part_size") or [1.0] * len(ind["positions"])
        sc = ind.get("size_scale", 1.0)
        for k, p in enumerate(ind["positions"]):
            pts.append((ind["id"], p[0], p[1], girth[k] * sc))
    for i, (iid, x, y, g) in enumerate(pts):
        cells.setdefault((int(x // 2), int(y // 2)), []).append(i)
    inside = 0
    for i, (iid, x, y, g) in enumerate(pts):
        cx, cy = int(x // 2), int(y // 2)
        hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    if j == i:
                        continue
                    jid, jx, jy, jg = pts[j]
                    if jid == iid:
                        continue
                    reach = COLLISION_RADIUS * 0.5 * (g + jg)
                    if (jx - x) ** 2 + (jy - y) ** 2 < reach ** 2:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            inside += 1
    return inside / max(1, len(pts)), len(pts)


print(f"{TICKS} ticks | regrow {REGROW}\n")
print(f"{'stiff':>6} {'seed':>5} {'pop':>6} {'meanPx':>7} {'overlap':>8} {'parts':>7} {'ticks/s':>8}")
for stiff in STIFFNESS:
    for seed in SEEDS:
        w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
        w.debug_set_collision_stiffness(stiff)
        w.spawn_random(300)
        t0 = time.time()
        for _ in range(TICKS):
            w.tick()
        rate = TICKS / (time.time() - t0)
        inds = w.individuals_state()
        if not inds:
            print(f"{stiff:>6.1f} {seed:>5}   EXTINCT")
            continue
        frac, nparts = overlap_stats(inds)
        counts = [len(i["positions"]) for i in inds]
        print(f"{stiff:>6.1f} {seed:>5} {len(inds):>6} {statistics.mean(counts):>7.2f} "
              f"{frac*100:>7.1f}% {nparts:>7} {rate:>8.1f}")
        sys.stdout.flush()
