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
import random
import statistics
import sys
import time

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [11, 22, 33]
# (label, collision_stiffness, contact_correction). Correction 0.0 is exactly
# the old force-only solver, so the sweep carries its own control.
ARMS = [
    ("force-only ", 1.2, 0.0),
    ("stiff x12  ", 15.0, 0.0),
    ("positional ", 1.2, 0.35),
    ("both       ", 6.0, 0.35),
]

COLLISION_RADIUS = 1.2  # mirrors the engine constant


def overlap_stats(inds, jitter=None, rng=None):
    """Fraction of components sitting inside a component of another animal.

    A raw fraction is confounded by density -- a run that ends with twice the
    population will show more overlap whatever the solver does, and that
    confound is not small: raising collision stiffness twelvefold drove
    population from 856 to 1741 and the overlap figure from 56% to 95%, which
    says nothing whatever about whether contact is being resolved. So this is
    also run against a null model: the same bodies, rigidly displaced to
    uniformly random positions. The ratio of observed overlap to null overlap
    is density-controlled, and it is the number that actually answers the
    question. Below 1.0 means bodies are avoiding each other more than chance;
    at 1.0 the solver is doing nothing.
    """
    cells = {}
    pts = []
    for ind in inds:
        girth = ind.get("part_girth") or ind.get("part_size") or [1.0] * len(ind["positions"])
        sc = ind.get("size_scale", 1.0)
        # The null model moves each BODY as a rigid unit, preserving its shape
        # and its own internal spacing, so only the arrangement of animals
        # relative to one another is randomised -- which is the only thing the
        # contact solver controls.
        ox = oy = 0.0
        if jitter is not None:
            root = ind["positions"][0]
            ox = rng.uniform(0, jitter) - root[0]
            oy = rng.uniform(0, jitter) - root[1]
        for k, p in enumerate(ind["positions"]):
            pts.append((ind["id"], p[0] + ox, p[1] + oy, girth[k] * sc))
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
print("observed overlap, the same bodies at random positions, and the ratio.")
print("the RATIO is the density-controlled answer: <1 means bodies avoid each")
print("other more than chance, 1.0 means the contact solver is doing nothing.\n")
print(f"{'arm':>11} {'seed':>5} {'pop':>6} {'meanPx':>7} {'overlap':>8} {'null':>8} "
      f"{'ratio':>7} {'parts':>7} {'ticks/s':>8}")
for label, stiff, corr in ARMS:
    for seed in SEEDS:
        w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
        w.debug_set_collision_stiffness(stiff)
        w.debug_set_contact_correction(corr)
        w.spawn_random(300)
        t0 = time.time()
        for _ in range(TICKS):
            w.tick()
        rate = TICKS / (time.time() - t0)
        inds = w.individuals_state()
        if not inds:
            print(f"{label:>11} {seed:>5}   EXTINCT")
            continue
        frac, nparts = overlap_stats(inds)
        rng = random.Random(seed)
        null = statistics.mean(overlap_stats(inds, jitter=W, rng=rng)[0] for _ in range(3))
        ratio = frac / null if null > 1e-9 else float("nan")
        counts = [len(i["positions"]) for i in inds]
        print(f"{label:>11} {seed:>5} {len(inds):>6} {statistics.mean(counts):>7.2f} "
              f"{frac*100:>7.1f}% {null*100:>7.1f}% {ratio:>7.2f} {nparts:>7} {rate:>8.1f}")
        sys.stdout.flush()
