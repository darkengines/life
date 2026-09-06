"""How far should a creature smell WITHOUT eyes?

Gating long-range food detection behind eyes is the right idea -- if a blind
animal forages as well as a sighted one, an eye is pure upkeep and selection
deletes it, which is exactly what was measured (eyes at 2.8% of tissue against
a ~5% random baseline). But the first attempt set blind range to 4 against a
full range of 18 and the world went nearly extinct: the founding population
has almost no eyes, so gating hard starves everything long before eyes can
evolve. A bootstrapping cliff, not a gradient.

So the question is quantitative: how much worse must blind foraging be to make
eyes worth carrying, while still leaving a blind founder able to feed itself?
This sweeps that one number. Sighted range is blind + 3 per eye, capped at the
full 18, so two eyes always restores full-range foraging.

Seeds vary enormously in this engine (4.7 vs 13.3 mean parts at identical
settings), so every point runs several seeds.
"""
import collections
import statistics
import sys
import time

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = 18000
SEEDS = [11, 22, 33]
METAB = [1.0, 1.15, 1.3, 1.3, 1.5, 1.8, 1.35]
PART_NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper", "filter"]

BLINDS = [int(a) for a in sys.argv[1:]] or [8, 12, 15, 18]


def run(blind, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_smell_ranges(blind, 3)
    w.debug_set_part_metabolism(METAB)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    counts = [len(i["positions"]) for i in inds]
    tally = collections.Counter()
    for i in inds:
        for t in i.get("part_type", []):
            tally[t] += 1
    total = sum(tally.values()) or 1
    return dict(
        pop=len(inds),
        mean=statistics.mean(counts),
        big=sum(1 for c in counts if c >= 12) / len(counts),
        eye=tally.get(1, 0) / total,
        mouth=tally.get(2, 0) / total,
        flip=tally.get(6, 0) / total,
        # what share of animals carry at least one eye? the population-level
        # question, which the tissue share alone does not answer
        seeing=sum(1 for i in inds if 1 in i.get("part_type", [])) / len(inds),
    )


print(f"{TICKS} ticks | sighted range = blind + 3/eye, capped at 18 | seeds {SEEDS}\n")
print(f"{'blind':>6} {'seed':>5} {'pop':>6} {'meanPx':>7} {'>=12px':>7} "
      f"{'eye%':>6} {'mouth%':>7} {'flip%':>6} {'has-eye':>8}")
for blind in BLINDS:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        t0 = time.time()
        r = run(blind, seed)
        if r is None:
            print(f"{blind:>6} {seed:>5}   EXTINCT")
            continue
        print(f"{blind:>6} {seed:>5} {r['pop']:>6} {r['mean']:>7.2f} {r['big']*100:>6.1f}% "
              f"{r['eye']*100:>5.1f}% {r['mouth']*100:>6.1f}% {r['flip']*100:>5.1f}% "
              f"{r['seeing']*100:>7.1f}%   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["mean"]:
        print(f"{blind:>6} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>7.2f} {statistics.mean(agg['big'])*100:>6.1f}% "
              f"{statistics.mean(agg['eye'])*100:>5.1f}% {statistics.mean(agg['mouth'])*100:>6.1f}% "
              f"{statistics.mean(agg['flip'])*100:>5.1f}% {statistics.mean(agg['seeing'])*100:>7.1f}%\n")
        sys.stdout.flush()
