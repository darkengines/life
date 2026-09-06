"""Why do identical settings produce either a worm world or a big-bodied one?

A scarcity sweep landed on 4.7 mean parts / 1.9% large bodies on one seed and
13.3 mean parts / 63.1% large bodies on another, with every parameter held
fixed. That is a far bigger effect than anything being deliberately tuned, and
it means the complex basin already exists -- reaching it reliably is the whole
problem.

The suspected mechanism is a priority effect (Fukami 2015): whichever body
plan gets established first holds the world. Worms breed fast and pin the food
field near zero, and once that happens nobody can accumulate the surplus a
large body needs to get going. If instead a large lineage establishes early it
eats the worms and keeps the space.

That story makes a sharp, falsifiable prediction: the seeds must SEPARATE
EARLY, and the separation must be visible in the food field before it is
visible in body size. If the divergence instead appears late, or food looks
the same in both, the priority-effect story is wrong.
"""
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = 18000
EVERY = 1000
SEEDS = [int(a) for a in sys.argv[1:]] or [11, 22, 33, 44]


def row(w):
    inds = w.individuals_state()
    if not inds:
        return None
    counts = [len(i["positions"]) for i in inds]
    return (
        len(inds),
        statistics.mean(counts),
        sum(1 for c in counts if c >= 12) / len(counts),
        w.debug_mean_food() if hasattr(w, "debug_mean_food") else float("nan"),
        statistics.mean(i["energy"] for i in inds),
    )


print(f"{TICKS} ticks | regrow {REGROW} | tracking when seeds diverge\n")
for seed in SEEDS:
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.spawn_random(300)
    print(f"--- seed {seed} ---")
    print(f"{'tick':>6} {'pop':>6} {'meanPx':>7} {'>=12px':>7} {'food':>7} {'energy':>7}")
    for t in range(1, TICKS + 1):
        w.tick()
        if t % EVERY == 0:
            r = row(w)
            if r is None:
                print(f"{t:>6}  EXTINCT")
                break
            pop, mean, big, food, en = r
            print(f"{t:>6} {pop:>6} {mean:>7.2f} {big*100:>6.1f}% {food:>7.3f} {en:>7.1f}")
            sys.stdout.flush()
    print()
