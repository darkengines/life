"""What kills a world outright?

Runs at identical settings do not merely differ in how complex the animals get
-- some end at a population of two. That is a third attractor alongside "worm
world" and "big-bodied world", and a world that sometimes empties itself is a
worse problem than a world of worms.

Two candidate stories, and the death causes separate them cleanly:

  * Predator overshoot. A large-bodied lineage establishes, eats the prey base
    faster than it regenerates, and then starves in the emptiness it made.
    The signature is a predation spike FOLLOWED by a starvation spike, with
    the population crashing after the peak of predation.
  * Starvation alone. Nobody ever gets established; the food economy simply
    cannot support the standing crop. The signature is starvation dominating
    from the start, with no predation peak at all.

The distinction matters because the fixes are opposites. Overshoot is cured by
saturating the predator's intake rate (a Holling type II functional response:
a predator that must spend time handling each meal cannot clear a world no
matter how abundant the prey), which would preserve the requested "big animals
eat several small ones in a row" while removing the ability to eat everything.
A food-economy failure is cured by the opposite -- more food, not less.
"""
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = 18000
EVERY = 500
SEEDS = [int(a) for a in sys.argv[1:]] or [33]

for seed in SEEDS:
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.spawn_random(300)
    print(f"--- seed {seed} ---")
    print(f"{'tick':>6} {'pop':>6} {'meanPx':>7} {'maxPx':>6} {'food':>7} {'energy':>7} "
          f"{'+eaten':>7} {'+starv':>7} {'+births':>8}")
    prev = {"eaten": 0, "starved": 0, "reproductions": 0}
    for t in range(1, TICKS + 1):
        w.tick()
        if t % EVERY:
            continue
        inds = w.individuals_state()
        ev = w.events()
        d_eat = ev["eaten"] - prev["eaten"]
        d_st = ev["starved"] - prev["starved"]
        d_rep = ev["reproductions"] - prev["reproductions"]
        prev = {k: ev[k] for k in prev}
        if not inds:
            print(f"{t:>6}  EXTINCT   (last window: eaten {d_eat}, starved {d_st})")
            break
        counts = [len(i["positions"]) for i in inds]
        print(f"{t:>6} {len(inds):>6} {statistics.mean(counts):>7.2f} {max(counts):>6} "
              f"{w.debug_mean_food():>7.3f} "
              f"{statistics.mean(i['energy'] for i in inds):>7.1f} "
              f"{d_eat:>7} {d_st:>7} {d_rep:>8}")
        sys.stdout.flush()
    print()
