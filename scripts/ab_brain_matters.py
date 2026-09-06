"""Does the brain do anything measurable at all?

Before spending more effort on bigger networks or better training, this
answers the prior question: if a world of animals deciding at RANDOM performs
the same as a world of evolved ones, then behaviour is not on the critical
path to fitness, and no amount of network capacity or training will change
that. Every attempt to improve intelligence would be measuring noise.

There is real reason to suspect it. Evolved brains were previously measured
performing no better than random ones at steering toward food, and a claimed
gain from trained perception failed to replicate and was retracted. Meanwhile
foraging works on a gradient that needs no brain, and reproduction needs only
proximity -- so it is entirely possible to live a full life here without
deciding anything.

The comparison is between identical worlds differing only in `brain_noise`:
0.0 leaves decisions alone, 1.0 replaces every one with deterministic noise,
and intermediate values blend. If the columns match, the brain is inert.

What counts as the answer: population and mean body size are ecosystem-level
outcomes, but the sharpest single number is lifespan. An animal that decides
well should live longer than one flailing at random, in the same world, under
the same hazards.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEEDS = [11, 22, 33, 44]
ARMS = [("evolved   ", 0.0), ("half-noise", 0.5), ("random    ", 1.0)]


def run(noise, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_brain_noise(noise)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    ev = w.events()
    if not inds:
        return None
    counts = [len(i["positions"]) for i in inds]
    return dict(
        pop=len(inds),
        mean=statistics.mean(counts),
        age=statistics.mean(i["age"] for i in inds),
        energy=statistics.mean(i["energy"] for i in inds),
        births=ev["reproductions"],
        eaten=ev["eaten"],
        starved=ev["starved"],
    )


print(f"{TICKS} ticks | seeds {SEEDS}")
print("if these columns match, the brain is not on the path to fitness\n")
print(f"{'arm':>11} {'seed':>5} {'pop':>6} {'meanPx':>7} {'age':>7} {'energy':>7} "
      f"{'births':>8} {'eaten':>8} {'starved':>8}")
for label, noise in ARMS:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        r = run(noise, seed)
        if r is None:
            print(f"{label:>11} {seed:>5}   EXTINCT")
            continue
        print(f"{label:>11} {seed:>5} {r['pop']:>6} {r['mean']:>7.2f} {r['age']:>7.0f} "
              f"{r['energy']:>7.1f} {r['births']:>8} {r['eaten']:>8} {r['starved']:>8}")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["pop"]:
        print(f"{label:>11} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>7.2f} {statistics.mean(agg['age']):>7.0f} "
              f"{statistics.mean(agg['energy']):>7.1f} {statistics.mean(agg['births']):>8.0f} "
              f"{statistics.mean(agg['eaten']):>8.0f} {statistics.mean(agg['starved']):>8.0f}\n")
        sys.stdout.flush()
