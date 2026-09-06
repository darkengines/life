"""How do you get FEWER, LARGER animals rather than more, smaller ones?

Two different knobs are involved and they are easy to confuse.

The metabolic EXPONENT decides how upkeep scales with body size, and therefore
whether being large is viable at all. That one is settled: sublinear scaling is
what made large complex bodies possible in the first place.

The metabolic MULTIPLIER decides how much the whole economy costs, and
therefore how many animals the food supply can support. Because upkeep is
sublinear, raising the multiplier is not size-neutral -- a one-part animal pays
close to the full per-part rate while a twenty-part animal pays roughly a third
of it, so a harsher economy should thin the world out while pressing hardest on
the smallest animals. That is exactly the requested shape: "life should become
rarer", "evolution should continue with less individuals", fewer and larger.

Worth being explicit about why this matters now. Fixing contact resolution
removed the clumping (bodies went from overlapping 46% more than chance to
sitting at chance) but roughly DOUBLED the population, because most deaths here
are predation and anything that keeps bodies apart also keeps predators off
their prey. So the world reads as crowded even though the physics is now
correct: at twenty thousand components in a 240x240 world, overlap is
unavoidable no matter how good the solver is. The remaining lever is density
itself.

Watched for: population, body size, and the fraction of runs that collapse --
a harsher economy that empties the world is not an improvement.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEEDS = [11, 22, 33]
MULTS = [float(a) for a in sys.argv[2:]] or [1.0, 1.6, 2.4, 3.5]


def run(mult, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_metabolism_multiplier(mult)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    counts = sorted(len(i["positions"]) for i in inds)
    n = len(counts)
    ev = w.events()
    return dict(
        pop=n,
        mean=statistics.mean(counts),
        p95=counts[int(n * 0.95)],
        biggest=counts[-1],
        big=sum(1 for c in counts if c >= 12) / n,
        worms=sum(1 for c in counts if c <= 3) / n,
        # how much of the world's area the animals actually occupy: the
        # honest measure of "does this look crowded"
        parts=sum(counts),
        starved=ev["starved"] / max(1, ev["deaths"]),
    )


print(f"{TICKS} ticks | regrow {REGROW} | seeds {SEEDS}\n")
print(f"{'mult':>5} {'seed':>5} {'pop':>6} {'mean':>6} {'p95':>5} {'max':>5} "
      f"{'>=12px':>7} {'worms':>7} {'parts':>7} {'starv%':>7}")
for mult in MULTS:
    agg = collections.defaultdict(list)
    dead = 0
    for seed in SEEDS:
        r = run(mult, seed)
        if r is None or r["pop"] < 50:
            print(f"{mult:>5.1f} {seed:>5}   COLLAPSED" + (f" (pop {r['pop']})" if r else ""))
            dead += 1
            continue
        print(f"{mult:>5.1f} {seed:>5} {r['pop']:>6} {r['mean']:>6.2f} {r['p95']:>5} "
              f"{r['biggest']:>5} {r['big']*100:>6.1f}% {r['worms']*100:>6.1f}% "
              f"{r['parts']:>7} {r['starved']*100:>6.1f}%")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["pop"]:
        print(f"{mult:>5.1f} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>6.2f} {statistics.mean(agg['p95']):>5.1f} "
              f"{statistics.mean(agg['biggest']):>5.1f} {statistics.mean(agg['big'])*100:>6.1f}% "
              f"{statistics.mean(agg['worms'])*100:>6.1f}% {statistics.mean(agg['parts']):>7.0f} "
              f"{statistics.mean(agg['starved'])*100:>6.1f}%   collapsed {dead}/{len(SEEDS)}\n")
        sys.stdout.flush()
