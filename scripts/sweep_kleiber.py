"""Does a realistic metabolic-scaling discount let large bodies exist?

Upkeep here was strictly linear in body size: a sixteen-part animal paid
sixteen times the running cost of a one-part blob, with nothing whatsoever
offsetting it. Complexity was therefore a pure tax, and selection stripped it
out as fast as growth added it -- which is a completely sufficient explanation
for "they are all worms" without invoking anything about brains or behaviour.

Real metabolic rate scales as roughly mass^(3/4) (Kleiber 1932; West, Brown &
Enquist 1997), so large animals get a substantial per-gram energy DISCOUNT,
and that discount is much of why being big is viable at all.

Exponent 1.0 reproduces the old linear cost exactly, so this sweep contains
its own control. The exponent is normalised at one part, so it never makes
small bodies more expensive -- only large ones cheaper.

The risk being watched for is the opposite failure: too generous a discount
and everything becomes a giant, which is just as unecological as everything
being a worm. So body-size SPREAD is reported alongside the mean -- what is
wanted is a world with both, not a world with one.
"""
import collections
import statistics
import sys
import time

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = 18000
SEEDS = [11, 22, 33, 44, 55, 66]
EXPONENTS = [float(a) for a in sys.argv[1:]] or [1.0, 0.85, 0.75, 0.65]


def run(exp, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_metabolic_exponent(exp)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    counts = sorted(len(i["positions"]) for i in inds)
    n = len(counts)
    tally = collections.Counter()
    for i in inds:
        for t in i.get("part_type", []):
            tally[t] += 1
    total = sum(tally.values()) or 1
    return dict(
        pop=n,
        mean=statistics.mean(counts),
        p50=counts[n // 2],
        p95=counts[int(n * 0.95)],
        biggest=counts[-1],
        big=sum(1 for c in counts if c >= 12) / n,
        worms=sum(1 for c in counts if c <= 3) / n,
        organ=1.0 - tally.get(0, 0) / total,
    )


print(f"{TICKS} ticks | regrow {REGROW} | seeds {SEEDS}")
print("exp 1.00 IS the old linear cost, i.e. the control\n")
print(f"{'exp':>5} {'seed':>5} {'pop':>6} {'mean':>6} {'p50':>5} {'p95':>5} {'max':>5} "
      f"{'>=12px':>7} {'worms':>7} {'organ%':>7}")
for exp in EXPONENTS:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        t0 = time.time()
        r = run(exp, seed)
        if r is None:
            print(f"{exp:>5.2f} {seed:>5}   EXTINCT")
            agg["dead"].append(1)
            continue
        agg["dead"].append(1 if r["pop"] < 50 else 0)
        print(f"{exp:>5.2f} {seed:>5} {r['pop']:>6} {r['mean']:>6.2f} {r['p50']:>5} "
              f"{r['p95']:>5} {r['biggest']:>5} {r['big']*100:>6.1f}% {r['worms']*100:>6.1f}% "
              f"{r['organ']*100:>6.1f}%   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["mean"]:
        print(f"{exp:>5.2f} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>6.2f} {statistics.mean(agg['p50']):>5.1f} "
              f"{statistics.mean(agg['p95']):>5.1f} {statistics.mean(agg['biggest']):>5.1f} "
              f"{statistics.mean(agg['big'])*100:>6.1f}% {statistics.mean(agg['worms'])*100:>6.1f}% "
              f"{statistics.mean(agg['organ'])*100:>6.1f}%\n")
        sys.stdout.flush()
