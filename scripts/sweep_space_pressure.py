"""How hard should crowding have to bite before it regulates the population?

The world ran to 1778 animals and 30799 components in a 240x240 space with
starvation at only 5.8% of deaths -- so food abundance was doing no regulating
whatever, and neither was the existing crowding energy tax. Animals stacked on
each other because nothing stopped them.

The tax was wrong twice over. It was charged on KIN density around the root,
so a body buried in unrelated strangers paid nothing at all, and counting
roots ignores that a forty-part animal occupies vastly more room than a
three-part one. Density-dependent mortality replaces it, measured against
other animals' components pressing on this one's, per component, blind to
kinship -- a sibling takes up exactly as much room as a stranger, so a lineage
cannot escape it by filling the world with copies of itself, which is exactly
what it had been doing.

This sweeps how hard it bites. Two failure modes bracket the answer: too weak
and nothing changes, too strong and the world empties. What is wanted between
them is fewer animals with room around them, and crucially NOT a collapse in
body size -- if pressure kills the large animals preferentially it has simply
undone the complexity that was so much work to get.

The `crowded%` column is the share of deaths this mechanism is responsible
for. If it is near zero the mechanism is inert; if it is nearly everything,
it has become the only thing happening in the world, which is its own failure.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SEEDS = [11, 22, 33]
# (label, tolerance, mortality-per-tick coefficient)
ARMS = [
    ("off       ", 1.5, 0.0),
    ("gentle    ", 1.5, 0.0004),
    ("default   ", 1.5, 0.0009),
    ("harsh     ", 1.2, 0.0025),
]


def run(tol, mort, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_space_pressure(tol, mort)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    counts = sorted(len(i["positions"]) for i in inds)
    n = len(counts)
    ev = w.events()
    deaths = max(1, ev["deaths"])
    return dict(
        pop=n,
        mean=statistics.mean(counts),
        p95=counts[int(n * 0.95)],
        biggest=counts[-1],
        big=sum(1 for c in counts if c >= 12) / n,
        parts=sum(counts),
        crowded=ev.get("crowded", 0) / deaths,
        starved=ev["starved"] / deaths,
        eaten=ev["eaten"] / deaths,
    )


print(f"{TICKS} ticks | regrow {REGROW} | seeds {SEEDS}")
print("'parts' is total components in a 240x240 world -- the honest crowding number\n")
print(f"{'arm':>11} {'seed':>5} {'pop':>6} {'mean':>6} {'p95':>5} {'max':>5} {'>=12px':>7} "
      f"{'parts':>7} {'crowd%':>7} {'starv%':>7} {'eaten%':>7}")
for label, tol, mort in ARMS:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        r = run(tol, mort, seed)
        if r is None or r["pop"] < 40:
            print(f"{label:>11} {seed:>5}   COLLAPSED" + (f" (pop {r['pop']})" if r else ""))
            continue
        print(f"{label:>11} {seed:>5} {r['pop']:>6} {r['mean']:>6.2f} {r['p95']:>5} {r['biggest']:>5} "
              f"{r['big']*100:>6.1f}% {r['parts']:>7} {r['crowded']*100:>6.1f}% "
              f"{r['starved']*100:>6.1f}% {r['eaten']*100:>6.1f}%")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["pop"]:
        print(f"{label:>11} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>6.2f} {statistics.mean(agg['p95']):>5.1f} "
              f"{statistics.mean(agg['biggest']):>5.1f} {statistics.mean(agg['big'])*100:>6.1f}% "
              f"{statistics.mean(agg['parts']):>7.0f} {statistics.mean(agg['crowded'])*100:>6.1f}% "
              f"{statistics.mean(agg['starved'])*100:>6.1f}% {statistics.mean(agg['eaten'])*100:>6.1f}%\n")
        sys.stdout.flush()
