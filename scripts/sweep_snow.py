"""How much plankton should the ocean actually produce?

Food no longer regrows in place. It enters at the lit surface in intermittent
blooms and sinks as marine snow, so it has to be swum to and a patch is
something that passes rather than somewhere to sit. That is the point of the
change -- but it also means the old food rate no longer transfers, and the
first setting starved the world down to two animals in 5700 ticks.

Calibration target, from what the old field actually delivered: roughly 0.001
per cell per tick over 57600 cells, about 58 units of food per tick. The ocean
should sit BELOW that -- scarcity, hard competition and real selection are the
whole point -- but an empty ocean selects for nothing at all.

What is being looked for: a population that persists across seeds, with
animals that are hungry rather than idle, and famine visible in the death
causes without famine being the only thing that happens. `starv%` near zero
means food is still free; near a hundred means nothing else in the world
matters.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.004, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
SEEDS = [11, 22, 33]
# (label, plume strength, plumes per tick)
ARMS = [
    ("lean      ", 0.20, 3),
    ("calibrated", 0.34, 3),
    ("generous  ", 0.55, 3),
    ("rich      ", 0.85, 4),
]


def run(strength, plumes, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_snow(strength, plumes)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    ev = w.events()
    if not inds:
        return None
    counts = sorted(len(i["positions"]) for i in inds)
    n = len(counts)
    deaths = max(1, ev["deaths"])
    ys = [i["positions"][0][1] for i in inds]
    return dict(
        pop=n,
        mean=statistics.mean(counts),
        biggest=counts[-1],
        big=sum(1 for c in counts if c >= 12) / n,
        energy=statistics.mean(i["energy"] for i in inds),
        # depth is now a real gradient, so where animals sit in the water
        # column is a genuine result rather than a curiosity
        depth=statistics.mean(ys),
        starved=ev["starved"] / deaths,
        eaten=ev["eaten"] / deaths,
        crowded=ev.get("crowded", 0) / deaths,
    )


print(f"{TICKS} ticks | seeds {SEEDS} | marine snow, cylinder world\n")
print(f"{'arm':>11} {'seed':>5} {'pop':>6} {'mean':>6} {'max':>5} {'>=12px':>7} "
      f"{'energy':>7} {'depth':>6} {'starv%':>7} {'eaten%':>7} {'crowd%':>7}")
for label, strength, plumes in ARMS:
    agg = collections.defaultdict(list)
    dead = 0
    for seed in SEEDS:
        r = run(strength, plumes, seed)
        if r is None or r["pop"] < 40:
            print(f"{label:>11} {seed:>5}   COLLAPSED" + (f" (pop {r['pop']})" if r else ""))
            dead += 1
            continue
        print(f"{label:>11} {seed:>5} {r['pop']:>6} {r['mean']:>6.2f} {r['biggest']:>5} "
              f"{r['big']*100:>6.1f}% {r['energy']:>7.1f} {r['depth']:>6.1f} "
              f"{r['starved']*100:>6.1f}% {r['eaten']*100:>6.1f}% {r['crowded']*100:>6.1f}%")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["pop"]:
        print(f"{label:>11} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>6.2f} {statistics.mean(agg['biggest']):>5.1f} "
              f"{statistics.mean(agg['big'])*100:>6.1f}% {statistics.mean(agg['energy']):>7.1f} "
              f"{statistics.mean(agg['depth']):>6.1f} {statistics.mean(agg['starved'])*100:>6.1f}% "
              f"{statistics.mean(agg['eaten'])*100:>6.1f}% {statistics.mean(agg['crowded'])*100:>6.1f}%"
              f"   collapsed {dead}/{len(SEEDS)}\n")
        sys.stdout.flush()
