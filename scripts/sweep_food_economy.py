"""Find a food economy the world can actually live in.

Every setting tried by hand tonight produced the same shape: the founding
population crashes within a few hundred ticks, then a remnant of ten or so
persists, occasionally spiking and crashing again. Hand-tuning one constant at
a time against a moving target has not worked, and each point measurement was
taken at a different world age, which cannot distinguish "wrong food level"
from "world in slow decline".

So this sweeps the two knobs that actually set the economy, together, on
several seeds, at long horizon, and reports the POPULATION TRAJECTORY rather
than a single endpoint -- because the failure being hunted is a crash in the
first few hundred ticks, which any endpoint measurement misses entirely.

  calories   energy per unit of plankton swallowed
  snow       total production, as equivalent fully-lit rows

What is wanted: a population that survives founding, settles somewhere well
above the handful that has been limping along, and stays there. The
`min-after-2k` column is the one that matters -- a run that dips to three
animals has not survived, whatever it recovers to later.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
SEEDS = [11, 22, 33]
GRID = [
    (0.36, 6.0),
    (0.36, 14.0),
    (0.80, 6.0),
    (0.80, 14.0),
    (1.60, 14.0),
    (1.60, 30.0),
]
SAMPLE = 500


def run(cal, snow, seed):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_plankton(cal, snow)
    w.spawn_random(300)
    traj = []
    for t in range(1, TICKS + 1):
        w.tick()
        if t % SAMPLE == 0:
            traj.append(len(w.individuals_state()))
            if traj[-1] == 0:
                break
    inds = w.individuals_state()
    ev = w.events()
    if not inds:
        return None, traj
    counts = [len(i["positions"]) for i in inds]
    dd = max(1, ev["deaths"])
    return dict(
        pop=len(inds),
        mean=statistics.mean(counts),
        biggest=max(counts),
        energy=statistics.mean(i["energy"] for i in inds),
        starved=ev["starved"] / dd,
        eaten=ev["eaten"] / dd,
    ), traj


print(f"{TICKS} ticks | seeds {SEEDS} | population sampled every {SAMPLE}\n")
print(f"{'cal':>5} {'snow':>5} {'seed':>5} {'pop':>6} {'min@2k+':>8} {'mean_pop':>9} "
      f"{'parts':>6} {'energy':>7} {'starv%':>7} {'eaten%':>7}")
for cal, snow in GRID:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        r, traj = run(cal, snow, seed)
        settled = traj[int(2000 / SAMPLE):] or traj
        lo = min(settled) if settled else 0
        avg = statistics.mean(settled) if settled else 0
        if r is None:
            print(f"{cal:>5.2f} {snow:>5.1f} {seed:>5}   EXTINCT (min {lo})")
            agg["lo"].append(0)
            continue
        print(f"{cal:>5.2f} {snow:>5.1f} {seed:>5} {r['pop']:>6} {lo:>8} {avg:>9.0f} "
              f"{r['mean']:>6.1f} {r['energy']:>7.0f} {r['starved']*100:>6.0f}% {r['eaten']*100:>6.0f}%")
        sys.stdout.flush()
        agg["lo"].append(lo)
        agg["avg"].append(avg)
        for k in ("pop", "mean", "energy"):
            agg[k].append(r[k])
    if agg["lo"]:
        print(f"{cal:>5.2f} {snow:>5.1f} {'MEAN':>5} "
              f"{statistics.mean(agg['pop']) if agg['pop'] else 0:>6.0f} "
              f"{statistics.mean(agg['lo']):>8.0f} "
              f"{statistics.mean(agg['avg']) if agg['avg'] else 0:>9.0f}\n")
        sys.stdout.flush()
