"""Least food that still supports a living world.

The ablation was unambiguous: of every mechanism added tonight, only richer
plankton restored viability, so the world is food-starved rather than broken.
But the standing requirement is that plankton be THIN -- animals fat on
plankton is the complaint that started this -- so the point wanted is the
leanest setting the world survives, not the most comfortable one.

Reported together, because they trade off: whether the population persists, and
how fat the survivors are. A setting that keeps 400 animals alive at 1500
energy each has not solved anything.
"""
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
SEEDS = [11, 22, 33]
SAMPLE = 500
GRID = [(0.8, 10.0), (0.8, 14.0), (0.8, 20.0), (1.2, 14.0), (1.2, 20.0)]

print(f"{TICKS} ticks | seeds {SEEDS}\n")
print(f"{'cal':>5} {'rows':>5} {'seed':>5} {'end':>6} {'min':>6} {'mean':>6} "
      f"{'parts':>6} {'energy':>7} {'starv%':>7} {'eaten%':>7}")
for cal, rows in GRID:
    ends, mins, ens = [], [], []
    for seed in SEEDS:
        w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
        w.debug_set_plankton(cal, rows)
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
        dd = max(1, ev["deaths"])
        settled = traj[4:] or traj
        if not inds:
            print(f"{cal:>5.2f} {rows:>5.0f} {seed:>5}   EXTINCT")
            ends.append(0); mins.append(0)
            continue
        en = statistics.mean(i["energy"] for i in inds)
        pa = statistics.mean(len(i["positions"]) for i in inds)
        print(f"{cal:>5.2f} {rows:>5.0f} {seed:>5} {len(inds):>6} {min(settled):>6} "
              f"{statistics.mean(settled):>6.0f} {pa:>6.1f} {en:>7.0f} "
              f"{100*ev['starved']/dd:>6.0f}% {100*ev['eaten']/dd:>6.0f}%")
        sys.stdout.flush()
        ends.append(len(inds)); mins.append(min(settled)); ens.append(en)
    if ends:
        print(f"{cal:>5.2f} {rows:>5.0f} {'MEAN':>5} {statistics.mean(ends):>6.0f} "
              f"{statistics.mean(mins):>6.0f} {'':>6} {'':>6} "
              f"{statistics.mean(ens) if ens else 0:>7.0f}\n")
        sys.stdout.flush()
