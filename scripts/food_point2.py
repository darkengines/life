"""The refuge threshold was set above the water's normal concentration.

Saturating intake was added to leave depleted patches a refuge, which is the
right mechanism. But the half-saturation constant was picked without checking
what concentration the water actually carries: at 0.05 against typical
concentrations well below that, it was not protecting depleted water, it was
cutting intake by most of its value EVERYWHERE. The same class of error as the
crowding tolerance of 1.5 against a measured pressure of 18 -- a threshold
chosen without looking at the distribution it thresholds.

This sweeps the half-saturation together with food supply.
"""
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
SEEDS = [11, 22, 33]
SAMPLE = 500
GRID = [(1.2, 14.0, 0.008), (1.2, 20.0, 0.008), (0.8, 20.0, 0.008), (0.8, 14.0, 0.008)]

print(f"{TICKS} ticks | seeds {SEEDS}\n")
print(f"{'cal':>5} {'rows':>5} {'hsat':>7} {'seed':>5} {'end':>6} {'min':>6} "
      f"{'parts':>6} {'energy':>7} {'food':>7} {'starv%':>7} {'eaten%':>7}")
for cal, rows, hsat in GRID:
    ends, mins, ens = [], [], []
    for seed in SEEDS:
        w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
        w.debug_set_plankton(cal, rows)
        w.debug_set_graze_half_saturation(hsat)
        w.spawn_random(300)
        traj = []
        for t in range(1, TICKS + 1):
            w.tick()
            if t % SAMPLE == 0:
                traj.append(len(w.individuals_state()))
                if traj[-1] == 0:
                    break
        inds = w.individuals_state(); ev = w.events(); dd = max(1, ev["deaths"])
        settled = traj[4:] or traj
        if not inds:
            print(f"{cal:>5.2f} {rows:>5.0f} {hsat:>7.3f} {seed:>5}   EXTINCT")
            ends.append(0); mins.append(0); continue
        en = statistics.mean(i["energy"] for i in inds)
        pa = statistics.mean(len(i["positions"]) for i in inds)
        print(f"{cal:>5.2f} {rows:>5.0f} {hsat:>7.3f} {seed:>5} {len(inds):>6} "
              f"{min(settled):>6} {pa:>6.1f} {en:>7.0f} {ev.get('mean_food',0):>7.4f} "
              f"{100*ev['starved']/dd:>6.0f}% {100*ev['eaten']/dd:>6.0f}%")
        sys.stdout.flush()
        ends.append(len(inds)); mins.append(min(settled)); ens.append(en)
    if ends:
        print(f"{cal:>5.2f} {rows:>5.0f} {hsat:>7.3f} {'MEAN':>5} "
              f"{statistics.mean(ends):>6.0f} {statistics.mean(mins):>6.0f} "
              f"{'':>6} {statistics.mean(ens) if ens else 0:>7.0f}\n")
        sys.stdout.flush()
