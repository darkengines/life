"""Does the resource refuge stop the boom-bust?

The world has been oscillating between roughly twenty animals and five
hundred. Grazing used to take a cell's plankton to zero, so a population boom
stripped the water bare and then starved in it -- a textbook consumer-resource
cycle driven by a resource with no refuge. Saturating intake leaves an
unharvestable remainder, which is the classic stabiliser.

This measures the oscillation directly rather than the endpoint. What matters
is not the average population but how far it SWINGS: the ratio of peak to
trough after founding. A world that runs steadily at 200 is a better world than
one that averages 300 by alternating between 20 and 600, and an endpoint
measurement cannot tell them apart.

  swing      peak / trough over the settled period. 1.0 is perfectly steady;
             anything above ~5 is a world lurching between famine and plague.
  cv         standard deviation over mean -- the same story, less sensitive to
             a single extreme sample.
  min        the trough itself, because a world that dips to three animals has
             very nearly ended regardless of what it recovers to.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
SEEDS = [11, 22, 33]
SAMPLE = 250
SETTLE = 4000
# (label, half-saturation). 0.0 restores the old take-everything grazing, so
# the sweep carries its own control.
ARMS = [
    ("no refuge ", 0.0),
    ("weak      ", 0.02),
    ("default   ", 0.05),
    ("strong    ", 0.12),
]


def run(half_sat, seed):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_graze_half_saturation(half_sat)
    w.spawn_random(300)
    traj, food = [], []
    for t in range(1, TICKS + 1):
        w.tick()
        if t % SAMPLE == 0:
            n = len(w.individuals_state())
            traj.append(n)
            food.append(w.events().get("mean_food", 0.0))
            if n == 0:
                break
    return traj, food


print(f"{TICKS} ticks | seeds {SEEDS} | settled period = after tick {SETTLE}\n")
print(f"{'arm':>11} {'seed':>5} {'min':>6} {'peak':>6} {'mean':>6} {'swing':>7} "
      f"{'cv':>6} {'food':>7}")
for label, hs in ARMS:
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        traj, food = run(hs, seed)
        settled = traj[SETTLE // SAMPLE:] or traj
        settled_food = food[SETTLE // SAMPLE:] or food
        if not settled or max(settled) == 0:
            print(f"{label:>11} {seed:>5}   EXTINCT")
            continue
        lo, hi = min(settled), max(settled)
        avg = statistics.mean(settled)
        swing = hi / max(1, lo)
        cv = (statistics.pstdev(settled) / avg) if avg > 0 else 0
        mf = statistics.mean(settled_food) if settled_food else 0
        print(f"{label:>11} {seed:>5} {lo:>6} {hi:>6} {avg:>6.0f} {swing:>7.1f} "
              f"{cv:>6.2f} {mf:>7.4f}")
        sys.stdout.flush()
        agg["min"].append(lo)
        agg["swing"].append(swing)
        agg["cv"].append(cv)
        agg["mean"].append(avg)
    if agg["swing"]:
        print(f"{label:>11} {'MEAN':>5} {statistics.mean(agg['min']):>6.0f} "
              f"{'':>6} {statistics.mean(agg['mean']):>6.0f} "
              f"{statistics.mean(agg['swing']):>7.1f} {statistics.mean(agg['cv']):>6.2f}\n")
        sys.stdout.flush()
