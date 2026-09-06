"""Are organs actually selected FOR, or just tolerated?

Tissue share alone cannot answer this. Roughly 5% of newly grown parts become
any given organ (a 30% chance of differentiating, spread over six kinds), so
5% is what pure mutation produces with no selection at all. An organ sitting
below that is being actively removed; one sitting above it is being kept. But
a share is a snapshot, and a snapshot cannot distinguish "eyes are useless"
from "eyes are useful but only just appeared".

So this measures selection two ways instead:

  * The TRAJECTORY of each organ's share. A share that falls steadily from the
    founder population is being selected against, whatever its final value.
  * A direct fitness contrast: for each organ, the mean age and mean energy of
    animals carrying at least one, against animals carrying none. Age is a
    survival proxy -- an animal that lives longer has, by definition, been
    killed less. If carrying an eye were useless the two groups would match;
    if it is a liability the carriers do worse.

The contrast is confounded by body size (bigger animals both live longer and
have more parts, so more chances to carry any organ), so it is also reported
restricted to a narrow size band, where that confound is largely removed.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.009, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEEDS = [int(a) for a in sys.argv[2:]] or [11, 22]
NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper", "filter"]
# What mutation alone produces with no selection: a 30% chance that a new part
# differentiates, spread evenly over the seven organ kinds. Adding an eighth
# part type moved this, which is exactly why it is computed rather than typed
# in -- a stale baseline would silently reclassify which organs are "winning".
BASELINE = 0.30 / 7


def shares(inds):
    tally = collections.Counter()
    for i in inds:
        for t in i["part_type"]:
            tally[t] += 1
    total = sum(tally.values()) or 1
    return {k: tally.get(k, 0) / total for k in range(8)}


def contrast(inds, kind, lo=0, hi=10**9):
    band = [i for i in inds if lo <= len(i["positions"]) <= hi]
    have = [i for i in band if kind in i["part_type"]]
    lack = [i for i in band if kind not in i["part_type"]]
    if len(have) < 15 or len(lack) < 15:
        return None
    return (
        statistics.mean(i["age"] for i in have),
        statistics.mean(i["age"] for i in lack),
        len(have),
        len(lack),
    )


for seed in SEEDS:
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.spawn_random(300)
    print(f"--- seed {seed} | organ share over time ({BASELINE*100:.1f}% = mutation alone) ---")
    print(f"{'tick':>6} {'pop':>6} {'meanPx':>7} " + " ".join(f"{NAMES[k][:5]:>6}" for k in range(1, 8)))
    for t in range(1, TICKS + 1):
        w.tick()
        if t % 2000:
            continue
        inds = w.individuals_state()
        if not inds:
            print(f"{t:>6}  EXTINCT")
            break
        sh = shares(inds)
        print(f"{t:>6} {len(inds):>6} "
              f"{statistics.mean(len(i['positions']) for i in inds):>7.2f} "
              + " ".join(f"{sh[k]*100:>5.1f}%" for k in range(1, 8)))
        sys.stdout.flush()

    inds = w.individuals_state()
    if not inds:
        continue
    print(f"\n  mean age of carriers vs non-carriers (age is a survival proxy)")
    print(f"  {'organ':>9} {'all: have':>10} {'lack':>8} {'8-16 parts: have':>18} {'lack':>8}")
    for k in range(1, 8):
        a = contrast(inds, k)
        b = contrast(inds, k, 8, 16)
        fa = f"{a[0]:>10.0f} {a[1]:>8.0f}" if a else f"{'-':>10} {'-':>8}"
        fb = f"{b[0]:>18.0f} {b[1]:>8.0f}" if b else f"{'-':>18} {'-':>8}"
        print(f"  {NAMES[k]:>9} {fa} {fb}")
    print()
