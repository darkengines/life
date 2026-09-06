"""Do bodies have a plan, or are they shrubs?

On screen the animals were unmistakably radial clumps rather than anything
with a front and a back, and measurement agreed: the longest chain through a
body was only 0.4 of its part count and 37% of nodes carried more than one
child. A body like that has no axis, so there is no direction for it to swim
in and nothing useful for a turn to do -- which is exactly what "an
ineffective mess" looks like.

The cause is arithmetic. Every part offers a free neighbouring cell, so
interior growth sites grow with the body while the number of tips stays at
two. At equal weight, interior sites outvote axial extension by roughly ten to
one in a twenty-part animal, and the result is a shrub no matter what else is
tuned.

Reported here:
  elongation   longest root-to-tip chain / total parts. 1.0 is a pure worm,
               ~0.2 a round bush. A finned eel sits somewhere in between --
               high, but not 1.0, because appendages are not part of the trunk.
  branch nodes fraction of nodes carrying more than one child.
  pairs        fraction of bodies showing a mirrored pair -- appendages.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEEDS = [int(a) for a in sys.argv[2:]] or [11, 22, 33]


def longest_chain(ind):
    par = ind["parents"]
    dep = [0] * len(par)
    for k, p in enumerate(par):
        dep[k] = 0 if p < 0 else dep[p] + 1
    return max(dep) + 1


def has_pair(ind):
    byp = collections.defaultdict(list)
    for k, p in enumerate(ind["parents"]):
        if p >= 0:
            byp[p].append(k)
    ra = ind["rest_angle"]
    for ks in byp.values():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                if abs(ra[ks[a]] + ra[ks[b]]) < 0.15 and abs(ra[ks[a]]) > 0.15:
                    return True
    return False


print(f"{TICKS} ticks | bodies of >=8 parts only, since shape is meaningless below that\n")
print(f"{'seed':>5} {'pop':>6} {'meanPx':>7} {'n>=8':>6} {'elong':>7} {'branch':>7} {'pairs':>7}")
for seed in SEEDS:
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        print(f"{seed:>5}   EXTINCT")
        continue
    big = [i for i in inds if len(i["positions"]) >= 8]
    if not big:
        print(f"{seed:>5} {len(inds):>6} "
              f"{statistics.mean(len(i['positions']) for i in inds):>7.2f} {0:>6}")
        continue
    elong = statistics.mean(longest_chain(i) / len(i["positions"]) for i in big)
    branch = []
    for i in big:
        c = collections.Counter(p for p in i["parents"] if p >= 0)
        branch.append(sum(1 for v in c.values() if v > 1) / max(1, len(c)))
    pairs = sum(1 for i in big if has_pair(i)) / len(big)
    print(f"{seed:>5} {len(inds):>6} "
          f"{statistics.mean(len(i['positions']) for i in inds):>7.2f} {len(big):>6} "
          f"{elong:>7.3f} {statistics.mean(branch)*100:>6.1f}% {pairs*100:>6.1f}%")
    sys.stdout.flush()
