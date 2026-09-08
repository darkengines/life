"""Is any of this actually Darwinian?

The suspicion is that the world produces random creatures and calls the result
evolution. It is a fair suspicion and it has a decisive test: break HEREDITY and
nothing else. Same world, same food, same deaths, same reproduction -- but each
offspring gets random traits instead of its parents'.

Selection needs three things: variation, heredity, and differential survival.
Variation and differential survival are obviously present. If removing heredity
changes nothing, then the third leg is missing and what has been happening is
drift with extra steps -- and no amount of tuning selection pressure would help,
because there would be nothing for pressure to act ON.

Reported per arm: population, body size, and the traits selection should be
shaping if it is working at all. If the inherited arm does not pull away from
the scrambled one, the answer is no.
"""
import collections
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEEDS = [11, 22, 33]


def run(scramble, seed):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
    w.debug_scramble_inheritance(scramble)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    ev = w.events()
    if not inds:
        return None
    c = [len(i["positions"]) for i in inds]
    return dict(
        pop=len(inds),
        parts=statistics.mean(c),
        biggest=max(c),
        age=statistics.mean(i["age"] for i in inds),
        births=ev["reproductions"],
        align=ev.get("mean_motor_align", 0.0),
    )


print(f"{TICKS} ticks | seeds {SEEDS}")
print("'inherited' = normal.  'scrambled' = offspring get RANDOM traits.\n")
print(f"{'arm':>11} {'seed':>5} {'pop':>6} {'parts':>7} {'max':>5} {'age':>7} "
      f"{'births':>8} {'align':>7}")
for scramble, label in ((False, "inherited"), (True, "scrambled")):
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        r = run(scramble, seed)
        if r is None:
            print(f"{label:>11} {seed:>5}   EXTINCT")
            continue
        print(f"{label:>11} {seed:>5} {r['pop']:>6} {r['parts']:>7.1f} {r['biggest']:>5} "
              f"{r['age']:>7.0f} {r['births']:>8} {r['align']:>+7.3f}")
        sys.stdout.flush()
        for k, v in r.items():
            agg[k].append(v)
    if agg["pop"]:
        print(f"{label:>11} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['parts']):>7.1f} {statistics.mean(agg['biggest']):>5.0f} "
              f"{statistics.mean(agg['age']):>7.0f} {statistics.mean(agg['births']):>8.0f} "
              f"{statistics.mean(agg['align']):>+7.3f}\n")
        sys.stdout.flush()
