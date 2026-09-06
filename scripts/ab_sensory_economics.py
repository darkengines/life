"""Do eyes, mouths and flippers pay for themselves?

Measured on the live world: the passive organs thrive (gut 13.9% of all
tissue, tentacle 6.1%) while every ACTIVE organ sits below the ~5% that pure
random differentiation would produce -- flipper 3.7%, eye 2.8%, mouth 2.1%.
They are being selected against. That matters most for eyes, because with no
eyes there is no perception, and without perception no amount of brain can
help.

Two suspects, and they must not be conflated:

1. Foraging was never gated on eyes. The food gradient was sampled out to
   range 18 for everybody, so a blind animal found distant food exactly as
   well as a sighted one. An eye bought only threat/prey/mate perception --
   and paid upkeep for it.
2. Sensors were priced like armour plate (eye 1.4x, mouth 1.5x, flipper
   1.6x upkeep against body 1.0x).

Three arms separate them. Seeds matter enormously here -- an earlier sweep
had two seeds at identical settings land on 4.7 vs 13.3 mean parts -- so
every arm runs several seeds and the spread is reported, never one number.
"""
import collections
import statistics
import sys
import time

import rust_world

W, POP_CAP, REGROW, PATCHES = 240, 6000, 0.015, 50
TICKS = 18000
SEEDS = [11, 22, 33]

PART_NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper"]
OLD_METAB = [1.0, 1.4, 1.5, 1.3, 1.5, 1.8, 1.6]
NEW_METAB = [1.0, 1.15, 1.3, 1.3, 1.5, 1.8, 1.35]

# (name, blind_range, sight_per_eye, metabolism)
ARMS = [
    ("old       ", 18, 0, OLD_METAB),
    ("cheap     ", 18, 0, NEW_METAB),
    ("gated+chp ", 4, 5, NEW_METAB),
]


def organ_mix(inds):
    tally = collections.Counter()
    for i in inds:
        for t in i.get("part_type", []):
            tally[t] += 1
    total = sum(tally.values()) or 1
    return {PART_NAMES[k]: tally.get(k, 0) / total for k in range(len(PART_NAMES))}


def run(blind, per_eye, metab, seed):
    w = rust_world.World(W, REGROW, 1.0, POP_CAP, seed, PATCHES)
    w.debug_set_smell_ranges(blind, per_eye)
    w.debug_set_part_metabolism(metab)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    counts = [len(i["positions"]) for i in inds]
    return dict(
        pop=len(inds),
        mean=statistics.mean(counts),
        big=sum(1 for c in counts if c >= 12) / len(counts),
        mix=organ_mix(inds),
    )


only = sys.argv[1] if len(sys.argv) > 1 else None
print(f"{TICKS} ticks | regrow {REGROW} | seeds {SEEDS}")
print("organ shares are % of ALL tissue; ~5% is what random differentiation gives\n")
print(f"{'arm':>10} {'seed':>5} {'pop':>6} {'meanPx':>7} {'>=12px':>7} "
      f"{'eye':>6} {'mouth':>6} {'flip':>6} {'gut':>6} {'tent':>6} {'armor':>6}")

for name, blind, per_eye, metab in ARMS:
    if only and only.strip() != name.strip():
        continue
    agg = collections.defaultdict(list)
    for seed in SEEDS:
        t0 = time.time()
        r = run(blind, per_eye, metab, seed)
        if r is None:
            print(f"{name:>10} {seed:>5}   EXTINCT")
            continue
        m = r["mix"]
        print(f"{name:>10} {seed:>5} {r['pop']:>6} {r['mean']:>7.2f} {r['big']*100:>6.1f}% "
              f"{m['eye']*100:>5.1f}% {m['mouth']*100:>5.1f}% {m['flipper']*100:>5.1f}% "
              f"{m['gut']*100:>5.1f}% {m['tentacle']*100:>5.1f}% {m['armor']*100:>5.1f}%"
              f"   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
        for k in ("pop", "mean", "big"):
            agg[k].append(r[k])
        for k, v in m.items():
            agg[k].append(v)
    if agg["mean"]:
        print(f"{name:>10} {'MEAN':>5} {statistics.mean(agg['pop']):>6.0f} "
              f"{statistics.mean(agg['mean']):>7.2f} {statistics.mean(agg['big'])*100:>6.1f}% "
              f"{statistics.mean(agg['eye'])*100:>5.1f}% {statistics.mean(agg['mouth'])*100:>5.1f}% "
              f"{statistics.mean(agg['flipper'])*100:>5.1f}% {statistics.mean(agg['gut'])*100:>5.1f}% "
              f"{statistics.mean(agg['tentacle'])*100:>5.1f}% {statistics.mean(agg['armor'])*100:>5.1f}%\n")
        sys.stdout.flush()
