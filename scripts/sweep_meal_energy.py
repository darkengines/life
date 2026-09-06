"""Can predation ever pay for a large body?

Measured economics: a 20-part predator burns 41 energy per 100 ticks in
upkeep, while engulfing a 3-part prey yields 3.1. It would need a kill every
eight ticks merely to break even, so no predator can sustain a large body and
the live world collapses to 3-part worms with no organs -- exactly what was
observed, and the reason nothing complex emerges.

A meal should be worth something like what the prey's body cost to build and
run. This sweeps that price and asks whether large, organ-bearing creatures
appear, without the population running away.
"""
import collections
import rust_world

W, POP_CAP, TICKS, SEEDS = 240, 6000, 6000, (11, 22)
NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper"]


def run(meal, seed):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
    w.debug_set_meal_energy(meal)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    inds = w.individuals_state()
    if not inds:
        return None
    sizes = [len(i["positions"]) for i in inds]
    tally = collections.Counter()
    total = 0
    for i in inds:
        for t in i.get("part_type", []):
            tally[t] += 1
            total += 1
    chains = 0
    for i in inds:
        kids = collections.Counter(p for p in i["parents"] if p >= 0)
        if not any(c > 1 for c in kids.values()):
            chains += 1
    ev = w.events()
    return dict(pop=len(inds), mean=sum(sizes) / len(sizes), mx=max(sizes),
                big=100 * sum(1 for n in sizes if n >= 12) / len(sizes),
                chains=100 * chains / len(inds),
                organs=100 * (total - tally.get(0, 0)) / max(1, total),
                eaten=100 * ev["eaten"] / max(1, ev["deaths"]))


print(f"{'meal':>6} {'seed':>5} {'pop':>6} {'meanPx':>7} {'maxPx':>6} {'>=12px':>7} "
      f"{'worms':>7} {'organ%':>7} {'eaten%':>7}")
for meal in (1.4, 4.0, 8.0, 16.0, 30.0):
    for seed in SEEDS:
        r = run(meal, seed)
        if r is None:
            print(f"{meal:>6.1f} {seed:>5}   EXTINCT", flush=True)
            continue
        print(f"{meal:>6.1f} {seed:>5} {r['pop']:>6} {r['mean']:>7.1f} {r['mx']:>6} "
              f"{r['big']:>6.1f}% {r['chains']:>6.0f}% {r['organs']:>6.1f}% {r['eaten']:>6.0f}%",
              flush=True)
print("\nWant: mean size and >=12px rising, worms falling, organs above the ~23%")
print("random-differentiation baseline, and population not exploding.")
