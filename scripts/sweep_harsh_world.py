"""Does a harsher world produce complex creatures instead of worms?

The live world at 19400 ticks was 3.5-part worms, 53% unbranched, organs at
their random baseline -- while 6000-tick runs looked fine. Complexity emerges
and then collapses, because abundant food plus overcrowding lets small fast
breeders out-compete everything else.

This runs LONG (18000 ticks, past the point where the collapse showed up) and
sweeps food scarcity together with meal value, asking whether large,
organ-bearing, branched creatures persist rather than merely appear.
"""
import collections
import rust_world

W, POP_CAP, TICKS, SEEDS = 240, 6000, 18000, (11, 22)


def summarise(w):
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
    chains = sum(1 for i in inds
                 if not any(c > 1 for c in collections.Counter(
                     p for p in i["parents"] if p >= 0).values()))
    return dict(pop=len(inds), mean=sum(sizes)/len(sizes), mx=max(sizes),
                big=100*sum(1 for n in sizes if n >= 12)/len(sizes),
                worms=100*chains/len(inds),
                organs=100*(total - tally.get(0, 0))/max(1, total))


print(f"{'regrow':>7} {'meal':>6} {'seed':>5} {'pop':>6} {'meanPx':>7} {'maxPx':>6} "
      f"{'>=12px':>7} {'worms':>7} {'organ%':>7}")
for regrow, meal in ((0.030, 1.4), (0.006, 1.4), (0.006, 8.0), (0.002, 8.0)):
    for seed in SEEDS:
        w = rust_world.World(W, regrow, 1.0, POP_CAP, seed, 50)
        w.debug_set_meal_energy(meal)
        w.spawn_random(300)
        for _ in range(TICKS):
            w.tick()
        r = summarise(w)
        if r is None:
            print(f"{regrow:>7.3f} {meal:>6.1f} {seed:>5}   EXTINCT", flush=True)
            continue
        print(f"{regrow:>7.3f} {meal:>6.1f} {seed:>5} {r['pop']:>6} {r['mean']:>7.1f} "
              f"{r['mx']:>6} {r['big']:>6.1f}% {r['worms']:>6.0f}% {r['organs']:>6.1f}%",
              flush=True)
print("\nOrgans above ~23% means selection favours them over the random baseline.")
