"""How scarce must food be before survival actually depends on behaviour?

Measured starting point: mean energy sits near 96 against a reproduction
threshold around 20, and only 11% of deaths are starvation. Food is
effectively infinite, so foraging skill buys nothing and there is no
selective pressure for intelligence in the energy domain -- which is a large
part of why brains learn almost nothing except pursuit.

This sweeps food abundance looking for a regime where:
  * energy is a real constraint (mean energy in the low tens, not ~100),
  * starvation is a meaningful share of deaths (roughly a quarter to a half),
  * predation still matters,
  * and the population does NOT collapse.

Food patch count and regrowth rate are constructor parameters, so this needs
no rebuild.
"""
import rust_world

W, POP_CAP, TICKS = 240, 6000, 4000
SEEDS = (11, 22)


def run(seed, patches, regrow):
    w = rust_world.World(W, regrow, 1.0, POP_CAP, seed, patches)
    w.spawn_random(300)
    lo = 10 ** 9
    for t in range(1, TICKS + 1):
        w.tick()
        if t > 800 and t % 200 == 0:
            lo = min(lo, w.population())
    pop = w.population()
    if pop == 0:
        return None
    inds = w.individuals_state()
    ev = w.events()
    deaths = max(1, ev["deaths"])
    energy = sum(i["energy"] for i in inds) / len(inds)
    size = sum(len(i["positions"]) for i in inds) / len(inds)
    return dict(pop=pop, trough=lo, energy=energy, size=size,
                starved=100.0 * ev["starved"] / deaths,
                eaten=100.0 * ev["eaten"] / deaths)


print(f"{'patches':>8} {'regrow':>7} {'seed':>5} {'pop':>6} {'trough':>7} "
      f"{'energy':>7} {'size':>6} {'starved%':>9} {'eaten%':>7}")
for patches, regrow in ((50, 0.030), (30, 0.020), (18, 0.012), (10, 0.008), (6, 0.005)):
    for seed in SEEDS:
        r = run(seed, patches, regrow)
        if r is None:
            print(f"{patches:>8} {regrow:>7.3f} {seed:>5}   EXTINCT", flush=True)
        else:
            print(f"{patches:>8} {regrow:>7.3f} {seed:>5} {r['pop']:>6} {r['trough']:>7} "
                  f"{r['energy']:>7.1f} {r['size']:>6.1f} {r['starved']:>8.1f}% "
                  f"{r['eaten']:>6.1f}%", flush=True)
