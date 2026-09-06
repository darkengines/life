"""Where should grazing stop paying, so that a food chain forms without collapse?

Making grazing yield fall with body mass is what separates grazers from
predators -- but at the first value tried (halving by mass 6) essentially
every creature was too big to feed itself and the world collapsed to a single
individual, with starvation jumping from 11% to 37% of deaths. The mechanism
is right; the threshold was not. What we want is a world where small bodies
still live comfortably off the field, large ones genuinely cannot, and BOTH
persist.
"""
import rust_world

W, POP_CAP, TICKS, SEEDS = 240, 6000, 4000, (11, 22)

print(f"{'ref':>6} {'seed':>5} {'pop':>6} {'trough':>7} {'energy':>7} {'meanSz':>7} "
      f"{'maxSz':>6} {'starved%':>9} {'eaten%':>7} {'big(>12)':>9}")
for ref in (6.0, 15.0, 30.0, 60.0, 1e9):
    for seed in SEEDS:
        w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
        w.debug_set_graze_mass_ref(ref)
        w.spawn_random(300)
        lo = 10**9
        for t in range(1, TICKS + 1):
            w.tick()
            if t > 800 and t % 200 == 0:
                lo = min(lo, w.population())
        pop = w.population()
        if pop == 0:
            print(f"{ref:>6.0f} {seed:>5}   EXTINCT", flush=True)
            continue
        inds = w.individuals_state()
        ev = w.events(); deaths = max(1, ev["deaths"])
        sizes = [len(i["positions"]) for i in inds]
        energy = sum(i["energy"] for i in inds) / len(inds)
        big = sum(1 for n in sizes if n > 12)
        print(f"{ref:>6.0f} {seed:>5} {pop:>6} {lo:>7} {energy:>7.1f} "
              f"{sum(sizes)/len(sizes):>7.1f} {max(sizes):>6} "
              f"{100*ev['starved']/deaths:>8.1f}% {100*ev['eaten']/deaths:>6.1f}% "
              f"{big:>9}", flush=True)
