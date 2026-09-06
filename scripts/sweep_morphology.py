"""How much of "creatures are basically worms" is one growth constant?

Growth weights "extend an existing tip in the same direction" at 8x every
other placement. That is a strong, deliberate bias toward elongation, and it
is the prime suspect for bodies reading as worms. This sweeps it and reports
what shapes actually result, plus whether the population survives -- a bushier
body is only an improvement if it can still live.
"""
import rust_world

W, POP_CAP, TICKS, SEEDS = 240, 6000, 3000, (11, 22)

print(f"{'tipW':>6} {'seed':>5} {'pop':>6} {'meanPx':>7} {'maxPx':>6} {'pureChains':>11} "
      f"{'branchPts':>10} {'depth/parts':>12}")
for tw in (8.0, 4.0, 2.0, 1.0):
    for seed in SEEDS:
        w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
        w.debug_set_growth_tip_weight(tw)
        w.spawn_random(300)
        for _ in range(TICKS):
            w.tick()
        inds = w.individuals_state()
        if not inds:
            print(f"{tw:>6.1f} {seed:>5}   EXTINCT", flush=True)
            continue
        chains = branches = 0
        ratios = []
        sizes = []
        for i in inds:
            parents = i["parents"]
            n = len(parents)
            sizes.append(n)
            kids = {}
            for k, p in enumerate(parents):
                if p >= 0:
                    kids[p] = kids.get(p, 0) + 1
            b = sum(1 for c in kids.values() if c > 1)
            branches += b
            if b == 0:
                chains += 1
            depth = {}
            for k, p in enumerate(parents):
                depth[k] = 1 if p < 0 else depth.get(p, 1) + 1
            if n > 1:
                ratios.append(max(depth.values()) / n)
        print(f"{tw:>6.1f} {seed:>5} {len(inds):>6} {sum(sizes)/len(sizes):>7.1f} "
              f"{max(sizes):>6} {100*chains/len(inds):>10.1f}% {branches/len(inds):>10.2f} "
              f"{(sum(ratios)/len(ratios) if ratios else float('nan')):>12.2f}", flush=True)
