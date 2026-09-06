"""Find a per-part reproduction cost that self-limits population without
courting extinction.

A flat reproduction fee let population slam into the cap; pricing offspring
by their body size fixed that but at 1.2/part the first run dipped to 170
individuals, which is close enough to extinction (and to the Allee trap this
world already has) to be luck-dependent. What we want is a population that
oscillates inside a band: never pinned at the cap, never near zero.
"""
import rust_world

W, POP_CAP, TICKS = 240, 6000, 6000


def run(seed, cost):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
    w.debug_set_repro_cost_per_part(cost)
    w.spawn_random(300)
    lo, hi, series = 10**9, 0, []
    for t in range(1, TICKS + 1):
        w.tick()
        if t % 250 == 0:
            p = w.population()
            if p == 0:
                return dict(dead_at=t, lo=0, hi=hi, final=0, mean_px=0.0, capped=False)
            lo, hi = min(lo, p), max(hi, p)
            series.append(p)
    inds = w.individuals_state()
    mean_px = sum(len(i["positions"]) for i in inds) / max(1, len(inds))
    # only count the trough AFTER the initial growth phase, so the empty
    # starting world isn't scored as a crash
    settled = series[4:] or series
    return dict(dead_at=None, lo=min(settled), hi=hi, final=series[-1],
                mean_px=mean_px, capped=hi >= POP_CAP)


print(f"{'cost':>5} {'seed':>5} {'trough':>7} {'peak':>7} {'final':>7} {'mean_px':>8} {'capped':>7}")
for cost in (0.8, 1.2, 1.6, 2.2):
    for seed in (11, 22):
        r = run(seed, cost)
        if r["dead_at"]:
            print(f"{cost:>5.1f} {seed:>5}  EXTINCT at tick {r['dead_at']}", flush=True)
        else:
            print(f"{cost:>5.1f} {seed:>5} {r['lo']:>7} {r['hi']:>7} {r['final']:>7} "
                  f"{r['mean_px']:>8.1f} {str(r['capped']):>7}", flush=True)
