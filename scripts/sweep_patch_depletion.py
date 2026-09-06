"""Make patches exhaustible, so that travelling between them pays.

Measured: creatures that swim straight LOSE energy (-41) while circling ones
gain (+38 to +78). Nothing selects for holding a course, so brains never learn
to, and directed movement -- the thing that would make vision and intelligence
worth having -- is pointless. The cause is that food regrows at 3% per tick, so
a patch never runs out and staying inside one beats leaving it. The creatures
found area-restricted search, which is a real strategy and correct here.

The fix is not less food overall (swept before: that just collapses the
population via the Allee trap) but SLOWER LOCAL REGROWTH with richer patches:
the same standing crop, but exhaustible, so a grazer must move on. This sweeps
regrowth against patch richness and asks whether straightness starts to pay.
"""
import rust_world

W, POP_CAP, WARMUP, WINDOW = 240, 6000, 1500, 600


def run(regrow, cap, patches, seed=11):
    w = rust_world.World(W, regrow, cap, POP_CAP, seed, patches)
    w.spawn_random(300)
    for _ in range(WARMUP):
        w.tick()
    samples = {}
    for _ in range(200):
        w.tick()
        for i in w.individuals_state():
            s = samples.setdefault(i["id"], [0.0, 0, 0.0])
            s[0] += abs(i.get("turn_curvature", 0.0))
            s[1] += 1
            s[2] = i["energy"]
    start = {p: (v[0] / v[1], v[2]) for p, v in samples.items() if v[1] >= 20}
    for _ in range(WINDOW):
        w.tick()
    alive = {i["id"]: i for i in w.individuals_state()}
    if not start:
        return None
    straight = [(p, e) for p, (c, e) in start.items() if c < 0.2]
    curved = [(p, e) for p, (c, e) in start.items() if c >= 0.45]

    def stats(grp):
        if not grp:
            return None, None
        surv = [p for p, _ in grp if p in alive]
        gains = [alive[p]["energy"] - e for p, e in grp if p in alive]
        return (100 * len(surv) / len(grp),
                sum(gains) / len(gains) if gains else float("nan"))

    ss, sg = stats(straight)
    cs, cg = stats(curved)
    return dict(pop=w.population(), n_str=len(straight), n_cur=len(curved),
                s_surv=ss, s_gain=sg, c_surv=cs, c_gain=cg)


print(f"{'regrow':>7} {'cap':>5} {'patches':>8} {'pop':>6} "
      f"{'straight surv/gain':>20} {'curved surv/gain':>19}")
for regrow, cap, patches in ((0.030, 1.0, 50), (0.010, 3.0, 40),
                             (0.004, 6.0, 30), (0.0015, 12.0, 24)):
    r = run(regrow, cap, patches)
    if r is None or r["pop"] == 0:
        print(f"{regrow:>7.4f} {cap:>5.1f} {patches:>8}   EXTINCT", flush=True)
        continue
    def fmt(s, g, n):
        if s is None:
            return f"{'n/a':>20}"
        return f"{s:>6.0f}% {g:>8.1f} (n={n})"
    print(f"{regrow:>7.4f} {cap:>5.1f} {patches:>8} {r['pop']:>6} "
          f"{fmt(r['s_surv'], r['s_gain'], r['n_str'])} "
          f"{fmt(r['c_surv'], r['c_gain'], r['n_cur'])}", flush=True)
print("\nWant: straight swimmers doing BETTER than circling ones, population alive.")
