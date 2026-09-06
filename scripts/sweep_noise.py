"""How much of a creature's movement is its own doing, versus random shoving?

Heading-vs-movement alignment improves with speed but plateaus around 58
degrees even for the fastest swimmers. Thermal noise is 0.35, a value tuned
when bodies were a couple of parts; if it now dominates thrust, then outcomes
are largely random and NO amount of intelligence can matter. This sweeps the
noise floor against alignment and population health.
"""
import math
import rust_world

W, TICKS, STEP = 240, 2500, 6

print(f"{'noise':>7} {'pop':>6} {'meanAngle':>10} {'aligned<60':>11} {'fastAngle':>10}")
for noise in (0.35, 0.20, 0.10, 0.04, 0.0):
    w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
    w.debug_set_thermal_noise(noise)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    before = {i["id"]: (i["positions"][0], i["heading"]) for i in w.individuals_state()}
    for _ in range(STEP):
        w.tick()
    after = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    rows = []
    for pid, (p0, h) in before.items():
        p1 = after.get(pid)
        if p1 is None:
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        cos = (math.cos(h) * dx + math.sin(h) * dy) / dist
        rows.append((dist / STEP, math.degrees(math.acos(max(-1.0, min(1.0, cos))))))
    if not rows:
        print(f"{noise:>7.2f}   no movers"); continue
    angles = [a for _, a in rows]
    fast = [a for sp, a in rows if sp > 0.10]
    print(f"{noise:>7.2f} {w.population():>6} {sum(angles)/len(angles):>9.1f} "
          f"{100*sum(1 for a in angles if a<60)/len(angles):>10.0f}% "
          f"{(sum(fast)/len(fast) if fast else float('nan')):>9.1f}", flush=True)
