"""Is the residual misalignment noise, or is propulsion still not axis-locked?

Mean heading-vs-movement angle is ~79 degrees. If that is thermal noise and
gravity swamping slow drifters, then FAST movers should be well aligned and
slow ones random. If even fast movers are scattered, propulsion itself is not
locked to the body axis and the physics is still wrong.
"""
import math
import rust_world

W, TICKS, STEP = 240, 2500, 6
w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
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

rows.sort()
n = len(rows)
print(f"n={n}  (speed = world units per tick)\n")
print(f"{'speed band':>16} {'count':>6} {'mean angle':>11} {'aligned<60deg':>14}")
bands = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 99)]
for lo, hi in bands:
    sel = [a for s, a in rows if lo <= s < hi]
    if not sel:
        continue
    aligned = sum(1 for a in sel if a < 60)
    print(f"{lo:>7.2f}-{hi:<7.2f} {len(sel):>6} {sum(sel)/len(sel):>10.1f} deg "
          f"{100*aligned/len(sel):>12.0f}%")
print("\nIf alignment improves sharply with speed, propulsion works and the")
print("noise floor explains the rest. If it stays flat, thrust is not")
print("directional and steering still cannot control movement.")
