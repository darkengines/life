"""Is poor population alignment a physics failure or a steering-in-circles artefact?

Isolated physics is now correct: thrust is perfectly body-locked, points
forward, and steering has real authority. Yet population alignment looks
worse, not better. One explanation needs no bug at all: giving brains real
turning authority means an untrained brain holding a constant curvature swims
in a tight ARC, and an arc's displacement never lines up with instantaneous
heading no matter how correct the physics is.

If creatures holding a near-straight posture align well and only the strongly
curved ones scatter, the physics is fine and what remains is a learning
problem. If even straight swimmers scatter, something is still wrong.
"""
import math
import rust_world

W, TICKS, STEP = 240, 2200, 6
w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)
for _ in range(TICKS):
    w.tick()

before = {i["id"]: (i["positions"][0], i["heading"], i.get("turn_curvature", 0.0))
          for i in w.individuals_state()}
for _ in range(STEP):
    w.tick()
after = {i["id"]: i["positions"][0] for i in w.individuals_state()}

rows = []
for pid, (p0, h, curv) in before.items():
    p1 = after.get(pid)
    if p1 is None:
        continue
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        continue
    cos = (math.cos(h) * dx + math.sin(h) * dy) / dist
    rows.append((abs(curv), dist / STEP, math.degrees(math.acos(max(-1.0, min(1.0, cos))))))

print(f"n={len(rows)}\n")
print(f"{'|curvature| band':>18} {'count':>6} {'mean angle':>11} {'aligned<60':>11} {'mean speed':>11}")
for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 99)):
    sel = [(s, a) for c, s, a in rows if lo <= c < hi]
    if not sel:
        continue
    angles = [a for _, a in sel]
    speeds = [s for s, _ in sel]
    print(f"{lo:>8.2f}-{hi:<9.2f} {len(sel):>6} {sum(angles)/len(angles):>10.1f} "
          f"{100*sum(1 for a in angles if a < 60)/len(angles):>10.0f}% "
          f"{sum(speeds)/len(speeds):>11.4f}")
print("\nStraight-posture creatures aligning well => physics is fine and the")
print("remaining problem is that brains have not learned to hold a course.")
