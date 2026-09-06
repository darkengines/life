"""Does swimming straight actually pay?

Creatures that hold a straight posture go where they point (58.7 deg mean
alignment vs ~100 for heavily curved ones), but 61% of the population swims in
tight arcs. Selection will only straighten them out if straightness earns
something. If circling costs nothing, brains will never learn to hold a
course, and directed movement -- the thing that makes intelligence worth
having -- stays pointless.

This tracks individuals over time and asks whether the straight ones end up
with more energy, live longer, and reproduce more.
"""
import math
import rust_world

W, WARMUP, WINDOW = 240, 1500, 600
w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)
for _ in range(WARMUP):
    w.tick()

# characterise each individual's typical posture over a stretch of its life
samples = {}
for _ in range(200):
    w.tick()
    for i in w.individuals_state():
        c = abs(i.get("turn_curvature", 0.0))
        s = samples.setdefault(i["id"], [0.0, 0, 0.0])
        s[0] += c
        s[1] += 1
        s[2] = i["energy"]

start = {pid: (v[0] / v[1], v[2]) for pid, v in samples.items() if v[1] >= 20}
start_ages = {i["id"]: i["age"] for i in w.individuals_state()}

for _ in range(WINDOW):
    w.tick()

alive_now = {i["id"]: i for i in w.individuals_state()}
bands = [(0.0, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 99)]
print(f"tracked {len(start)} individuals over {WINDOW} ticks\n")
print(f"{'|curvature|':>14} {'n':>5} {'survived':>9} {'energy gain':>12} {'aged':>7}")
for lo, hi in bands:
    grp = [(pid, c, e) for pid, (c, e) in start.items() if lo <= c < hi]
    if not grp:
        continue
    survived = [p for p, _, _ in grp if p in alive_now]
    gains = [alive_now[p]["energy"] - e for p, _, e in grp if p in alive_now]
    ages = [alive_now[p]["age"] - start_ages.get(p, 0) for p, _, _ in grp if p in alive_now]
    print(f"{lo:>6.2f}-{hi:<7.2f} {len(grp):>5} {100*len(survived)/len(grp):>8.0f}% "
          f"{(sum(gains)/len(gains) if gains else float('nan')):>12.1f} "
          f"{(sum(ages)/len(ages) if ages else 0):>7.0f}")
print("\nIf straight swimmers survive and gain no better than circling ones,")
print("nothing selects for holding a course and intelligence has no purchase.")
