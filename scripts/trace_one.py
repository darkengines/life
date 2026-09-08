"""Trace the steering loop on ONE animal, tick by tick.

Population statistics said "cannot steer" and three separate hypotheses --
sinking, inverted polarity, thermal noise -- each explained part of it and none
fixed it. At that point the useful move is to stop testing hypotheses against
aggregates and watch the actual chain of causation on a single animal:

    intent -> heading error -> turn_curvature -> rotation -> heading -> travel

Whichever link fails to move when the one before it moves is the broken one.
"""
import math
import rust_world

W = 240
w = rust_world.World(W, 0.004, 1.0, 6000, 11, 50)
w.debug_set_life_history(1000.0, 0.95)
w.debug_set_thermal_noise(0.0)
w.spawn_random(1)
ind = w.individuals_state()[0]
tid = ind["id"]
for _ in range(10):
    w.debug_grow(tid)

# Command due east and hold it.
w.debug_force_intent(1.0, 0.0)
for _ in range(80):
    w.tick()

print(f"{'tick':>5} {'heading':>9} {'curv':>8} {'swim':>6} {'pos':>18} {'travel/tick':>12}")
prev = None
for step in range(14):
    for _ in range(25):
        w.tick()
    st = [i for i in w.individuals_state() if i["id"] == tid]
    if not st:
        print("died")
        break
    i = st[0]
    p = i["positions"][0]
    if prev is None:
        d = "-"
    else:
        dx, dy = p[0] - prev[0], p[1] - prev[1]
        d = f"{dx:+.2f},{dy:+.2f}"
    prev = p
    print(f"{step*25:>5} {math.degrees(i['heading']):>8.1f}d {i['turn_curvature']:>8.3f} "
          f"{i['swim_gain']:>6.2f} {p[0]:>8.1f},{p[1]:<8.1f} {d:>12}")
print("\ncommanded due EAST: heading should converge toward 0 deg and x should climb")
