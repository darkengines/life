"""Rotational drag: the filter between thrashing and swimming.

Measured, animals hold a mean speed of 0.76 units per tick and a net travel of
0.052 -- fifteen times more motion than progress. Their thrust is locked to
their body (R = 1.000), so if the heading held still that motion would
accumulate into travel. It does not: the body's own beat swings its heading, so
the thrust direction keeps changing and the animal jitters in place.

Rotational damping is the right instrument because it is a low-pass filter. A
real elongated body in water resists rotation strongly -- the head stays
pointed while the body undulates -- so beat-frequency wobble is suppressed
while a sustained, deliberate asymmetry still turns the animal.

Both numbers are reported together, because damping trades them off: too little
and the animal thrashes, too much and it cannot turn at all. This world has
already produced each of those failures separately.
"""
import math
import statistics
import sys

import rust_world


def measure(damping):
    w = rust_world.World(240, 0.004, 1.0, 6000, 11, 50)
    w.debug_set_life_history(1000.0, 0.95)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_angular_damping(damping)
    w.spawn_random(60)
    for ind in w.individuals_state():
        for _ in range(10):
            w.debug_grow(ind["id"])
    w.debug_force_intent(1.0, 0.0)
    for _ in range(500):
        w.tick()
    a = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    h0 = {i["id"]: i["heading"] for i in w.individuals_state()}
    sp = [i["speed"] for i in w.individuals_state() if "speed" in i]
    for _ in range(500):
        w.tick()
    b = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    common = set(a) & set(b)
    if not common:
        return None
    travel = statistics.mean(math.dist(a[i], b[i]) for i in common) / 500.0
    speed = statistics.mean(sp) if sp else 0.0

    # turn rate: reverse the command and see how fast they come about
    w.debug_force_intent(-1.0, 0.0)
    for _ in range(300):
        w.tick()
    h1 = {i["id"]: i["heading"] for i in w.individuals_state()}
    turned = []
    for i in set(h0) & set(h1):
        d = abs((h1[i] - h0[i] + math.pi) % (2 * math.pi) - math.pi)
        turned.append(math.degrees(d) / 300.0)
    return travel, speed, (statistics.mean(turned) if turned else 0.0)


print("efficiency = net travel / speed. 1.0 would be a perfectly straight swimmer.\n")
print(f"{'damping':>8} {'travel':>9} {'speed':>8} {'efficiency':>11} {'turn deg/tick':>14}")
for d in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    r = measure(d)
    if r is None:
        print(f"{d:>8.1f}   all died")
        continue
    travel, speed, turn = r
    eff = travel / max(1e-6, speed)
    print(f"{d:>8.1f} {travel:>9.4f} {speed:>8.4f} {eff:>11.3f} {turn:>14.3f}")
    sys.stdout.flush()
