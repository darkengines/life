"""Find turning parameters that give BOTH agility and speed.

These four constants have been tuned twice by rebuild-and-look, and each time
the result overshot into the opposite failure: animals nailed to their heading,
then animals spinning on the spot. The reason is that looking at one of them
tells you nothing -- turning hard deforms a body, a deformed body pushes less
water, and past some point commanding a hard turn destroys the propulsion that
the turn itself depends on. So the two must be measured together.

Two forced-intent runs per setting: one commanding a straight line, to see how
fast the animal can travel, and one commanding the opposite of where it is
pointing, to see how quickly it comes round. A setting is only good if it scores
on both -- fast and unable to turn is a projectile, agile and unable to move is
a spinning top, and this world has now produced each of those in turn.
"""
import math
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
N, PARTS = 60, 10

# The first grid returned 0.024-0.046 deg/tick across every setting -- four
# thousand ticks to turn round. That is not a tuning question inside a range,
# it is an order-of-magnitude shortfall, so this sweeps inertia across decades
# rather than nudging it. A usable animal turns 180 degrees in a few hundred
# ticks, i.e. somewhere near half a degree per tick.
GRID = [
    (0.15,  0.30, 0.45, 0.55),
    (0.05,  0.30, 0.45, 0.55),
    (0.015, 0.30, 0.45, 0.55),
    (0.005, 0.30, 0.45, 0.55),
    (0.005, 0.30, 0.80, 0.80),
    (0.0015, 0.30, 0.45, 0.55),
]


def build(params):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
    w.debug_set_turn(*params)
    w.debug_set_life_history(1000.0, 0.95)
    w.debug_set_thermal_noise(0.0)
    w.spawn_random(N)
    for ind in w.individuals_state():
        for _ in range(PARTS):
            w.debug_grow(ind["id"])
    return w


def measure(params):
    # Straight-line speed: command one direction and see how far they get.
    w = build(params)
    w.debug_force_intent(1.0, 0.0)
    for _ in range(400):
        w.tick()
    a = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    for _ in range(400):
        w.tick()
    b = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    common = set(a) & set(b)
    if not common:
        return None
    speed = statistics.mean(math.dist(a[i], b[i]) for i in common) / 400.0
    aligned = []
    for i in common:
        mx, my = b[i][0] - a[i][0], b[i][1] - a[i][1]
        d = math.hypot(mx, my)
        if d > 0.5:
            aligned.append(mx / d)
    on_target = (sum(1 for x in aligned if x > 0.7) / len(aligned)) if aligned else 0.0

    # Turn rate: how much heading changes per tick while commanded to reverse.
    w2 = build(params)
    w2.debug_force_intent(1.0, 0.0)
    for _ in range(200):
        w2.tick()
    h0 = {i["id"]: i["heading"] for i in w2.individuals_state()}
    w2.debug_force_intent(-1.0, 0.0)
    for _ in range(200):
        w2.tick()
    h1 = {i["id"]: i["heading"] for i in w2.individuals_state()}
    turned = []
    for i in set(h0) & set(h1):
        d = abs((h1[i] - h0[i] + math.pi) % (2 * math.pi) - math.pi)
        turned.append(math.degrees(d) / 200.0)
    return speed, on_target, (statistics.mean(turned) if turned else 0.0)


print("commanded straight, then commanded to reverse\n")
print(f"{'inert':>6} {'maxang':>7} {'asym':>6} {'lean':>6} | {'speed/tick':>11} "
      f"{'on-target':>10} {'turn deg/tick':>14}")
for params in GRID:
    r = measure(params)
    if r is None:
        print(f"{params[0]:>6.2f} {params[1]:>7.2f} {params[2]:>6.2f} {params[3]:>6.2f} |  all died")
        continue
    sp, on, turn = r
    print(f"{params[0]:>6.2f} {params[1]:>7.2f} {params[2]:>6.2f} {params[3]:>6.2f} | "
          f"{sp:>11.4f} {on*100:>9.0f}% {turn:>14.3f}")
    sys.stdout.flush()
