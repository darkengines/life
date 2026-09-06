"""Can a creature turn on purpose more than it spins by accident?

Bodies self-rotate 0.4-2.7 degrees per tick while trying to swim straight,
because an asymmetric body pushes water unevenly. If that involuntary spin is
comparable to the rotation a deliberate body curvature produces, then steering
is drowned out by noise from the animal's own anatomy -- and no brain can hold
a course.

The figure of merit is the RATIO: deliberate turn rate divided by involuntary
spin. Damping suppresses both, so the question is whether it suppresses spin
more than it suppresses steering.
"""
import math
import rust_world

W = 240


def rotation(damping, curvature, seed, parts=10, ticks=200, window=4):
    w = rust_world.World(W, 0.0, 1.0, 50, seed, 0)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
    w.debug_set_angular_damping(damping)
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    pid = inds[0]["id"]
    for _ in range(parts):
        w.debug_grow(pid)
    w.debug_set_position(pid, W * 0.5, W * 0.5)
    w.debug_set_heading(pid, 0.0)
    w.debug_freeze_locomotion(curvature, 1.0)
    rates, prev = [], 0.0
    for t in range(1, ticks + 1):
        w.debug_set_energy(pid, 60.0)
        w.debug_set_position(pid, W * 0.5, W * 0.5)
        w.tick()
        st = w.individuals_state()
        if not st:
            return None
        h = st[0]["heading"]
        if t % window == 0:
            d = (h - prev + math.pi) % (2 * math.pi) - math.pi
            rates.append(d / window)
            prev = h
    return math.degrees(sum(rates) / len(rates)) if rates else None


SEEDS = (7, 11, 22, 33)
print(f"{'damping':>8} {'|self-spin|':>12} {'|steer turn|':>13} {'ratio':>8}")
for damp in (0.5, 2.0, 5.0, 12.0, 30.0):
    spins, turns = [], []
    for seed in SEEDS:
        s0 = rotation(damp, 0.0, seed)
        # deliberate turn measured as the CHANGE caused by curvature
        sp = rotation(damp, 0.6, seed)
        if s0 is None or sp is None:
            continue
        spins.append(abs(s0))
        turns.append(abs(sp - s0))
    if not spins:
        print(f"{damp:>8.1f}   (failed)"); continue
    ms = sum(spins) / len(spins)
    mt = sum(turns) / len(turns)
    print(f"{damp:>8.1f} {ms:>12.3f} {mt:>13.3f} {(mt/ms if ms>1e-9 else float('inf')):>8.2f}", flush=True)
print("\nHigher ratio = steering wins over the body's own wobble.")
