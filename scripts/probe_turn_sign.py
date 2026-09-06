"""Two questions about rotation, measured properly.

1. Does positive body curvature produce positive rotation? Steering assumes it
   does; if the sign is inverted every correction drives the wrong way.
2. Do bodies SELF-rotate when swimming straight? An asymmetric body pushes
   unevenly and spins on its own, and a creature that cannot hold a course
   cannot steer no matter how good its brain is. This is precisely why real
   swimmers are bilaterally symmetric -- so it also tests whether the symmetry
   mechanic confers a real advantage.

Headings wrap, so rotation is measured as an unwrapped rate over short windows
and averaged, not as accumulated change over a long run.
"""
import math
import rust_world

W = 240


def angular_rate(curvature, seed, parts=8, ticks=240, window=4):
    w = rust_world.World(W, 0.0, 1.0, 50, seed, 0)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
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
    rates = []
    prev = 0.0
    for t in range(1, ticks + 1):
        w.debug_set_energy(pid, 60.0)
        w.debug_set_position(pid, W * 0.5, W * 0.5)
        w.tick()
        st = w.individuals_state()
        if not st:
            return None
        h = st[0]["heading"]
        if t % window == 0:
            d = (h - prev + math.pi) % (2 * math.pi) - math.pi   # unwrap
            rates.append(d / window)
            prev = h
    if not rates:
        return None
    return math.degrees(sum(rates) / len(rates))   # deg per tick


print("does curvature steer, and in which direction?  (deg/tick)\n")
print(f"{'curvature':>10} " + " ".join(f"{'seed'+str(s):>9}" for s in (7, 11, 22)))
for c in (-0.8, -0.4, 0.0, 0.4, 0.8):
    row = []
    for seed in (7, 11, 22):
        r = angular_rate(c, seed)
        row.append("n/a" if r is None else f"{r:+.3f}")
    print(f"{c:>10.1f} " + " ".join(f"{v:>9}" for v in row))

print("\nself-rotation while swimming straight (curvature 0) is the row above at 0.0:")
print("a large non-zero value means the body spins on its own and cannot hold")
print("a course -- which no brain can compensate for.")
