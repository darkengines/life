"""Is the backwards-thrust offset a global convention error, or body-specific?

With the brain frozen out, thrust is perfectly locked to the body (R=1.000)
and points +162.6 degrees from "forward" for one particular body -- i.e.
almost exactly backwards, which is precisely the tail-first swimming that was
reported by eye.

If that offset is the SAME across different body plans it is one global
convention error and correcting it is a one-line fix. If it varies per body,
then the centroid-based anterior axis is simply the wrong basis for "forward"
and needs replacing with something derived from how the body actually pushes.
"""
import math
import statistics
import rust_world

W = 240


def thrust_offset(seed, parts, ticks=300):
    w = rust_world.World(W, 0.0, 1.0, 50, seed, 0)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
    w.debug_freeze_locomotion(0.0, 1.0)
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    pid = inds[0]["id"]
    for _ in range(parts):
        w.debug_grow(pid)
    fx = fy = 0.0
    for _ in range(ticks):
        w.debug_set_position(pid, W * 0.5, W * 0.75)
        w.debug_set_heading(pid, 0.0)
        w.debug_set_energy(pid, 60.0)
        w.tick()
        if not w.debug_is_alive(pid):
            return None
        f = w.debug_thrust_force(pid)
        if f is None:
            return None
        fx += f[0]
        fy += f[1]
    if math.hypot(fx, fy) < 1e-9:
        return None
    n = len(w.individuals_state()[0]["positions"])
    return math.degrees(math.atan2(fy, fx)), math.hypot(fx, fy) / ticks, n


print(f"{'seed':>5} {'grown':>6} {'parts':>6} {'|thrust|':>10} {'thrust angle':>13}")
angles = []
for seed in (7, 11, 22, 33, 44):
    for grow in (4, 8, 14):
        r = thrust_offset(seed, grow)
        if r is None:
            print(f"{seed:>5} {grow:>6}   (no net thrust / died)")
            continue
        ang, mag, n = r
        print(f"{seed:>5} {grow:>6} {n:>6} {mag:>10.4f} {ang:>12.1f} deg")
        angles.append(ang)

if len(angles) >= 3:
    mx = statistics.mean(math.cos(math.radians(a)) for a in angles)
    my = statistics.mean(math.sin(math.radians(a)) for a in angles)
    R = math.hypot(mx, my)
    print(f"\nacross {len(angles)} different bodies: mean angle "
          f"{math.degrees(math.atan2(my, mx)):+.1f} deg, consistency R={R:.3f}")
    print("R near 1 => one global convention error, correctable in one line.")
    print("R low    => 'forward' must be derived from thrust, not from shape.")
