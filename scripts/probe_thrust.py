"""Does a body swim along its own axis, in isolation?

Population-level measurements kept returning ~85 degrees (statistically
random) no matter what was changed, but the live world confounds everything:
contact with neighbours, terrain, gravity, capture, and a different body plan
for every creature. This strips all of it away.

One creature. No neighbours. No thermal noise. No gravity. Heading held fixed
each tick so rotation cannot drift. Then: over many ticks, which way does it
actually travel relative to the direction it is facing?

The key question is CONSISTENCY, not the absolute angle. A constant offset
across headings would just be a sign convention. Angles that scatter mean
thrust is not locked to the body at all, which would mean no brain can ever
steer effectively -- the ceiling on how much intelligence can matter.
"""
import math
import statistics
import rust_world

W = 240


def probe(heading_deg, parts=8, ticks=600):
    w = rust_world.World(W, 0.03, 1.0, 50, 7, 4)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    pid = inds[0]["id"]
    for _ in range(parts):
        w.debug_grow(pid)
    # open water, clear of the floor and any reef
    w.debug_set_position(pid, W * 0.5, W * 0.75)
    h = math.radians(heading_deg)
    w.debug_set_heading(pid, h)
    start = w.debug_root_pos(pid)
    for _ in range(ticks):
        w.debug_set_energy(pid, 60.0)  # keep it alive; we are measuring physics, not survival
        w.tick()
        if not w.debug_is_alive(pid):
            return None
        w.debug_set_heading(pid, h)   # hold course; isolate translation
    end = w.debug_root_pos(pid)
    if start is None or end is None:
        return None
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return dist, None, w.debug_axis_offset(pid)
    rel = (math.degrees(math.atan2(dy, dx)) - heading_deg + 180) % 360 - 180
    return dist, rel, w.debug_axis_offset(pid)


print("one creature | no noise | no gravity | no neighbours | heading held fixed\n")
print(f"{'heading':>8} {'distance':>9} {'travel vs heading':>19} {'axis offset':>12}")
rels, dists = [], []
for hdg in range(0, 360, 30):
    r = probe(hdg)
    if r is None:
        print(f"{hdg:>8}   (died / failed)")
        continue
    dist, rel, axis = r
    axis_s = "n/a" if axis is None else f"{math.degrees(axis):.1f}"
    if rel is None:
        print(f"{hdg:>8} {dist:>9.3f} {'did not move':>19} {axis_s:>12}")
    else:
        print(f"{hdg:>8} {dist:>9.3f} {rel:>16.1f} deg {axis_s:>12}")
        rels.append(rel)
        dists.append(dist)

if len(rels) >= 3:
    # circular spread: how consistent is the travel direction across headings?
    mx = statistics.mean(math.cos(math.radians(a)) for a in rels)
    my = statistics.mean(math.sin(math.radians(a)) for a in rels)
    R = math.hypot(mx, my)
    mean_dir = math.degrees(math.atan2(my, mx))
    print(f"\nmean travel offset {mean_dir:+.1f} deg,  consistency R={R:.3f}")
    print(f"mean distance travelled {statistics.mean(dists):.2f} over 600 ticks")
    print("R near 1 = thrust is locked to the body (a constant offset is just a")
    print("sign/axis convention and is trivially fixable). R near 0 = propulsion")
    print("direction is unrelated to the body, and steering cannot work.")
