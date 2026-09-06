"""Is propulsion locked to the body axis?

Measuring net displacement turned out to be the wrong instrument: a creature
travels far enough to meet different terrain, food and boundaries depending
which way it set off, so the environment breaks the rotational symmetry the
test depends on and the result swings wildly for reasons that have nothing to
do with propulsion.

This measures the thrust VECTOR instead, with the creature pinned in place and
its heading held, so nothing about the surroundings varies. Averaged over many
ticks -- several full undulation cycles -- the mean thrust direction in the
body's own frame is the honest answer to "which way does this body push?".

Consistency across headings is what matters. A constant offset is only a sign
or axis convention. Angles that scatter mean thrust is not tied to the body,
and no brain could steer.
"""
import math
import statistics
import rust_world

W = 240


def probe(heading_deg, parts=8, ticks=400):
    # No food patches, so grazing cannot vary by position either.
    w = rust_world.World(W, 0.0, 1.0, 50, 7, 0)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
    # Take the brain out of the loop: hold the body straight and swimming at a
    # fixed effort, so what is measured is propulsion and not behaviour.
    w.debug_freeze_locomotion(0.0, 1.0)
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    pid = inds[0]["id"]
    for _ in range(parts):
        w.debug_grow(pid)
    h = math.radians(heading_deg)
    fx = fy = 0.0
    n = 0
    for _ in range(ticks):
        # Pin position and heading every tick: the body may deform and push,
        # but it never travels anywhere new, so the environment is constant.
        w.debug_set_position(pid, W * 0.5, W * 0.75)
        w.debug_set_heading(pid, h)
        w.debug_set_energy(pid, 60.0)
        w.tick()
        if not w.debug_is_alive(pid):
            return None
        f = w.debug_thrust_force(pid)
        if f is None:
            return None
        fx += f[0]
        fy += f[1]
        n += 1
    if n == 0 or math.hypot(fx, fy) < 1e-9:
        return 0.0, None, w.debug_axis_offset(pid)
    mag = math.hypot(fx / n, fy / n)
    rel = (math.degrees(math.atan2(fy, fx)) - heading_deg + 180) % 360 - 180
    return mag, rel, w.debug_axis_offset(pid)


print("one creature | pinned in place | no noise, gravity, food or neighbours")
print("mean THRUST direction over 400 ticks, in the body's own frame\n")
print(f"{'heading':>8} {'|mean thrust|':>14} {'thrust vs heading':>19}")
rels, mags = [], []
for hdg in range(0, 360, 30):
    r = probe(hdg)
    if r is None:
        print(f"{hdg:>8}   (died / failed)")
        continue
    mag, rel, _axis = r
    if rel is None:
        print(f"{hdg:>8} {mag:>14.5f} {'no net thrust':>19}")
    else:
        print(f"{hdg:>8} {mag:>14.5f} {rel:>16.1f} deg")
        rels.append(rel)
        mags.append(mag)

if len(rels) >= 3:
    mx = statistics.mean(math.cos(math.radians(a)) for a in rels)
    my = statistics.mean(math.sin(math.radians(a)) for a in rels)
    R = math.hypot(mx, my)
    print(f"\nmean thrust offset {math.degrees(math.atan2(my, mx)):+.1f} deg   consistency R={R:.3f}")
    print(f"mean |thrust| {statistics.mean(mags):.5f}")
    print("R near 1 = thrust is locked to the body; the offset is then just a")
    print("convention and is trivially corrected. R near 0 = it is not.")
