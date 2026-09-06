"""How strong is self-propulsion, really, against everything opposing it?

In isolation a creature travelled 1.85 units in 600 ticks -- 0.003 units per
tick, when live creatures move at 0.02-0.20. That suggests self-propulsion is
negligible and the movement seen in the world is mostly external shoving,
which would explain why heading and travel look unrelated no matter what is
fixed upstream.

This measures the actual thrust vector against the body axis, and the speed
it can sustain against linear damping.
"""
import math
import rust_world

W = 240
w = rust_world.World(W, 0.03, 1.0, 50, 7, 4)
w.debug_set_thermal_noise(0.0)
w.debug_set_gravity(0.0)
w.spawn_random(1)
pid = w.individuals_state()[0]["id"]
for _ in range(8):
    w.debug_grow(pid)
w.debug_set_position(pid, W * 0.5, W * 0.75)
w.debug_set_heading(pid, 0.0)

print(f"{'tick':>5} {'|thrust|':>10} {'thrust angle':>13} {'speed':>9}")
prev = w.debug_root_pos(pid)
for t in range(1, 61):
    w.debug_set_energy(pid, 60.0)
    w.tick()
    w.debug_set_heading(pid, 0.0)
    f = w.debug_thrust_force(pid)
    now = w.debug_root_pos(pid)
    if f is None or now is None:
        break
    mag = math.hypot(f[0], f[1])
    ang = math.degrees(math.atan2(f[1], f[0]))
    speed = math.hypot(now[0] - prev[0], now[1] - prev[1])
    prev = now
    if t % 10 == 0:
        print(f"{t:>5} {mag:>10.4f} {ang:>12.1f} {speed:>9.4f}")

inds = w.individuals_state()
if inds:
    i = inds[0]
    parts = len(i["positions"])
    print(f"\nbody: {parts} parts, size_scale {i.get('size_scale',1):.2f}, "
          f"bend_amplitude {i.get('bend_amplitude',0):.3f}, "
          f"bend_frequency {i.get('bend_frequency',0):.3f}, "
          f"swim_gain {i.get('swim_gain',1):.2f}")
    print(f"axis offset {math.degrees(w.debug_axis_offset(pid)):.1f} deg")
