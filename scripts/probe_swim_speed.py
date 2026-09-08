"""Can these animals actually swim, and how fast?

A fair question after a session in which they have been shown to sink, spin,
and fail to turn. Thrust has been verified to exist and to be locked to the
body (R = 1.000), but existing is not the same as being enough: what matters is
the speed an animal can actually hold against drag, measured as a fraction of
what the engine permits.
"""
import math
import statistics

import rust_world

w = rust_world.World(240, 0.004, 1.0, 6000, 11, 50)
w.debug_set_life_history(1000.0, 0.95)   # no births during the measurement
w.debug_set_thermal_noise(0.0)           # no random walk to flatter the number
w.spawn_random(60)
for ind in w.individuals_state():
    for _ in range(10):
        w.debug_grow(ind["id"])
w.debug_force_intent(1.0, 0.0)           # brain out of the loop, one command

for _ in range(600):
    w.tick()
a = {i["id"]: i["positions"][0] for i in w.individuals_state()}
engine_speed = [i["speed"] for i in w.individuals_state() if "speed" in i]
for _ in range(600):
    w.tick()
b = {i["id"]: i["positions"][0] for i in w.individuals_state()}

common = set(a) & set(b)
travel = [math.dist(a[i], b[i]) / 600.0 for i in common]
print("CAN THEY SWIM?  straight command, no noise, no births\n")
print(f"  net travel     {statistics.mean(travel):.4f} units/tick")
print(f"  fastest        {max(travel):.4f} units/tick")
if engine_speed:
    print(f"  engine speed   {statistics.mean(engine_speed):.4f} mean  "
          f"{max(engine_speed):.4f} max")
print(f"  MAX_SPEED cap  3.0000")
print(f"\n  -> using {100 * statistics.mean(travel) / 3.0:.1f}% of the speed the engine allows")
print(f"  -> a body is ~10 units long, so that is one body length every "
      f"{10 / max(1e-6, statistics.mean(travel)):.0f} ticks")
