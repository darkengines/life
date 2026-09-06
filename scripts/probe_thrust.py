"""Isolate locomotion: does a body actually swim along its own axis?

In the live world, the angle between where a creature points and where it
moves averages ~91 degrees -- statistically indistinguishable from random.
That is not "swims backwards", it is "steering does not control movement",
which would cap how much any brain can ever matter.

Confounds in the live world are many: thermal noise, gravity, contact with
other bodies, terrain. This strips all of that away -- one creature, no
neighbours, and the same heading held fixed -- and asks the narrow question:
over many ticks, which way does it go relative to its own heading, and how
far? Then it repeats across headings, because a propulsion bug that happens
to work at one orientation but not others would otherwise hide.
"""
import math
import rust_world

W = 240


def probe(heading_deg, ticks=400, noise_free=True):
    w = rust_world.World(W, 0.03, 1.0, 50, 7, 6)
    # A lone individual in open water, well clear of the floor and any rock.
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    # Grow it into a body with several segments so undulation has something
    # to act on, the way a real swimmer does.
    for _ in range(10):
        w.debug_grow(0)
    w.debug_set_position(0, W * 0.5, W * 0.6)
    w.debug_set_heading(0, math.radians(heading_deg))
    if noise_free:
        w.debug_set_noise(0.0)     # remove thermal jitter
        w.debug_set_gravity(0.0)   # remove sinking
    start = w.debug_root_pos(0)
    for _ in range(ticks):
        w.tick()
        if not w.debug_is_alive(0):
            return None
        w.debug_set_heading(0, math.radians(heading_deg))  # hold course
    end = w.debug_root_pos(0)
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return dist, None
    moved = math.degrees(math.atan2(dy, dx))
    rel = (moved - heading_deg + 180) % 360 - 180
    return dist, rel


print("one creature, no noise, no gravity, no neighbours, heading held fixed")
print(f"{'heading':>8} {'distance':>9} {'movement vs heading':>21}")
results = []
for h in (0, 45, 90, 135, 180, 225, 270, 315):
    r = probe(h)
    if r is None:
        print(f"{h:>8}   (died or failed to spawn)")
        continue
    dist, rel = r
    if rel is None:
        print(f"{h:>8} {dist:>9.2f}   (did not move)")
    else:
        print(f"{h:>8} {dist:>9.2f} {rel:>20.1f} deg")
        results.append(rel)

if results:
    # Consistency matters more than the absolute angle: a constant offset is a
    # sign convention, scattered angles mean propulsion is not axis-locked.
    import statistics
    print(f"\nmean offset {statistics.mean(results):+.1f} deg, "
          f"spread (stdev) {statistics.pstdev(results):.1f} deg")
    print("A tight spread means thrust IS locked to the body axis (even if the")
    print("sign is wrong). A wide spread means propulsion is not directional.")
