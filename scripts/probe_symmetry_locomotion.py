"""Does bilateral symmetry actually make a body swim straight?

Measured: bodies self-rotate up to ~1 degree per tick while trying to swim
straight, because an asymmetric body pushes water unevenly. A creature that
spins cannot hold a course, and no brain can compensate for that -- which is a
hard ceiling on how much intelligence can matter.

Real swimmers are bilaterally symmetric for exactly this reason. This builds
the same body twice, once with every node symmetric (so growth emits mirrored
pairs) and once without, and compares how much each spins and how far each
travels.
"""
import math
import rust_world

W = 240


def build(seed, symmetric, parts=10, ticks=240, window=4):
    w = rust_world.World(W, 0.0, 1.0, 50, seed, 0)
    w.debug_set_thermal_noise(0.0)
    w.debug_set_gravity(0.0)
    w.spawn_random(1)
    inds = w.individuals_state()
    if not inds:
        return None
    pid = inds[0]["id"]
    # Set symmetry BEFORE growing, so mirrored pairs are actually emitted.
    w.debug_set_symmetry(pid, symmetric)
    for _ in range(parts):
        w.debug_grow(pid)
        w.debug_set_symmetry(pid, symmetric)
    w.debug_set_position(pid, W * 0.5, W * 0.5)
    w.debug_set_heading(pid, 0.0)
    w.debug_freeze_locomotion(0.0, 1.0)     # swim straight, brain out of the loop
    start = w.debug_root_pos(pid)
    rates, prev = [], 0.0
    for t in range(1, ticks + 1):
        w.debug_set_energy(pid, 60.0)
        w.tick()
        st = w.individuals_state()
        if not st:
            return None
        h = st[0]["heading"]
        if t % window == 0:
            d = (h - prev + math.pi) % (2 * math.pi) - math.pi
            rates.append(abs(d) / window)
            prev = h
    end = w.debug_root_pos(pid)
    n = len(w.individuals_state()[0]["positions"])
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    return math.degrees(sum(rates) / len(rates)), dist, n


print("swimming straight, brain frozen out: spin rate and distance covered\n")
print(f"{'seed':>5} {'symmetric':>10} {'parts':>6} {'|spin| deg/tick':>16} {'distance':>10}")
sym_spin, asym_spin = [], []
for seed in (7, 11, 22, 33, 44):
    for sym in (False, True):
        r = build(seed, sym)
        if r is None:
            print(f"{seed:>5} {str(sym):>10}   (failed)")
            continue
        spin, dist, n = r
        (sym_spin if sym else asym_spin).append(spin)
        print(f"{seed:>5} {str(sym):>10} {n:>6} {spin:>16.3f} {dist:>10.2f}")

if sym_spin and asym_spin:
    a = sum(asym_spin) / len(asym_spin)
    b = sum(sym_spin) / len(sym_spin)
    print(f"\nmean spin  asymmetric {a:.3f}   symmetric {b:.3f}   "
          f"({'symmetry helps' if b < a else 'no benefit'}: {a/b if b>1e-9 else float('inf'):.2f}x)")
