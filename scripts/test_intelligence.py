"""Is the brain actually doing anything?

Before making the network bigger, establish whether intelligence contributes
at all. A bigger network is not automatically smarter: with mutation-only
neuroevolution, more parameters can mean slower adaptation, so "brains are
too small" is a hypothesis to test rather than assume.

The test is a behavioural one that does not depend on any notion of
"fitness": does an individual's movement correlate with the food gradient it
can smell? A brain that has learned to forage should show a positive
correlation. A brain that ignores its senses should score ~0, which is what
a random-weight control gives. Comparing evolved against freshly randomised
brains in the SAME world state separates "the network can't learn" from "the
world doesn't reward learning".
"""
import math
import rust_world

W, POP_CAP, SEED = 240, 6000, 11


def forage_alignment(w, sample=800):
    """Mean cosine between heading and the local food gradient.

    +1 = swimming straight up-gradient toward food, 0 = no relationship,
    -1 = consistently swimming away from it.
    """
    inds = w.individuals_state()
    if not inds:
        return None, 0
    food = w.food()
    n = food.shape[0]
    step = max(1, len(inds) // sample)
    total, counted = 0.0, 0
    for i in inds[::step]:
        x, y = i["positions"][0]
        gx, gy = int(x), int(y)
        if not (1 <= gx < n - 1 and 1 <= gy < n - 1):
            continue
        # central difference at a range the creatures can actually smell
        r = 6
        x0, x1 = max(0, gx - r), min(n - 1, gx + r)
        y0, y1 = max(0, gy - r), min(n - 1, gy + r)
        dfx = float(food[x1][gy] - food[x0][gy])
        dfy = float(food[gx][y1] - food[gx][y0])
        gmag = math.hypot(dfx, dfy)
        if gmag < 1e-6:
            continue
        h = i["heading"]
        hx, hy = math.cos(h), math.sin(h)
        total += (hx * dfx + hy * dfy) / gmag
        counted += 1
    return (total / counted if counted else None), counted


def run(label, randomize_brains, ticks=3000):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, SEED, 50)
    w.spawn_random(300)
    for t in range(1, ticks + 1):
        if randomize_brains:
            # Wipe learned weights constantly, so behaviour can never be
            # anything but reflexes from random weights.
            w.debug_randomize_all_brains()
        w.tick()
    align, n = forage_alignment(w)
    inds = w.individuals_state()
    pop = len(inds)
    mean_age = sum(i["age"] for i in inds) / pop if pop else 0
    mean_energy = sum(i["energy"] for i in inds) / pop if pop else 0
    print(f"{label:>18}  pop={pop:>5}  mean_age={mean_age:>6.0f}  "
          f"mean_energy={mean_energy:>6.1f}  forage_alignment="
          f"{align if align is None else round(align, 4)}  (n={n})", flush=True)
    return align


print("Does evolved behaviour beat random behaviour?\n")
evolved = run("evolved brains", False)
control = run("randomised brains", True)
if evolved is not None and control is not None:
    print(f"\nevolved - random = {evolved - control:+.4f}")
    print("If this is ~0, the brains are not contributing and the problem is")
    print("selection pressure or sensing, not network capacity.")
