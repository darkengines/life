"""Does the GPU-trained perception encoder actually improve behaviour?

The architecture splits perception (shared, trainable) from decision (private,
evolved). The trainer has learned an encoder that predicts the world 5x better
than a "nothing changes" baseline -- but predictive quality is not the point.
The point is whether creatures that perceive through it behave better.

Same seed, same everything, one variable: the shared encoder is either the
world's random initialisation or the trained weights. Behaviour is scored the
same way as test_behaviour.py -- alignment with fleeing, chasing and mating --
so the two are directly comparable.
"""
import math
import sys
from pathlib import Path

import numpy as np
import rust_world

VAR = Path(__file__).resolve().parent.parent / "var"
WEIGHTS = VAR / "shared_encoder.npz"
W, POP_CAP, SEED, TICKS = 240, 6000, 11, 3000
VISION = 12.0


def body_size(i):
    return len(i["positions"]) * i.get("size_scale", 1.0)


def measure(w):
    inds = w.individuals_state()
    if len(inds) < 20:
        return None
    cell = VISION
    grid = {}
    for i in inds:
        x, y = i["positions"][0]
        grid.setdefault((int(x // cell), int(y // cell)), []).append(i)

    def neighbours(i):
        x, y = i["positions"][0]
        cx, cy = int(x // cell), int(y // cell)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out += grid.get((cx + dx, cy + dy), [])
        return out

    acc = {k: [0.0, 0] for k in ("flee", "chase", "mate")}
    sighted = {k: [0.0, 0] for k in ("flee", "chase", "mate")}
    for i in inds:
        x, y = i["positions"][0]
        my = body_size(i)
        hx, hy = math.cos(i["heading"]), math.sin(i["heading"])
        eyes = sum(1 for t in i.get("part_type", []) if t == 1)
        best, bestd = {}, {"flee": 1e9, "chase": 1e9, "mate": 1e9}
        for o in neighbours(i):
            if o is i:
                continue
            ox, oy = o["positions"][0]
            d = math.hypot(ox - x, oy - y)
            if d > VISION or d < 1e-6:
                continue
            osz = body_size(o)
            kind = "flee" if osz > my * 1.15 else ("chase" if osz < my * 0.85 else None)
            if kind and d < bestd[kind]:
                bestd[kind], best[kind] = d, (ox - x, oy - y, d)
            if o.get("female") != i.get("female") and d < bestd["mate"]:
                bestd["mate"], best["mate"] = d, (ox - x, oy - y, d)
        for kind, v in best.items():
            dx, dy, d = v
            want = -1.0 if kind == "flee" else 1.0
            cos = (hx * (dx / d) + hy * (dy / d)) * want
            acc[kind][0] += cos
            acc[kind][1] += 1
            if eyes > 0:
                sighted[kind][0] += cos
                sighted[kind][1] += 1
    return acc, sighted, len(inds)


def run(label, use_trained):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, SEED, 50)
    if use_trained:
        if not WEIGHTS.exists():
            print("no trained weights on disk yet"); sys.exit(1)
        d = np.load(WEIGHTS)
        ok = w.set_shared_encoder(d["w"].astype(np.float32).tolist(),
                                  d["b"].astype(np.float32).tolist())
        if not ok:
            print("trained weights rejected (shape mismatch)"); sys.exit(1)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    r = measure(w)
    if r is None:
        print(f"{label}: collapsed")
        return None
    acc, sighted, pop = r
    out = {}
    line = []
    for k in ("flee", "chase", "mate"):
        t, n = acc[k]
        st, sn = sighted[k]
        out[k] = t / n if n else float("nan")
        out[k + "_sighted"] = st / sn if sn else float("nan")
        line.append(f"{k}={out[k]:+.4f} (sighted {out[k+'_sighted']:+.4f}, n={sn})")
    print(f"{label}: pop={pop}\n    " + "\n    ".join(line), flush=True)
    return out


print("Random vs GPU-trained shared perception encoder\n")
rnd = run("random encoder ", False)
trn = run("trained encoder", True)
if rnd and trn:
    print("\ndifference (trained - random):")
    for k in ("flee", "chase", "mate"):
        print(f"    {k:>5}: {trn[k]-rnd[k]:+.4f}   sighted: "
              f"{trn[k+'_sighted']-rnd[k+'_sighted']:+.4f}")
