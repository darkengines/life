"""Does the GPU-trained perception encoder actually improve behaviour?

The architecture splits perception (shared, trainable) from decision (private,
evolved). The trainer learns an encoder that predicts the world several times
better than a "nothing changes" baseline -- but predictive quality is not the
point. The point is whether creatures perceiving through it behave better.

Same seeds, same everything, one variable: the shared encoder is either the
world's random initialisation or the trained weights.

MEASUREMENT NOTE: a first attempt scored only the final frame, giving ~100
sighted samples whose standard error (~0.11) was larger than the effects being
measured -- the result was uninterpretable. This samples throughout each run
and pools across seeds, and prints an explicit noise band so a difference is
never read as real when it isn't.
"""
import math
import sys
from pathlib import Path

import numpy as np
import rust_world

VAR = Path(__file__).resolve().parent.parent / "var"
WEIGHTS = VAR / "shared_encoder.npz"
W, POP_CAP, TICKS = 240, 6000, 3000
VISION = 12.0
SEEDS = (11, 22, 33, 44)
WARMUP = 800
SAMPLE_EVERY = 100
KINDS = ("flee", "chase", "mate")


def body_size(i):
    return len(i["positions"]) * i.get("size_scale", 1.0)


def sample_alignment(w):
    """Mean cosine between heading and each behaviour's ideal direction.

    Scored for SIGHTED individuals only: a blind creature cannot be
    responding to something it sees, so including them only adds noise.
    """
    inds = w.individuals_state()
    if len(inds) < 20:
        return None
    grid = {}
    for i in inds:
        x, y = i["positions"][0]
        grid.setdefault((int(x // VISION), int(y // VISION)), []).append(i)

    def neighbours(i):
        x, y = i["positions"][0]
        cx, cy = int(x // VISION), int(y // VISION)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out += grid.get((cx + dx, cy + dy), [])
        return out

    acc = {k: [0.0, 0] for k in KINDS}
    for i in inds:
        if not any(t == 1 for t in i.get("part_type", [])):
            continue  # blind
        x, y = i["positions"][0]
        my = body_size(i)
        hx, hy = math.cos(i["heading"]), math.sin(i["heading"])
        best, bestd = {}, {k: 1e9 for k in KINDS}
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
        for kind, (dx, dy, d) in best.items():
            want = -1.0 if kind == "flee" else 1.0
            acc[kind][0] += (hx * (dx / d) + hy * (dy / d)) * want
            acc[kind][1] += 1
    return acc


def run_seed(seed, use_trained):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
    if use_trained:
        d = np.load(WEIGHTS)
        if not w.set_shared_encoder(d["w"].astype(np.float32).tolist(),
                                    d["b"].astype(np.float32).tolist()):
            print("trained weights rejected (shape mismatch)")
            sys.exit(1)
    w.spawn_random(300)
    totals = {k: [0.0, 0] for k in KINDS}
    for t in range(1, TICKS + 1):
        w.tick()
        if t >= WARMUP and t % SAMPLE_EVERY == 0:
            s = sample_alignment(w)
            if s is None:
                continue
            for k in KINDS:
                totals[k][0] += s[k][0]
                totals[k][1] += s[k][1]
    return totals


if not WEIGHTS.exists():
    print("no trained weights on disk yet -- run app/train_encoder.py first")
    sys.exit(1)

print("Random vs GPU-trained shared perception encoder")
print(f"seeds={SEEDS}, sampled every {SAMPLE_EVERY} ticks after tick {WARMUP}\n")

results = {}
for cond, use_trained in (("random", False), ("trained", True)):
    pooled = {k: [0.0, 0] for k in KINDS}
    for seed in SEEDS:
        t = run_seed(seed, use_trained)
        for k in KINDS:
            pooled[k][0] += t[k][0]
            pooled[k][1] += t[k][1]
        print(f"  {cond:>7} seed {seed}: " + "  ".join(
            f"{k}={(t[k][0]/t[k][1] if t[k][1] else float('nan')):+.4f}(n={t[k][1]})"
            for k in KINDS), flush=True)
    results[cond] = pooled
    print()

print("pooled across seeds:")
for k in KINDS:
    rs, rn = results["random"][k]
    ts, tn = results["trained"][k]
    rm = rs / rn if rn else float("nan")
    tm = ts / tn if tn else float("nan")
    # 2 sigma on the difference of two mean cosines (unit-variance worst case)
    band = 2.0 * math.sqrt(1.0 / max(1, rn) + 1.0 / max(1, tn))
    diff = tm - rm
    verdict = "MEANINGFUL" if abs(diff) > band else "within noise"
    print(f"  {k:>5}: random {rm:+.4f} (n={rn:>6})   trained {tm:+.4f} (n={tn:>6})   "
          f"diff {diff:+.4f} +/-{band:.4f}  -> {verdict}")
