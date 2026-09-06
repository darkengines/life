"""Is the learned policy any good, or is the distillation just too weak?

The training loop demonstrably learns -- it predicts world dynamics several
times better than a do-nothing baseline and its policy loss falls steadily --
yet no behavioural benefit reproduces. Two candidate explanations:

  * newborns are only pulled 25% toward the learned policy, and mutation plus
    selection wash that out before it can matter;
  * the learned policy simply is not better than what evolution already does.

Pressing the policy in at full strength separates them. At rate 1.0 a newborn
IS the learned policy, so if that still changes nothing behaviourally, the
policy is not the missing ingredient and the distillation rate is a red
herring.
"""
import math
import sys
from pathlib import Path
import numpy as np
import rust_world

VAR = Path(__file__).resolve().parent.parent / "var"
WEIGHTS = VAR / "shared_encoder.npz"
W, POP_CAP, TICKS, WARMUP, EVERY = 240, 6000, 2600, 800, 100
VISION = 12.0
KINDS = ("flee", "chase", "mate")


def body_size(i):
    return len(i["positions"]) * i.get("size_scale", 1.0)


def sample(w):
    inds = w.individuals_state()
    if len(inds) < 20:
        return None
    grid = {}
    for i in inds:
        x, y = i["positions"][0]
        grid.setdefault((int(x // VISION), int(y // VISION)), []).append(i)
    acc = {k: [0.0, 0] for k in KINDS}
    for i in inds:
        if not any(t == 1 for t in i.get("part_type", [])):
            continue
        x, y = i["positions"][0]
        my = body_size(i)
        hx, hy = math.cos(i["heading"]), math.sin(i["heading"])
        cx, cy = int(x // VISION), int(y // VISION)
        best, bestd = {}, {k: 1e9 for k in KINDS}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for o in grid.get((cx + dx, cy + dy), []):
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
        for kind, (ddx, ddy, d) in best.items():
            want = -1.0 if kind == "flee" else 1.0
            acc[kind][0] += (hx * (ddx / d) + hy * (ddy / d)) * want
            acc[kind][1] += 1
    return acc


def run(rate, seed, use_policy):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
    if use_policy:
        d = np.load(WEIGHTS)
        w.set_shared_encoder(d["w"].astype(np.float32).tolist(), d["b"].astype(np.float32).tolist())
        w.set_shared_policy(d["pw1"].astype(np.float32).tolist(), d["pb1"].astype(np.float32).tolist(),
                            d["pw2"].astype(np.float32).tolist(), d["pb2"].astype(np.float32).tolist())
        w.debug_set_policy_distill_rate(rate)
    w.spawn_random(300)
    tot = {k: [0.0, 0] for k in KINDS}
    for t in range(1, TICKS + 1):
        w.tick()
        if t >= WARMUP and t % EVERY == 0:
            s = sample(w)
            if s:
                for k in KINDS:
                    tot[k][0] += s[k][0]
                    tot[k][1] += s[k][1]
    return tot, w.population()


if not WEIGHTS.exists():
    print("no trained weights"); sys.exit(1)

SEEDS = (11, 22, 33)
print(f"{'condition':>18} {'pop':>6} " + " ".join(f"{k:>16}" for k in KINDS))
for label, rate, use in (("evolution only", 0.0, False), ("distill 0.25", 0.25, True),
                         ("distill 0.60", 0.60, True), ("distill 1.00 (pure)", 1.0, True)):
    pooled = {k: [0.0, 0] for k in KINDS}
    pops = []
    for seed in SEEDS:
        tot, pop = run(rate, seed, use)
        pops.append(pop)
        for k in KINDS:
            pooled[k][0] += tot[k][0]
            pooled[k][1] += tot[k][1]
    cells = []
    for k in KINDS:
        v, n = pooled[k]
        cells.append(f"{(v/n if n else float('nan')):+.4f}(n={n})")
    print(f"{label:>18} {int(sum(pops)/len(pops)):>6} " + " ".join(f"{c:>16}" for c in cells), flush=True)
