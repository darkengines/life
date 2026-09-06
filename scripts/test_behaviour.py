"""What, if anything, do the brains actually do?

Foraging alignment came out at ~0 for both evolved and random brains, but
that may say more about the metric than the minds: with food abundant and
energy high, foraging is not what creatures are selected on. This measures
the behaviours that plausibly ARE under selection here -- fleeing threats,
chasing prey, approaching mates -- and does it separately for sighted and
blind individuals, since vision is now organ-gated and a blind animal
physically cannot be responding to anything it "sees".

Each score is a mean cosine in [-1, 1]: +1 = moving exactly the way the
behaviour would require, 0 = no relationship, -1 = exactly opposite.
"""
import math
import rust_world

W, POP_CAP, SEED, TICKS = 240, 6000, 11, 3000
VISION = 12.0


def body_size(ind):
    return len(ind["positions"]) * ind.get("size_scale", 1.0)


def measure(w):
    inds = w.individuals_state()
    if len(inds) < 20:
        return None
    # coarse spatial bucket so this is not O(n^2)
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

    scores = {k: [0.0, 0] for k in ("flee", "chase", "mate")}
    sighted_scores = {k: [0.0, 0] for k in ("flee", "chase", "mate")}

    for i in inds:
        x, y = i["positions"][0]
        my = body_size(i)
        hx, hy = math.cos(i["heading"]), math.sin(i["heading"])
        eyes = sum(1 for t in i.get("part_type", []) if t == 1)
        best = {"flee": None, "chase": None, "mate": None}
        bestd = {"flee": 1e9, "chase": 1e9, "mate": 1e9}
        for o in neighbours(i):
            if o is i:
                continue
            ox, oy = o["positions"][0]
            d = math.hypot(ox - x, oy - y)
            if d > VISION or d < 1e-6:
                continue
            osz = body_size(o)
            kind = None
            if osz > my * 1.15:
                kind = "flee"
            elif osz < my * 0.85:
                kind = "chase"
            if kind and d < bestd[kind]:
                bestd[kind], best[kind] = d, (ox - x, oy - y, d)
            if o.get("female") != i.get("female") and d < bestd["mate"]:
                bestd["mate"], best["mate"] = d, (ox - x, oy - y, d)
        for kind, v in best.items():
            if v is None:
                continue
            dx, dy, d = v
            ux, uy = dx / d, dy / d
            # fleeing wants the opposite direction; chasing and mating want it
            want = -1.0 if kind == "flee" else 1.0
            cos = (hx * ux + hy * uy) * want
            scores[kind][0] += cos
            scores[kind][1] += 1
            if eyes > 0:
                sighted_scores[kind][0] += cos
                sighted_scores[kind][1] += 1
    return scores, sighted_scores


def run(label, randomize):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, SEED, 50)
    w.spawn_random(300)
    for _ in range(TICKS):
        if randomize:
            w.debug_randomize_all_brains()
        w.tick()
    r = measure(w)
    if r is None:
        print(f"{label}: population collapsed")
        return None
    scores, sighted = r
    out = {}
    parts = []
    for k in ("flee", "chase", "mate"):
        tot, n = scores[k]
        st, sn = sighted[k]
        v = tot / n if n else float("nan")
        sv = st / sn if sn else float("nan")
        out[k] = v
        parts.append(f"{k}={v:+.4f}(n={n})  {k}_sighted={sv:+.4f}(n={sn})")
    print(f"{label}\n    " + "\n    ".join(parts), flush=True)
    return out


print("Behavioural alignment: evolved vs random brains\n")
ev = run("evolved brains", False)
rd = run("randomised brains", True)
if ev and rd:
    print("\ndifference (evolved - random):")
    for k in ev:
        print(f"    {k}: {ev[k] - rd[k]:+.4f}")
