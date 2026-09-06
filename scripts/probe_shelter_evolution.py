"""Now that shelter can be PERCEIVED, do creatures evolve to use it?

Sheltering pays -- small bodies survive 42% inside the deep reef against 20%
for large ones -- but until now creatures had no terrain sense at all and
discovered rock only by colliding with it, so the payoff was unreachable. With
a shelter channel added (gated on having eyes, like the other spatial senses),
selection finally has something to act on.

If it is working, small creatures should be found in progressively more
enclosed places over generations, and the gap between small and large should
widen. Flat lines mean the sense is present but unused.
"""
import rust_world

W, TICKS, R = 240, 9000, 4


def enclosure_map(terrain, size, r=R):
    pref = [[0] * (size + 1) for _ in range(size + 1)]
    for x in range(size):
        row = 0
        for y in range(size):
            row += 1 if terrain[x][y] != 0 else 0
            pref[x + 1][y + 1] = pref[x][y + 1] + row
    out = [[0.0] * size for _ in range(size)]
    for x in range(size):
        x0, x1 = max(0, x - r), min(size, x + r + 1)
        for y in range(size):
            y0, y1 = max(0, y - r), min(size, y + r + 1)
            s = pref[x1][y1] - pref[x0][y1] - pref[x1][y0] + pref[x0][y0]
            out[x][y] = s / ((x1 - x0) * (y1 - y0))
    return out


w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)
enc = enclosure_map(w.terrain_grid().tolist(), W)

print(f"{'tick':>6} {'pop':>6} {'small enc':>10} {'large enc':>10} {'gap':>7} {'sighted%':>9}")
for t in range(1, TICKS + 1):
    w.tick()
    if t % 900:
        continue
    inds = w.individuals_state()
    if not inds:
        print(f"{t:>6}  EXTINCT")
        break
    small, large = [], []
    sighted = 0
    for i in inds:
        x, y = i["positions"][0]
        gx, gy = int(min(W - 1, max(0, x))), int(min(W - 1, max(0, y)))
        e = enc[gx][gy]
        (small if len(i["positions"]) <= 6 else large).append(e)
        if any(t2 == 1 for t2 in i.get("part_type", [])):
            sighted += 1
    ms = sum(small) / len(small) if small else float("nan")
    ml = sum(large) / len(large) if large else float("nan")
    print(f"{t:>6} {len(inds):>6} {ms:>10.4f} {ml:>10.4f} {ms-ml:>7.4f} "
          f"{100*sighted/len(inds):>8.0f}%", flush=True)
print("\nRising 'small enc' and a widening gap => creatures are learning to shelter.")
