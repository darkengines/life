"""Is the reef actually safer than open water?

Location was shown to have no fitness consequence, which is the ceiling on
navigation intelligence. But the reef should already create one: large
predators physically cannot enter narrow passages, so a small creature inside
should be harder to eat. If that differential exists, seeking shelter is worth
evolving and vision has something to earn. If it does not, the refuge is
geometry without consequence.

Tracks where individuals are, then who is still alive later, bucketed by how
enclosed their surroundings are.
"""
import rust_world

W, WARMUP, WINDOW, R = 240, 2000, 400, 4


def enclosure_map(terrain, size, r=R):
    pref = [[0] * (size + 1) for _ in range(size + 1)]
    for x in range(size):
        row = 0
        for y in range(size):
            row += 1 if terrain[x][y] == 1 else 0
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
for _ in range(WARMUP):
    w.tick()
enc = enclosure_map(w.terrain_grid().tolist(), W)

tracked = {}
for i in w.individuals_state():
    x, y = i["positions"][0]
    gx, gy = int(min(W - 1, max(0, x))), int(min(W - 1, max(0, y)))
    tracked[i["id"]] = (enc[gx][gy], len(i["positions"]))

for _ in range(WINDOW):
    w.tick()
alive = {i["id"] for i in w.individuals_state()}

print(f"tracked {len(tracked)} individuals over {WINDOW} ticks\n")
print(f"{'enclosure':>14} {'n':>5} {'survived':>9}   {'small(<=6) surv':>16} {'large(>6) surv':>15}")
for lo, hi in ((0.0, 0.02), (0.02, 0.08), (0.08, 0.18), (0.18, 1.01)):
    grp = [(p, e, n) for p, (e, n) in tracked.items() if lo <= e < hi]
    if not grp:
        continue
    sm = [p for p, _, n in grp if n <= 6]
    lg = [p for p, _, n in grp if n > 6]
    def pct(ids):
        return f"{100*sum(1 for p in ids if p in alive)/len(ids):.0f}% (n={len(ids)})" if ids else "n/a"
    print(f"{lo:>6.2f}-{hi:<7.2f} {len(grp):>5} "
          f"{100*sum(1 for p,_,_ in grp if p in alive)/len(grp):>8.0f}%   "
          f"{pct(sm):>16} {pct(lg):>15}")
print("\nIf survival rises with enclosure -- especially for small bodies --")
print("then sheltering pays and navigating to it is worth evolving.")
