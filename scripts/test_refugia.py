"""Are the reef refuges actually doing anything?

The reef exists so small creatures can occupy places large ones physically
cannot follow into -- the textbook mechanism that lets predator and prey
coexist instead of the predator eating everything. Nothing in the engine
grants small creatures protection; the passages are simply narrow and
collision is per-pixel. So the question is empirical: do small and large
creatures actually end up in different places?

"Enclosure" is how much rock surrounds a position. If refugia work, small
creatures should be found in more enclosed spots than large ones, and their
survival there should be better than out in the open.
"""
import collections
import rust_world

W, POP_CAP, SEED, TICKS = 240, 6000, 11, 4000
SAMPLE_R = 4          # radius over which to count surrounding rock


def enclosure_map(terrain, size, r=SAMPLE_R):
    """For each cell, the fraction of nearby cells that are rock."""
    rock = [[1 if terrain[x][y] == 1 else 0 for y in range(size)] for x in range(size)]
    # separable box sum via prefix sums, so this stays cheap at 240x240
    pref = [[0] * (size + 1) for _ in range(size + 1)]
    for x in range(size):
        row = 0
        for y in range(size):
            row += rock[x][y]
            pref[x + 1][y + 1] = pref[x][y + 1] + row
    def box(x0, y0, x1, y1):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(size, x1), min(size, y1)
        if x0 >= x1 or y0 >= y1:
            return 0, 1
        return (pref[x1][y1] - pref[x0][y1] - pref[x1][y0] + pref[x0][y0],
                (x1 - x0) * (y1 - y0))
    out = [[0.0] * size for _ in range(size)]
    for x in range(size):
        for y in range(size):
            s, n = box(x - r, y - r, x + r + 1, y + r + 1)
            out[x][y] = s / n
    return out


w = rust_world.World(W, 0.03, 1.0, POP_CAP, SEED, 50)
w.spawn_random(300)
terrain = w.terrain_grid().tolist()
enc = enclosure_map(terrain, W)

buckets = collections.defaultdict(lambda: [0.0, 0])
for t in range(1, TICKS + 1):
    w.tick()
    if t < 1500 or t % 250:
        continue
    for i in w.individuals_state():
        x, y = i["positions"][0]
        gx, gy = int(min(W - 1, max(0, x))), int(min(W - 1, max(0, y)))
        parts = len(i["positions"])
        b = "small (<=4)" if parts <= 4 else ("medium (5-12)" if parts <= 12 else "large (>12)")
        buckets[b][0] += enc[gx][gy]
        buckets[b][1] += 1

print(f"world {W}x{W}, rock cells "
      f"{sum(1 for row in terrain for c in row if c == 1)}")
print("\nmean enclosure (fraction of rock within "
      f"{SAMPLE_R} cells) of theeach size class occupies:\n")
order = ["small (<=4)", "medium (5-12)", "large (>12)"]
for b in order:
    tot, n = buckets[b]
    if n:
        print(f"  {b:>15}: {tot/n:.4f}   (n={n})")
    else:
        print(f"  {b:>15}: none present")
print("\nIf refugia work, small should sit in measurably more enclosed places")
print("than large. If the numbers match, the reef is decorative.")
