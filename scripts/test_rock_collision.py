"""Is rock actually solid?

Refugia depend entirely on this: if bodies can sit inside or pass through
rock, narrow passages stop being narrow and the reef is decorative. The
collision check reverts a move that would put any part in rock -- but it
deliberately lets an individual ALREADY inside rock keep moving, so that
being stuck is never permanent. That escape hatch is only safe if creatures
essentially never end up inside rock in the first place.
"""
import rust_world

W, TICKS = 240, 3000
w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)
terrain = w.terrain_grid().tolist()

def scan(label):
    inds = w.individuals_state()
    if not inds:
        print(f"{label}: no individuals"); return
    parts_in_rock = 0
    total_parts = 0
    bodies_touching_rock = 0
    for i in inds:
        touched = False
        for (x, y) in i["positions"]:
            gx = min(W - 1, max(0, int(x)))
            gy = min(W - 1, max(0, int(y)))
            total_parts += 1
            if terrain[gx][gy] == 1:
                parts_in_rock += 1
                touched = True
        if touched:
            bodies_touching_rock += 1
    print(f"{label}: pop={len(inds)}  parts inside rock: {parts_in_rock}/{total_parts} "
          f"({100*parts_in_rock/max(1,total_parts):.2f}%)  bodies touching rock: "
          f"{bodies_touching_rock} ({100*bodies_touching_rock/len(inds):.1f}%)")

for t in range(1, TICKS + 1):
    w.tick()
    if t in (200, 1000, 2000, 3000):
        scan(f"tick {t:>5}")
