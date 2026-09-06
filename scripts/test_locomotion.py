"""Do creatures swim forwards, and are their body plans anything but worms?

Two reported problems, both checkable.

SWIMMING DIRECTION. Undulatory swimmers generate thrust by passing a bending
wave from head to TAIL: the wave pushes water backwards, the body goes
forwards. If the wave travels the other way, the animal swims backwards --
tail first. The engine's wave is sin(w*t + phase + 0.7k), whose phase
INCREASES with segment index k, which is a wave travelling from tail to head.
This measures the consequence directly: the angle between where a creature
points (heading) and where it actually moves (velocity). Near 0 means it
swims forwards; near 180 means it is swimming tail-first.

BODY PLANS. Growth weights "extend an existing tip in the same direction" at
8x everything else, which is a strong bias toward unbranched chains. This
measures the actual topology: how many parts have more than one child
(branch points), and how deep bodies are relative to their size. A pure worm
has zero branch points and depth == part count.
"""
import math
import rust_world

W, TICKS = 240, 2500
w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)
for _ in range(TICKS):
    w.tick()

inds = w.individuals_state()
print(f"population {len(inds)}\n")

# --- swimming direction -------------------------------------------------
angles = []
for i in inds:
    vx, vy = i.get("velocity", (None, None)) if isinstance(i.get("velocity"), (list, tuple)) else (None, None)
    if vx is None:
        # velocity isn't published; reconstruct from speed+heading is circular,
        # so use consecutive positions instead (below).
        break
if not angles:
    before = {i["id"]: i["positions"][0] for i in inds}
    headings = {i["id"]: i["heading"] for i in inds}
    for _ in range(6):
        w.tick()
    after = {i["id"]: i["positions"][0] for i in w.individuals_state()}
    for pid, p0 in before.items():
        p1 = after.get(pid)
        if p1 is None:
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 0.05:          # too slow to have a meaningful direction
            continue
        h = headings[pid]
        cos = (math.cos(h) * dx + math.sin(h) * dy) / dist
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))

if angles:
    angles.sort()
    mean = sum(angles) / len(angles)
    backwards = sum(1 for a in angles if a > 90)
    print("SWIMMING DIRECTION (angle between heading and actual movement)")
    print(f"  n={len(angles)}  mean={mean:.1f} deg  median={angles[len(angles)//2]:.1f} deg")
    print(f"  moving BACKWARDS (>90 deg): {backwards} ({100*backwards/len(angles):.1f}%)")
    print("  0 deg = swims forwards, 180 deg = swims tail-first\n")

# --- body plan topology -------------------------------------------------
branch_points = 0
total_parts = 0
worms = 0
max_children = 0
depth_ratio = []
for i in inds:
    parents = i["parents"]
    n = len(parents)
    total_parts += n
    kids = {}
    for k, p in enumerate(parents):
        if p >= 0:
            kids[p] = kids.get(p, 0) + 1
    branches = sum(1 for c in kids.values() if c > 1)
    branch_points += branches
    if kids:
        max_children = max(max_children, max(kids.values()))
    if branches == 0:
        worms += 1
    # depth of the deepest chain
    depth = {}
    for k, p in enumerate(parents):
        depth[k] = 1 if p < 0 else depth.get(p, 1) + 1
    if n > 1:
        depth_ratio.append(max(depth.values()) / n)

print("BODY PLAN TOPOLOGY")
print(f"  bodies that are pure unbranched chains: {worms}/{len(inds)} "
      f"({100*worms/max(1,len(inds)):.1f}%)")
print(f"  branch points per body: {branch_points/max(1,len(inds)):.2f}")
print(f"  most children on one node: {max_children}")
if depth_ratio:
    print(f"  mean depth/parts ratio: {sum(depth_ratio)/len(depth_ratio):.2f}  "
          f"(1.00 = perfect worm, lower = bushier)")
