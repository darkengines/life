"""Does the symmetry property actually produce mirrored pairs?

A symmetric node should emit twinned parts: same parent, same kind, opposite
rest angle. If bilateral bodies are working, a good fraction of the
population should show such pairs, and they should be structural (the twins
share a parent) rather than coincidental.
"""
import collections
import math
import rust_world

w = rust_world.World(240, 0.03, 1.0, 4000, 11, 50)
w.spawn_random(300)
for _ in range(2500):
    w.tick()

inds = w.individuals_state()
paired_bodies = 0
total_pairs = 0
pair_kinds = collections.Counter()
for i in inds:
    parents, angles = i["parents"], i["rest_angle"]
    types = i.get("part_type", [])
    by_parent = collections.defaultdict(list)
    for k in range(len(parents)):
        if parents[k] >= 0:
            by_parent[parents[k]].append(k)
    found = 0
    for _p, kids in by_parent.items():
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                ka, kb = kids[a], kids[b]
                # mirrored: equal and opposite rest angle, same part kind
                if abs(angles[ka] + angles[kb]) < 1e-3 and abs(angles[ka]) > 1e-3:
                    if not types or types[ka] == types[kb]:
                        found += 1
                        if types:
                            pair_kinds[types[ka]] += 1
    if found:
        paired_bodies += 1
        total_pairs += found

NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper"]
pop = len(inds)
print(f"population {pop}")
print(f"bodies containing at least one mirrored pair: {paired_bodies} ({100*paired_bodies/max(1,pop):.1f}%)")
print(f"total mirrored pairs: {total_pairs}")
print("paired organ kinds:", {NAMES[k]: v for k, v in sorted(pair_kinds.items())})
sizes = [len(i["positions"]) for i in inds]
print(f"mean body {sum(sizes)/max(1,len(sizes)):.1f} parts, max {max(sizes) if sizes else 0}")
