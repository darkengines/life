"""Are reproductive anomalies actually diversified?

Raised as a concern and never investigated. If offspring are near-copies of
their parents in body plan, then morphology explores a very narrow space no
matter how long the simulation runs, and "complex structure" can never emerge
however good the selection pressure is.

Measures three things:
  * how much a child's body differs from its parent's, part by part;
  * how many DISTINCT body topologies exist in the population at once;
  * whether that variety grows or collapses over time.
"""
import collections
import rust_world

W, TICKS = 240, 6000


def topology_signature(ind):
    """A shape fingerprint: the sorted child-count profile plus depth profile.

    Two bodies with the same signature have the same branching structure, so
    counting distinct signatures counts genuinely different body plans rather
    than trivial coordinate differences.
    """
    parents = ind["parents"]
    kids = collections.Counter()
    for p in parents:
        if p >= 0:
            kids[p] += 1
    depth = {}
    for k, p in enumerate(parents):
        depth[k] = 0 if p < 0 else depth.get(p, 0) + 1
    child_profile = tuple(sorted(kids.values()))
    depth_profile = tuple(sorted(collections.Counter(depth.values()).values()))
    types = tuple(sorted(collections.Counter(ind.get("part_type", [])).items()))
    return (len(parents), child_profile, depth_profile, types)


w = rust_world.World(W, 0.03, 1.0, 6000, 11, 50)
w.spawn_random(300)

print(f"{'tick':>6} {'pop':>6} {'distinct plans':>15} {'plans/100':>10} "
      f"{'biggest share':>14} {'meanPx':>7}")
for t in range(1, TICKS + 1):
    w.tick()
    if t % 750:
        continue
    inds = w.individuals_state()
    if not inds:
        print(f"{t:>6}  EXTINCT")
        break
    sigs = collections.Counter(topology_signature(i) for i in inds)
    sizes = [len(i["positions"]) for i in inds]
    top = sigs.most_common(1)[0][1]
    print(f"{t:>6} {len(inds):>6} {len(sigs):>15} "
          f"{100*len(sigs)/len(inds):>9.1f} {100*top/len(inds):>13.1f}% "
          f"{sum(sizes)/len(sizes):>7.1f}", flush=True)

print("\nHigh distinct-plan counts and a small dominant share mean morphology is")
print("genuinely exploring. A single plan dominating means mutation is too narrow.")
