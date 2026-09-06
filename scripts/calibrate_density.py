import rust_world
from collections import defaultdict

w = rust_world.World(240, 0.03, 1.0, 6000, 11, 50)
w.debug_set_pathogen_rate(0.0)   # measure UNDISTORTED conditions
w.spawn_random(300)
for _ in range(4000):
    w.tick()

inds = w.individuals_state()
dens = w.debug_conspecific_densities()
pop = len(inds)
assert len(dens) == pop, (len(dens), pop)

size = defaultdict(int)
for i in inds:
    size[tuple(i["color"])] += 1

big, rare = [], []
for i, d in zip(inds, dens):
    (big if size[tuple(i["color"])] >= pop * 0.05 else rare).append(float(d))

def stats(a, label):
    if not a:
        print(f"  {label}: n/a"); return
    a = sorted(a)
    q = lambda f: a[min(len(a)-1, int(len(a)*f))]
    print(f"  {label}: n={len(a)} mean={sum(a)/len(a):.2f} p10={q(.1):.2f} median={q(.5):.2f} p90={q(.9):.2f} p99={q(.99):.2f} max={a[-1]:.2f}")

print(f"pop={pop} lineages={len(size)} dominant={max(size.values())} ({100*max(size.values())/pop:.1f}%)")
stats(big,  "BIG lineage members (>=5% pop)")
stats(rare, "RARE lineage members (<5% pop)")

# What would each candidate curve charge these two groups?
def curve(d, thr, scale, cap):
    e = max(0.0, d - thr)
    return min((e/scale)**2, cap)
mean = lambda a: sum(a)/len(a) if a else 0.0
print("\ncandidate curves -> mean pressure (rare vs big, want rare~0 and big>>rare):")
for thr in (6.0, 8.0, 10.0, 12.0):
    for scale in (10.0, 14.0, 20.0):
        pr = mean([curve(d,thr,scale,2.5) for d in rare])
        pb = mean([curve(d,thr,scale,2.5) for d in big])
        ratio = (pb/pr) if pr > 1e-9 else float('inf')
        print(f"  thr={thr:>5} scale={scale:>5}: rare={pr:.3f} big={pb:.3f} ratio={ratio:.1f}")
