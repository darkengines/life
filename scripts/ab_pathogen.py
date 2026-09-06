import rust_world

def run(seed, rate, ticks=4000):
    w = rust_world.World(240, 0.03, 1.0, 6000, seed, 50)
    w.debug_set_pathogen_rate(rate)
    w.spawn_random(300)
    for _ in range(ticks):
        w.tick()
    sp = w.species_summary(); pop = w.population()
    if pop == 0 or not sp:
        return dict(pop=0, lin=0, dom=0.0, top5=0.0, eff=0.0, res=0.0)
    c = [r["count"] for r in sp]
    # inverse Simpson = effective number of EVENLY-abundant species.
    # Raw lineage count is confounded (fewer individuals -> fewer colors);
    # this measures actual evenness, which is what "diverse" should mean.
    eff = 1.0 / sum((x/pop)**2 for x in c)
    res = sum(r["avg_disease_resistance"]*r["count"] for r in sp)/pop
    return dict(pop=pop, lin=len(sp), dom=c[0]/pop, top5=sum(c[:5])/pop, eff=eff, res=res)

print(f"{'seed':>4} {'rate':>5} {'pop':>5} {'lineages':>8} {'effective':>9} {'dom%':>6} {'top5%':>6} {'resist':>7}")
for seed in (11, 22, 33):
    for rate in (0.0, 0.02, 0.035):
        r = run(seed, rate)
        print(f"{seed:>4} {rate:>5.3f} {r['pop']:>5} {r['lin']:>8} {r['eff']:>9.1f} "
              f"{r['dom']*100:>5.1f}% {r['top5']*100:>5.1f}% {r['res']:>7.3f}", flush=True)
