"""Why do bodies stay tiny and behaviour stay simple?

Every child is born with parent_count + 1 pixels (grow_one_pixel is called
unconditionally at birth), so body size should ratchet upward every
generation. It does not: live runs sit at ~2.7 mean pixels after tens of
thousands of births. Something is removing that complexity as fast as birth
adds it. This measures what.
"""
import collections
import rust_world

W, POP_CAP, SEED = 240, 6000, 11


def snapshot(w):
    inds = w.individuals_state()
    if not inds:
        return None
    counts = [len(i["positions"]) for i in inds]
    hist = collections.Counter(counts)
    n = len(counts)
    return dict(
        pop=n,
        mean=sum(counts) / n,
        biggest=max(counts),
        hist=hist,
        # what fraction of the population is a 1-2 pixel blob?
        blob_frac=sum(v for k, v in hist.items() if k <= 2) / n,
        mean_age=sum(i["age"] for i in inds) / n,
        mean_energy=sum(i["energy"] for i in inds) / n,
        captured=sum(1 for i in inds if i.get("captured")) / n,
        parts=part_mix(inds),
    )


PART_NAMES = ["body", "eye", "mouth", "gut", "tentacle", "armor", "flipper"]


def part_mix(inds):
    """What is the population actually built out of? If organs never appear,
    or one organ dominates everything, differentiation is not working."""
    tally = collections.Counter()
    for i in inds:
        for t in i.get("part_type", []):
            tally[t] += 1
    total = sum(tally.values()) or 1
    return {PART_NAMES[k]: tally.get(k, 0) / total for k in range(len(PART_NAMES))}


w = rust_world.World(W, 0.03, 1.0, POP_CAP, SEED, 50)
w.spawn_random(300)

print(f"{'tick':>6} {'pop':>6} {'mean_px':>8} {'max_px':>7} {'<=2px':>7} {'mean_age':>9} {'energy':>7} {'held':>6}")
for tick in range(1, 6001):
    w.tick()
    if tick % 500 == 0:
        s = snapshot(w)
        if s is None:
            print(f"{tick:>6}  EXTINCT")
            break
        mix = " ".join(f"{k[:4]}={v*100:.0f}%" for k, v in s["parts"].items() if v > 0.005)
        print(f"{tick:>6} {s['pop']:>6} {s['mean']:>8.2f} {s['biggest']:>7} "
              f"{s['blob_frac']*100:>6.1f}% {s['mean_age']:>9.0f} {s['mean_energy']:>7.1f} "
              f"{s['captured']*100:>5.1f}%  {mix}")

s = snapshot(w)
if s:
    print("\nfinal body-size histogram (pixels -> count):")
    for k in sorted(s["hist"]):
        print(f"  {k:>3}px: {s['hist'][k]:>5}  {'#' * min(60, s['hist'][k] * 60 // s['pop'])}")

    ev = w.events()
    print(f"\nevents: {ev}")
    print(f"births per death: {ev['reproductions'] / max(1, ev['deaths']):.2f}")
    print(f"fights per birth: {ev['fights'] / max(1, ev['reproductions']):.2f}")
