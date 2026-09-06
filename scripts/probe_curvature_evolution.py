"""Does evolution straighten these creatures out?

Tracking individual survival proved far too noisy to answer whether swimming
straight pays -- survival is single-digit percent, so the survivors are a tiny
biased sample and the same settings gave -41 and +50 energy gain on two runs.
Evolution's own verdict is a much cleaner instrument: if holding a course
earns anything, mean posture curvature should fall over generations, and if
circling pays it should rise.

Run long, sample often, and watch the trend.
"""
import rust_world

W, POP_CAP, TICKS = 240, 6000, 9000
w = rust_world.World(W, 0.03, 1.0, POP_CAP, 11, 50)
w.spawn_random(300)

print(f"{'tick':>6} {'pop':>6} {'mean|curv|':>11} {'straight%':>10} {'meanPx':>7} {'energy':>8}")
for t in range(1, TICKS + 1):
    w.tick()
    if t % 750:
        continue
    inds = w.individuals_state()
    if not inds:
        print(f"{t:>6}  EXTINCT")
        break
    curv = [abs(i.get("turn_curvature", 0.0)) for i in inds]
    sizes = [len(i["positions"]) for i in inds]
    energy = [i["energy"] for i in inds]
    straight = 100 * sum(1 for c in curv if c < 0.2) / len(curv)
    print(f"{t:>6} {len(inds):>6} {sum(curv)/len(curv):>11.3f} {straight:>9.0f}% "
          f"{sum(sizes)/len(sizes):>7.1f} {sum(energy)/len(energy):>8.1f}", flush=True)
print("\nFalling mean curvature = selection favours holding a course.")
print("Flat or rising = circling pays, and directed travel never will.")
