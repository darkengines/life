"""Where does a tick actually go?

Optimising without this is guesswork, and guesswork is what has cost the most
time in this project. The engine already times every phase; this runs a world
up to a realistic size and reports the breakdown, so effort goes where the
milliseconds are rather than where they are assumed to be.
"""
import collections
import statistics
import sys
import time

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
WARMUP = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SAMPLE = 400

w = rust_world.World(W, 0.004, 1.0, POP_CAP, 11, PATCHES)
w.spawn_random(300)
t0 = time.time()
for _ in range(WARMUP):
    w.tick()
warm = time.time() - t0
inds = w.individuals_state()
parts = sum(len(i["positions"]) for i in inds)
print(f"warmup {WARMUP} ticks in {warm:.1f}s = {WARMUP/warm:.1f} ticks/s")
print(f"world: pop {len(inds)}, {parts} components, {len(w.corpses_state())} corpses\n")

acc = collections.defaultdict(list)
t0 = time.time()
for _ in range(SAMPLE):
    w.tick()
    for k, v in w.timings().items():
        acc[k].append(v)
elapsed = time.time() - t0
rate = SAMPLE / elapsed
total_ms = sum(statistics.mean(v) for v in acc.values())
print(f"{SAMPLE} ticks at {rate:.1f} ticks/s ({1000/rate:.2f} ms/tick wall)\n")
print(f"{'phase':<26} {'ms/tick':>9} {'% of measured':>14}")
for k, v in sorted(acc.items(), key=lambda t: -statistics.mean(t[1])):
    m = statistics.mean(v)
    if m < 0.005:
        continue
    print(f"{k:<26} {m:>9.3f} {100*m/max(1e-9,total_ms):>13.1f}%")
print(f"{'(measured total)':<26} {total_ms:>9.3f}")
