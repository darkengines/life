import time
import rust_world

w = rust_world.World(240, 0.03, 1.0, 6000, 7, 50)
w.spawn_random(300)

# Warm up / grow population toward the cap.
for _ in range(1500):
    w.tick()

print("population after warmup:", w.population())

N = 200
t0 = time.perf_counter()
for _ in range(N):
    w.tick()
elapsed = time.perf_counter() - t0
print(f"population at end: {w.population()}")
print(f"{N} ticks in {elapsed:.3f}s -> {elapsed/N*1000:.2f} ms/tick, {N/elapsed:.1f} ticks/sec")

timings = w.timings()
for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v:.3f} ms")
