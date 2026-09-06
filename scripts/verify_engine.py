import rust_world
import numpy as np

w = rust_world.World(64, 0.03, 1.0, 200, 42, 10)
w.spawn_random(80)

for _ in range(300):
    w.tick()

terr = w.territory()
print("territory field: min", terr.min(), "max", terr.max(), "mean", terr.mean(), "nonzero cells", (terr > 1e-6).sum())

ind = w.individuals_state()
print("num alive individuals:", len(ind))
print("sample individual keys:", list(ind[0].keys()) if ind else "none")

batch = w.drain_experience_log()
if batch is None:
    print("experience batch: EMPTY (unexpected after 300 ticks with pop ~80)")
else:
    print("experience batch: ids shape", batch["ids"].shape, "sense shape", batch["sense"].shape,
          "action shape", batch["action"].shape, "energy shape", batch["energy"].shape)
    print("sample row: id", batch["ids"][0], "tick", batch["ticks"][0], "energy", batch["energy"][0])
    print("sense row sample (first 5 vals):", batch["sense"][0][:5])

# Drain again immediately -- should be empty/None since nothing ticked since last drain.
batch2 = w.drain_experience_log()
print("second immediate drain (should be None):", batch2)

for _ in range(300):
    w.tick()
batch3 = w.drain_experience_log()
print("third drain after 300 more ticks:", "None" if batch3 is None else batch3["ids"].shape)
