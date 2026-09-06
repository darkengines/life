"""Does learned instinct help FITNESS, even though it does not help alignment?

A sweep of distillation strength showed no benefit on the directional
behaviours being measured -- but population roughly doubled at moderate
strength (578 with evolution alone, 927 at 0.25, 1028 at 0.60, 637 at full).
That suggests the policy is learning something genuinely useful and that
heading-alignment was simply the wrong yardstick: the trainer optimises for
energy and reproduction, and in a world where circling pays and location
barely matters, what it learns need not look like directional chasing.

This measures fitness directly, across more seeds, with population and
reproduction as the outcome.
"""
import numpy as np
import rust_world
from pathlib import Path

VAR = Path(__file__).resolve().parent.parent / "var"
WEIGHTS = VAR / "shared_encoder.npz"
W, POP_CAP, TICKS = 240, 6000, 3000
SEEDS = (11, 22, 33, 44, 55)


def run(rate, seed, use_policy):
    w = rust_world.World(W, 0.03, 1.0, POP_CAP, seed, 50)
    if use_policy:
        d = np.load(WEIGHTS)
        w.set_shared_encoder(d["w"].astype(np.float32).tolist(), d["b"].astype(np.float32).tolist())
        w.set_shared_policy(d["pw1"].astype(np.float32).tolist(), d["pb1"].astype(np.float32).tolist(),
                            d["pw2"].astype(np.float32).tolist(), d["pb2"].astype(np.float32).tolist())
        w.debug_set_policy_distill_rate(rate)
    w.spawn_random(300)
    for _ in range(TICKS):
        w.tick()
    ev = w.events()
    inds = w.individuals_state()
    return dict(pop=w.population(),
                repro=ev["reproductions"],
                energy=(sum(i["energy"] for i in inds) / len(inds)) if inds else 0.0,
                size=(sum(len(i["positions"]) for i in inds) / len(inds)) if inds else 0.0)


print(f"{'condition':>18} {'mean pop':>9} {'mean repro':>11} {'energy':>8} {'meanPx':>7}   per-seed pops")
for label, rate, use in (("evolution only", 0.0, False), ("distill 0.25", 0.25, True),
                         ("distill 0.60", 0.60, True)):
    pops, repros, energies, sizes = [], [], [], []
    for seed in SEEDS:
        r = run(rate, seed, use)
        pops.append(r["pop"]); repros.append(r["repro"])
        energies.append(r["energy"]); sizes.append(r["size"])
    print(f"{label:>18} {sum(pops)/len(pops):>9.0f} {sum(repros)/len(repros):>11.0f} "
          f"{sum(energies)/len(energies):>8.1f} {sum(sizes)/len(sizes):>7.1f}   {pops}", flush=True)
print("\nIf population and reproduction rise consistently across seeds, the")
print("learned policy is a real fitness gain and alignment was the wrong metric.")
