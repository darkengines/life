"""Which change killed the world?

Population went from 361 to 1. Several mechanisms landed in quick succession
and patching them one at a time by eye has not converged, so this turns each
one OFF in turn against the same seeds and asks which restores viability.

An ablation is the right instrument here for the same reason a control arm is:
it tells you which change is responsible, rather than which change you happened
to look at last.
"""
import statistics
import sys

import rust_world

W, POP_CAP, PATCHES = 240, 6000, 50
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [11, 22]
SAMPLE = 500


def run(label, seed, **kw):
    w = rust_world.World(W, 0.004, 1.0, POP_CAP, seed, PATCHES)
    if "pathogen" in kw:
        w.debug_set_pathogen_rate(kw["pathogen"])
    if "life" in kw:
        w.debug_set_life_history(*kw["life"])
    if "pressure" in kw:
        w.debug_set_space_pressure(*kw["pressure"])
    if "halfsat" in kw:
        w.debug_set_graze_half_saturation(kw["halfsat"])
    if "plankton" in kw:
        w.debug_set_plankton(*kw["plankton"])
    w.spawn_random(300)
    traj = []
    for t in range(1, TICKS + 1):
        w.tick()
        if t % SAMPLE == 0:
            traj.append(len(w.individuals_state()))
            if traj[-1] == 0:
                break
    return traj


ARMS = [
    ("baseline (all on)", {}),
    ("no disease", {"pathogen": 0.0}),
    ("old life history", {"life": (0.2, 0.0)}),
    ("no crowd mortality", {"pressure": (12.0, 0.0)}),
    ("no graze refuge", {"halfsat": 0.0}),
    ("richer plankton", {"plankton": (1.6, 14.0)}),
]

print(f"{TICKS} ticks | seeds {SEEDS} | population every {SAMPLE}\n")
for label, kw in ARMS:
    for seed in SEEDS:
        traj = run(label, seed, **kw)
        end = traj[-1] if traj else 0
        print(f"{label:<20} seed {seed}: end {end:>5}  min {min(traj) if traj else 0:>5}  "
              f"traj {traj[:10]}")
        sys.stdout.flush()
    print()
