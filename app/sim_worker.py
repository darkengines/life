"""Runs the pixel-growth simulation as its OWN OS process, not a background
thread inside the web server. Measured root cause of a severe latency bug:
with the sim running as a thread sharing the interpreter with uvicorn's
FastAPI event loop, EVERY request -- even a no-op /ping touching no shared
state -- took ~200-220ms, because Python's GIL was almost entirely held by
the CPU-heavy tick() loop (numpy + torch + cKDTree work), starving the event
loop of scheduling time. A separate process has its own GIL and its own
scheduling: the web server (live_app.py) never competes with the simulation
for CPU time at the interpreter level.

This process writes the latest world state to a local file via atomic
rename (write-then-replace) every tick; live_app.py just reads that file on
each HTTP request. No sockets, no shared memory, no locks needed."""
import colorsys
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import orjson

# Engine v2: the ENTIRE simulation now runs natively in Rust (rust_world),
# not just the FK/NN kernels -- this replaced pixel_world.py's per-individual
# Python object loop (measured ~25-40ms/tick of pure CPython bookkeeping
# overhead at pop~1000, dwarfing the numeric kernels) with a real ECS-style
# structure-of-arrays engine. Measured result: 131ms/tick -> 12.8ms/tick at
# pop~1000 (>10x), and population now scales to several thousand instead of
# capping out around 1000. Python's only job here is orchestration: call
# tick(), pull out state, narrate the chronicle, serialize to disk.
import rust_world

WORLD_SIZE = 240  # was 160 -- a real "bigger, more diverse world": 2.25x the area, room for
                  # genuinely distinct regions (open water, sand floor, rock formations) far
                  # enough apart that different lineages can actually specialize regionally
                  # instead of the whole population being within a few body-lengths of every
                  # biome at once. Founders/food/pop-cap scaled to keep density comparable
                  # (not making the world emptier by just stretching it), backed by this
                  # session's real, measured performance headroom (tick/publish decoupling +
                  # the O(4ms)/tick engine cost leave plenty of room before a bigger field
                  # grid or spatial structure become the bottleneck).
# Food grows far more slowly than it used to. At 0.03 a grazed cell refilled
# almost immediately, so a patch never ran out, sitting still beat travelling,
# and a crowded world of small fast breeders out-competed anything complex --
# creatures ended up as 3-part worms with no organs, just turning in place and
# reproducing. Foraging only means something if food can actually be used up.
# Moderated from 0.006 after an 18000-tick sweep: at that rate BOTH test
# seeds collapsed to a handful of individuals, so while the live world happened
# to survive it, the setting is a coin flip. Half the original rate keeps
# foraging meaningful -- a grazed patch does deplete and travelling between
# them matters -- without making extinction the default outcome.
# Lowered again on the owner's instruction ("less food plz, but more storable
# energy"). The two go together: scarcer food only produces interesting
# behaviour if a body can bank a surplus and live off it, which is what the
# new storage capacity provides. Scarcity without a larder is just starvation.
# Cut again: at 0.009 the population still idled on 723 mean energy with
# animals standing still, so food was doing no selecting whatever.
FOOD_REGROW_RATE = 0.004
FOOD_PATCHES = 50  # was 25
POP_CAP = 6000  # was 4000
FOUNDER_COUNT = 300  # was 160 -- a bigger world needs more founders to avoid the Allee effect -- the new,
                     # more realistic reproduction economy (real feeding requirement, developmental
                     # growth, female recovery period) makes bootstrap failure a bigger risk than
                     # before, since reproduction is genuinely harder-won now; more founders is the
                     # established mitigation, not a full fix (some seeds will still struggle)
                    # (measured earlier: same code, different seeds, gave wildly
                    # different bootstrap outcomes at low founder counts)

# First concrete step toward "a shared brain trained asynchronously on the
# GPU from experience replay" (explicitly requested, deliberately deferred
# past this point -- see rust_world's ExperienceRow doc comment for why the
# engine only LOGS raw transitions and does no reward shaping or training
# itself). This just persists what the Rust side samples so a future,
# separate PyTorch/CUDA process has real data to train on whenever it's
# built; nothing currently reads these files back.
VAR = Path(__file__).resolve().parent.parent / "var"
EXPERIENCE_LOG_DIR = VAR / "experience_log"
# Weights produced by the asynchronous GPU trainer (app/train_encoder.py).
# The simulation never waits on training: it just picks up better perception
# whenever the file changes, and runs perfectly well if it never does.
ENCODER_WEIGHTS_PATH = VAR / "shared_encoder.npz"
ENCODER_RELOAD_INTERVAL = 20.0
EXPERIENCE_FLUSH_INTERVAL = 30.0  # seconds -- infrequent on purpose, this is background data collection, not a hot path
EXPERIENCE_LOG_MAX_FILES = 150  # rotation cap (~130MB). Was 500, which reached 446MB on disk for data nothing consumes yet;
                                # raise it again once a training process actually reads these.

STATE_PATH = VAR / "_live_state.json"
TMP_PATH = STATE_PATH.with_suffix(".tmp")
STATIC_STATE_PATH = VAR / "_static_state.json"
STATIC_TMP_PATH = STATIC_STATE_PATH.with_suffix(".tmp")
FIELD_STATE_PATH = VAR / "_field_state.json"
FIELD_TMP_PATH = FIELD_STATE_PATH.with_suffix(".tmp")
RESET_PATH = VAR / "_reset_request"
SPEED_PATH = VAR / "_speed_control.json"
FOOD_DROP_PATH = VAR / "_food_drops.json"
FOOD_DROP_AMOUNT = 6.0  # a generous single clump -- meant to be a real, visible local boost
# dt=0.1 simulated seconds per tick, so 1x real-time is 10 ticks/sec -- the
# rate at which simulated time and wall-clock time move together.
REALTIME_TICK_RATE = 1.0 / 0.1

# --- narrated world chronicle: milestone tracking, not per-tick noise -----
_chronicle = deque(maxlen=60)
_history = deque(maxlen=400)  # sampled every 5 ticks -> ~2000 ticks of population history
_records = {
    "max_age": 0,
    "max_size": 0,
    "pop_milestone_idx": 0,
    "alive_lineages": {},   # color -> True, currently alive
    "lineage_names": {},    # color -> last-known generated name (kept after extinction)
    "weather": None,
}
AGE_MILESTONES = [300, 600, 1200, 2500, 5000, 10000, 20000]
SIZE_MILESTONES = [5, 8, 12, 16, 20, 25, 30, 40]
POP_MILESTONES = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]
WEATHER_LABELS = {
    "food_bloom": "A food bloom sweeps the world -- abundance everywhere",
    "cold_snap": "A cold snap grips the world -- metabolism surges, the weak starve faster",
    "fertile_surge": "A fertile surge takes hold -- maturity comes early this season",
}

# --- procedural lineage names: color -> hue name, evolved average traits ->
# an adjective, a deterministic hash of the color -> a noun. Names can drift
# over generations as a lineage's average traits genuinely shift (this is
# not a fixed label assigned at founding), so a lineage's identity in the
# chronicle reflects what it currently IS, not just what color it started as.
_HUE_NAMES = [
    (0.00, "Crimson"), (0.06, "Amber"), (0.13, "Gold"), (0.20, "Lime"),
    (0.30, "Jade"), (0.45, "Teal"), (0.55, "Azure"), (0.63, "Sapphire"),
    (0.72, "Violet"), (0.80, "Magenta"), (0.88, "Rose"), (0.95, "Scarlet"),
]
_NOUNS = ["Drifter", "Weaver", "Biter", "Wanderer", "Glider", "Strider",
          "Coiler", "Lurker", "Grazer", "Hunter"]


def _hue_name(color):
    r, g, b = (c / 255.0 for c in color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:
        return "Pale"
    return min(_HUE_NAMES, key=lambda hn: min(abs(h - hn[0]), 1 - abs(h - hn[0])))[1]


def _adjective(row):
    if row["avg_bite_force"] > 1.3:
        return "Savage"
    if row["avg_bite_force"] < 0.3:
        return "Gentle"
    if row["avg_bend_amplitude"] > 0.7:
        return "Whirling"
    if row["avg_bend_amplitude"] < 0.15:
        return "Sluggish"
    if row["avg_toughness"] > 1.5:
        return "Armored"
    if row["avg_stickiness"] > 0.7:
        return "Clinging"
    if row["avg_size"] > 10:
        return "Sprawling"
    return "Wandering"


def generate_lineage_name(color, row):
    noun_idx = (int(color[0]) * 7 + int(color[1]) * 13 + int(color[2]) * 19) % len(_NOUNS)
    return f"{_adjective(row)} {_hue_name(color)} {_NOUNS[noun_idx]}"


def _fmt_lineage(color):
    return _records["lineage_names"].get(color, f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")


def species_summary(world, tick_count):
    """Names the lineages the ENGINE aggregated (see rust_world's
    `species_summary`) and records those names for the chronicle.

    The numeric grouping itself used to live here, iterating the full
    per-individual state list -- which meant seven traits had to be carried
    in every one of ~6000 per-individual dicts purely so this loop could sum
    them back up. Moving the accumulation into Rust (one pass over
    contiguous SoA arrays, already sorted by count) let those fields be
    dropped from the published payload entirely; what's left here is the
    stateful/textual part, which runs over ~200 lineages, not 6000
    individuals."""
    rows = world.species_summary()
    for row in rows:
        color = tuple(row["color"])
        row["name"] = generate_lineage_name(color, row)
        _records["lineage_names"][color] = row["name"]
    return rows


def update_chronicle(alive, tick_count, weather):
    if weather != _records["weather"]:
        if weather is not None:
            _chronicle.appendleft(f"[t{tick_count}] {WEATHER_LABELS.get(weather, weather)}")
        elif _records["weather"] is not None:
            _chronicle.appendleft(f"[t{tick_count}] the weather calms")
        _records["weather"] = weather

    if not alive:
        return
    current_colors = {tuple(ind["color"]) for ind in alive}

    for color in list(_records["alive_lineages"]):
        if color not in current_colors:
            _chronicle.appendleft(f"[t{tick_count}] the {_fmt_lineage(color)} lineage went extinct")
            del _records["alive_lineages"][color]
    for color in current_colors:
        _records["alive_lineages"].setdefault(color, True)

    oldest = max(alive, key=lambda i: i["age"])
    for m in AGE_MILESTONES:
        if _records["max_age"] < m <= oldest["age"]:
            _records["max_age"] = m
            _chronicle.appendleft(f"[t{tick_count}] longevity record: {m} ticks survived (the {_fmt_lineage(tuple(oldest['color']))} lineage)")

    # Physical size (pixel count * inflation), not raw pixel count -- growth
    # is now uniform inflation of a fixed-at-birth body plan (see rust_world's
    # size_scale), so pixel count alone barely changes over a lifetime and
    # would make this milestone go stale almost immediately.
    physical_size = lambda i: i["size"] * i.get("size_scale", 1.0)
    biggest = max(alive, key=physical_size)
    biggest_size = physical_size(biggest)
    if biggest_size > _records["max_size"]:
        crossed = [m for m in SIZE_MILESTONES if _records["max_size"] < m <= biggest_size]
        _records["max_size"] = biggest_size
        if crossed:
            _chronicle.appendleft(f"[t{tick_count}] new size record: {biggest_size:.1f} (the {_fmt_lineage(tuple(biggest['color']))} lineage)")

    pop = len(alive)
    idx = _records["pop_milestone_idx"]
    while idx < len(POP_MILESTONES) and pop >= POP_MILESTONES[idx]:
        _chronicle.appendleft(f"[t{tick_count}] population reached {POP_MILESTONES[idx]}")
        idx += 1
    _records["pop_milestone_idx"] = idx

    if tick_count % 5 == 0:
        _history.append({"tick": tick_count, "population": pop, "lineages": len(current_colors)})


def _new_world():
    world = rust_world.World(WORLD_SIZE, FOOD_REGROW_RATE, 1.0, POP_CAP, None, FOOD_PATCHES)
    world.spawn_random(FOUNDER_COUNT)
    return world


def _reset_records():
    _chronicle.clear()
    _history.clear()
    _records["max_age"] = 0
    _records["max_size"] = 0
    _records["pop_milestone_idx"] = 0
    _records["alive_lineages"] = {}
    _records["lineage_names"] = {}
    _records["weather"] = None
    _chronicle.appendleft("[t0] the world is reset -- a new population begins")


# Was 1/15 (66ms) when this constant was first tuned, at the old 160-size/
# 4000-cap world where a tick cost ~4ms. This session's bigger world+pop cap
# (see WORLD_SIZE/POP_CAP above) pushed measured tick() cost to ~35ms at the
# new population ceiling -- once a SINGLE tick already costs more than half
# of a 66ms interval, the "skip publishing until enough time has passed"
# throttle stops doing meaningful work (there's no room left to actually
# skip more than one iteration), and publish's own cost (measured ~130ms+ at
# pop 6000, dominated by individuals_state()'s 6000 per-individual Python
# dict constructions) ends up paid almost every single tick again -- the
# exact bottleneck this decoupling was built to avoid, just at a bigger
# scale. Raised so there's real headroom for multiple ticks to run between
# publishes at any population size actually reached in practice; the
# frontend's analytic FK reconstruction (see index.html's fkPositions) means
# visual smoothness was already decoupled from publish rate, so this has no
# perceptible cost.
PUBLISH_INTERVAL = 1.0 / 6.0
# Field grids (food/pheromone/blood/acid/light/quorum, each a
# 240x240 float array) diffuse/change slowly tick-to-tick -- unlike
# individuals and species stats, they don't need fresh serialization every
# single publish. Refreshed only every Nth publish; the state dict reuses
# the previous cycle's already-serialized lists otherwise, cutting this
# session's newly-7-fields-wide field serialization cost by ~2/3 with no
# visible staleness (a few hundred ms behind on a diffusing gradient overlay
# is imperceptible).
FIELD_PUBLISH_EVERY = 12

_speed_mtime = None
_speed_multiplier = None  # None = variable/uncapped (ticks run as fast as the CPU allows)


def _check_speed():
    """Re-reads the speed-control file only when it actually changed (an
    mtime check, not a full read+parse every tick -- this runs in the hot
    loop). Written by live_app.py's /speed endpoint. Missing/unreadable
    file or a null multiplier both mean "variable" (no throttle)."""
    global _speed_mtime, _speed_multiplier
    try:
        mtime = SPEED_PATH.stat().st_mtime
    except OSError:
        if _speed_mtime is not None:  # the file existed before and got removed -- back to variable
            _speed_mtime = None
            _speed_multiplier = None
        return
    if mtime == _speed_mtime:
        return
    _speed_mtime = mtime
    try:
        data = json.loads(SPEED_PATH.read_text())
        m = data.get("multiplier")
        _speed_multiplier = float(m) if m else None
    except (OSError, ValueError, AttributeError):
        _speed_multiplier = None


def _apply_food_drops(world):
    """Reads and clears the pending-drops queue every tick (cheap: usually
    empty, a single stat() call). Read-then-immediately-overwrite-empty
    isn't perfectly race-free against a concurrent /drop_food write, but a
    dropped or duplicated click here and there is a fully acceptable cost
    for not needing a real lock on what's otherwise the same atomic-file-
    handoff pattern as /reset and /speed."""
    try:
        if FOOD_DROP_PATH.stat().st_size == 0:
            return
        queue = json.loads(FOOD_DROP_PATH.read_text())
    except (OSError, ValueError):
        return
    if not queue:
        return
    FOOD_DROP_PATH.write_text("[]")
    for drop in queue:
        try:
            world.add_food_at(float(drop["x"]), float(drop["y"]), FOOD_DROP_AMOUNT)
        except (KeyError, TypeError, ValueError):
            pass


_encoder_mtime = None


def _maybe_load_encoder(world):
    """Hot-load trained shared-encoder weights if the trainer wrote new ones.

    Best-effort by design: a missing, half-written or wrong-shaped file just
    means the world keeps the perception it already has."""
    global _encoder_mtime
    try:
        mtime = ENCODER_WEIGHTS_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == _encoder_mtime:
        return
    _encoder_mtime = mtime
    try:
        d = np.load(ENCODER_WEIGHTS_PATH)
        ok = world.set_shared_encoder(d["w"].astype(np.float32).tolist(),
                                      d["b"].astype(np.float32).tolist())
        msg = "loaded perception" if ok else "rejected perception (shape mismatch)"
        # The learned baseline policy is optional: older weight files won't
        # have it, and the world runs perfectly well on evolution alone.
        if all(k in d.files for k in ("pw1", "pb1", "pw2", "pb2")):
            pok = world.set_shared_policy(
                d["pw1"].astype(np.float32).tolist(), d["pb1"].astype(np.float32).tolist(),
                d["pw2"].astype(np.float32).tolist(), d["pb2"].astype(np.float32).tolist())
            msg += ", instinct" if pok else ", instinct REJECTED (shape mismatch)"
        print(f"[encoder] {msg}", flush=True)
    except Exception as e:
        print(f"[encoder] load failed (non-fatal): {e}", flush=True)


def _flush_experience_log(world, tick_count):
    """Drains the Rust-side sampled-transition buffer and writes it as one
    .npz chunk. Best-effort: a failure here (disk full, permissions) should
    never take down the live simulation, so it's caught and logged, not
    raised."""
    try:
        batch = world.drain_experience_log()
        if batch is None:
            return
        EXPERIENCE_LOG_DIR.mkdir(exist_ok=True)
        # Wall-clock-based, not tick-based: a sim reset zeroes tick_count
        # back to 0, which would otherwise collide with (silently overwrite)
        # an earlier chunk file from before the reset.
        path = EXPERIENCE_LOG_DIR / f"exp_{int(time.time() * 1000):013d}_t{tick_count}.npz"
        np.savez_compressed(
            path,
            ids=batch["ids"],
            ticks=batch["ticks"],
            sense=batch["sense"],
            action=batch["action"],
            energy=batch["energy"],
            reward=batch["reward"],
        )
        files = sorted(EXPERIENCE_LOG_DIR.glob("exp_*.npz"))
        if len(files) > EXPERIENCE_LOG_MAX_FILES:
            for stale in files[: len(files) - EXPERIENCE_LOG_MAX_FILES]:
                stale.unlink(missing_ok=True)
    except Exception as e:
        print(f"[experience_log] flush failed (non-fatal): {e}")


def main():
    world = _new_world()
    # Terrain is fixed for the world's entire lifetime -- computed once per
    # world instead of every publish (it was being re-converted from a numpy
    # array to a nested Python list on every single tick for no reason).
    terrain_list = world.terrain_grid().tolist()
    static_version = int(time.time() * 1000)
    _write_json(STATIC_STATE_PATH, STATIC_TMP_PATH, {
        "static_version": static_version,
        "world_size": WORLD_SIZE,
        "terrain": terrain_list,
    })

    consecutive_errors = 0
    tick_count = 0
    last_publish = time.monotonic()
    last_experience_flush = time.monotonic()
    last_encoder_check = time.monotonic()
    publish_count = 0
    cached_fields = None
    field_version = 0
    while True:
        try:
            if RESET_PATH.exists():
                # A user-triggered reset (the frontend's "Reset simulation"
                # button, via live_app.py's /reset endpoint dropping this
                # sentinel) -- most commonly needed after the population
                # gets stuck in a degenerate state (e.g. the whole world
                # reduced to a handful of individuals locked in an
                # unbreakable predation deadlock) with no in-sim recovery
                # path. Rebuild the world from scratch, same as a fresh
                # process start, but without actually restarting the process.
                try:
                    RESET_PATH.unlink()
                except OSError:
                    pass
                world = _new_world()
                terrain_list = world.terrain_grid().tolist()
                static_version = int(time.time() * 1000)
                _write_json(STATIC_STATE_PATH, STATIC_TMP_PATH, {
                    "static_version": static_version,
                    "world_size": WORLD_SIZE,
                    "terrain": terrain_list,
                })
                tick_count = 0
                cached_fields = None
                field_version = 0
                _reset_records()

            # world.tick() itself is cheap (~4ms even at 1000+ population,
            # measured directly) -- what was actually limiting throughput to
            # ~15 ticks/sec was rebuilding the ENTIRE state payload (per-
            # individual Python dicts, species grouping, ~4.5MB of JSON) on
            # EVERY tick, even though the frontend only polls every ~80ms.
            # That's serialization work happening ~5-15x more often than
            # anything ever reads it. Decoupled: the simulation now steps as
            # fast as it actually can, and the expensive publish step only
            # runs often enough to keep up with what's actually being read.
            #
            # That decoupling made ticks run MUCH faster in wall-clock time
            # than before, which is genuinely useful for evolutionary
            # throughput -- but it also meant "watch it happen" became
            # "watch a blur", and made every existing tick-paced timing
            # constant (metabolism, cooldowns, growth) effectively run at
            # whatever multiple-of-real-time the CPU happened to allow, with
            # no way to dial it back. _check_speed()/_speed_multiplier let
            # the frontend pick a real-time-relative pace (or leave it
            # uncapped) without touching any of those tuned constants --
            # this throttles wall-clock ticks/sec, not simulated dt itself.
            _check_speed()
            _apply_food_drops(world)
            tick_start = time.monotonic()
            world.tick()
            tick_count = world.tick_count()
            if _speed_multiplier is not None:
                target_interval = 1.0 / (REALTIME_TICK_RATE * _speed_multiplier)
                remaining = target_interval - (time.monotonic() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)

            now = time.monotonic()
            if now - last_encoder_check >= ENCODER_RELOAD_INTERVAL:
                last_encoder_check = now
                _maybe_load_encoder(world)
            if now - last_experience_flush >= EXPERIENCE_FLUSH_INTERVAL:
                last_experience_flush = now
                _flush_experience_log(world, tick_count)
            if now - last_publish < PUBLISH_INTERVAL:
                consecutive_errors = 0
                continue
            last_publish = now

            alive = world.individuals_state()
            weather = world.weather_name()
            species = species_summary(world, tick_count)  # computed first: names feed the chronicle below
            update_chronicle(alive, tick_count, weather)

            if cached_fields is None or publish_count % FIELD_PUBLISH_EVERY == 0:
                field_version += 1
                cached_fields = {
                    "field_version": field_version,
                    "food": world.food().tolist(),
                    "pheromone": world.pheromone().tolist(),
                    "blood": world.blood().tolist(),
                    "acid": world.acid().tolist(),
                    "light": world.light().tolist(),
                    "quorum": world.quorum().tolist(),
                }
                _write_json(FIELD_STATE_PATH, FIELD_TMP_PATH, cached_fields)
            publish_count += 1

            state = {
                "tick": tick_count,
                "sim_time": world.sim_time(),
                "events": world.events(),
                "weather": weather,
                "static_version": static_version,
                "field_version": field_version,
                "individuals": alive,
                "corpses": world.corpses_state(),
                "species": species,
                "chronicle": list(_chronicle),
                "history": list(_history),
                "world_size": WORLD_SIZE,
                "population": len(alive),
            }
            _write_state(state)
            consecutive_errors = 0
        except Exception as e:
            # The simulation must NEVER die from one bad tick or a transient
            # OS-level hiccup -- a crashed worker previously left the web
            # server silently serving its last frame forever (looked like a
            # frozen page, was actually a dead process). Log and keep going;
            # only bail if something is persistently, unrecoverably broken.
            consecutive_errors += 1
            print(f"tick {tick_count} failed ({type(e).__name__}: {e}), continuing", flush=True)
            if consecutive_errors > 50:
                raise


def _write_state(state):
    _write_json(STATE_PATH, TMP_PATH, state)


def _write_json(path, tmp_path, state):
    """Atomic write-then-replace, with retries: on Windows, os.replace() onto
    a destination file that ANOTHER process currently has open for reading
    can raise PermissionError (WinError 5) -- unlike POSIX, where a rename
    over an open file is always safe. live_app.py opens+reads+closes the
    file very quickly, so a retry after a few milliseconds is virtually
    always enough; if it never clears, skip this write and keep the
    previous (still valid, one-tick-stale) state rather than crashing."""
    payload = orjson.dumps(state)
    tmp_path.write_bytes(payload)
    for attempt in range(5):
        try:
            tmp_path.replace(path)
            return
        except OSError:
            if attempt == 4:
                return  # give up silently for this tick; try again next tick
            time.sleep(0.002)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"sim_worker crashed, restarting in 2s: {e}", flush=True)
            time.sleep(2)
