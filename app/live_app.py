"""Thin FastAPI web server: serves the static page and the latest simulation
state. Runs NO simulation itself -- that lives entirely in sim_worker.py, a
separate OS process, which writes state to _live_state.json every tick. This
process just reads that file and returns its bytes directly. See
sim_worker.py's module docstring for why this split exists (GIL contention
between a CPU-heavy background thread and the HTTP event loop was measured
adding ~200ms to every request, including a no-op /ping)."""
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
# All mutable runtime state lives in var/ (gitignored), never beside the source.
VAR = ROOT / "var"
STATE_PATH = VAR / "_live_state.json"
STATIC_STATE_PATH = VAR / "_static_state.json"
FIELD_STATE_PATH = VAR / "_field_state.json"
RESET_PATH = VAR / "_reset_request"
SPEED_PATH = VAR / "_speed_control.json"
FOOD_DROP_PATH = VAR / "_food_drops.json"

_EMPTY_STATE = (b'{"tick":0,"events":{},"food":[],"pheromone":[],"blood":[],'
                b'"individuals":[],"species":[],"chronicle":[],"world_size":48,"population":0}')
_EMPTY_STATIC_STATE = b'{"static_version":0,"world_size":48,"terrain":[]}'
_EMPTY_FIELD_STATE = (b'{"field_version":0,"food":[],"pheromone":[],"blood":[],'
                      b'"acid":[],"light":[],"quorum":[]}')

app = FastAPI()


def _read_runtime_bytes(path: Path, fallback: bytes) -> bytes:
    # sim_worker.py replaces this file atomically every tick, but Windows can
    # still occasionally deny an open() that lands in the same instant as a
    # replace (measured: ~2% of reads under continuous polling) -- a few
    # retries a couple ms apart is far cheaper than the client ever seeing a
    # failed poll for something this transient.
    for attempt in range(5):
        try:
            return path.read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            time.sleep(0.002)
    return fallback


@app.get("/state")
def get_state():
    return Response(content=_read_runtime_bytes(STATE_PATH, _EMPTY_STATE), media_type="application/json")


@app.get("/static_state")
def get_static_state():
    return Response(content=_read_runtime_bytes(STATIC_STATE_PATH, _EMPTY_STATIC_STATE), media_type="application/json")


@app.get("/fields")
def get_fields():
    return Response(content=_read_runtime_bytes(FIELD_STATE_PATH, _EMPTY_FIELD_STATE), media_type="application/json")


@app.post("/reset")
def reset():
    # This process runs no simulation itself (see module docstring), so it
    # can't reinitialize the world directly -- it just drops a sentinel file
    # that sim_worker.py (the actual owner of the World) polls for once per
    # tick and acts on, matching the existing atomic-file-handoff pattern
    # rather than adding a second IPC channel just for this.
    RESET_PATH.touch()
    return {"ok": True}


@app.post("/speed")
async def set_speed(request: Request):
    # Same atomic-file-handoff pattern as /reset: this process owns no
    # simulation state, so it just writes what the frontend asked for and
    # lets sim_worker.py (which owns the actual tick loop) pick it up.
    # Body: {"multiplier": 1|2|4|8|16|32} for a real-time-relative throttle,
    # or {"multiplier": null} for "variable" (uncapped, as fast as possible).
    body = await request.json()
    SPEED_PATH.write_text(json.dumps(body))
    return {"ok": True}


@app.post("/drop_food")
async def drop_food(request: Request):
    # Queued, not overwritten: unlike /speed (a single current setting),
    # multiple drops can happen faster than sim_worker.py's next check --
    # each click should actually feed, not get clobbered by the next one.
    body = await request.json()
    x, y = float(body["x"]), float(body["y"])
    try:
        queue = json.loads(FOOD_DROP_PATH.read_text())
    except (OSError, ValueError):
        queue = []
    queue.append({"x": x, "y": y})
    FOOD_DROP_PATH.write_text(json.dumps(queue))
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
