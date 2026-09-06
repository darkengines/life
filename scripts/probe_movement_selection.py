"""Is survival here luck, or are the movers the ones surviving?

The world stopped collapsing once food became drifting marine snow, and the
obvious question is whether that is a lucky seed or whether the animals that
persist are the ones actually swimming to find food. Those two possibilities
predict different things, and the difference is measurable.

Method: watch the LIVE world over HTTP -- costing the simulation nothing --
and record how far each animal travels over a window. Then wait, and see who
is still alive. If movement is being selected, the survivors' earlier
displacement should be higher than that of the animals that died. If it is
luck, the two distributions match.

Reported alongside are body size and energy for the same two groups, because
both are confounds: a large animal both travels further and survives longer
for reasons that have nothing to do with deciding to swim. So the comparison
is also run within a narrow size band, where that confound is largely removed
and what is left is closer to the actual question.
"""
import json
import math
import statistics
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8002/state"
WINDOW_S = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
WAIT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0


def snap():
    d = json.load(urllib.request.urlopen(URL, timeout=20))
    return d, {
        i["id"]: (i["positions"][0], len(i["positions"]), i["energy"])
        for i in d["individuals"]
    }


def wrapped(p, q, w):
    """The world is a cylinder: x wraps, y does not."""
    dx = abs(p[0] - q[0])
    dx = min(dx, w - dx)
    return math.hypot(dx, p[1] - q[1])


d0, a = snap()
W = float(d0["world_size"])
print(f"tick {d0['tick']}, pop {len(a)} -- measuring movement over {WINDOW_S:.0f}s")
time.sleep(WINDOW_S)
d1, b = snap()

moved = {}
for i in set(a) & set(b):
    moved[i] = wrapped(a[i][0], b[i][0], W)
ticks = d1["tick"] - d0["tick"]
print(f"tick {d1['tick']} ({ticks} ticks elapsed), {len(moved)} animals tracked")
if not moved:
    sys.exit("nothing survived the measurement window")

print(f"waiting {WAIT_S:.0f}s to see who lives...")
time.sleep(WAIT_S)
d2, c = snap()
print(f"tick {d2['tick']}, pop {len(c)}\n")

lived = [i for i in moved if i in c]
died = [i for i in moved if i not in c]


def report(label, ids, band=None):
    sel = ids
    if band:
        lo, hi = band
        sel = [i for i in ids if lo <= b[i][1] <= hi]
    if len(sel) < 8:
        print(f"  {label:<22} too few ({len(sel)})")
        return None
    mv = statistics.mean(moved[i] for i in sel)
    sz = statistics.mean(b[i][1] for i in sel)
    en = statistics.mean(b[i][2] for i in sel)
    print(f"  {label:<22} n={len(sel):<5} moved {mv:6.2f}   parts {sz:5.1f}   energy {en:7.1f}")
    return mv


print(f"of {len(moved)} tracked: {len(lived)} alive, {len(died)} dead\n")
print("all sizes:")
m_l = report("survivors", lived)
m_d = report("died", died)
if m_l and m_d:
    print(f"  -> survivors moved {m_l / m_d:.2f}x as far as the dead")

print("\nwithin 8-20 parts (size confound largely removed):")
m_l = report("survivors", lived, (8, 20))
m_d = report("died", died, (8, 20))
if m_l and m_d:
    print(f"  -> survivors moved {m_l / m_d:.2f}x as far as the dead")
