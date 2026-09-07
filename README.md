# Pixel-Growth Ecosystem

A browser-visualised artificial-life simulation: creatures built from chains of connected
"pixels", with evolving neural-network brains, physics-based swimming, growth, sexual
reproduction, combat, terrain, and a growing set of ecological mechanics drawn from real biology.

Nothing in the creature behaviour layer is scripted. The engine provides **heritable traits**,
**sensory channels**, and **selection pressures**; what creatures actually *do* with them is left
to evolution. That constraint is the core design rule of the project and is why most features
below are described as "made possible" rather than "implemented".

---

## Contents

1. [Running it](#1-running-it)
2. [Layout](#2-layout)
3. [Architecture](#3-architecture)
4. [Rendering](#4-rendering)
5. [Biological concepts implemented](#5-biological-concepts-implemented)
6. [Performance](#6-performance)
7. [Findings, including negative ones](#7-findings-including-negative-ones)
8. [Roadmap / wishlist](#8-roadmap--wishlist)
8A. [Requirements, concerns and ideas from the project owner](#8a-requirements-concerns-and-ideas-from-the-project-owner)
9. [Development notes](#9-development-notes)
10. [Bibliography](#10-bibliography)

---

## 1. Running it

```bash
# build the engine
cd engine && cargo build --release

# install the extension (Windows)
cp target/release/rust_world.dll \
   <python>/Lib/site-packages/rust_world/rust_world.cp312-win_amd64.pyd

# run (two processes)
cd app
python sim_worker.py
python -m uvicorn live_app:app --host 127.0.0.1 --port 8002 --no-access-log
```

Then open <http://127.0.0.1:8002/>.

The simulation runs as a **separate process, not a thread**: with the sim as a thread sharing the
interpreter with uvicorn's event loop, every HTTP request took ~200ms because the GIL was held
almost continuously by the tick loop. Separate processes have separate GILs.

**Deployment is order-sensitive.** On Windows the running `sim_worker.py` holds a lock on the
compiled `.pyd`. The sequence is: build → stop worker → copy DLL → clear stale state → **restart
immediately** → verify afterwards. Doing verification work in the gap between stop and restart left
the live simulation visibly broken more than once.

---

## 2. Layout

```
engine/          Rust simulation engine (PyO3 extension module `rust_world`)
  src/           the whole engine: physics, combat, fields, individuals, terrain, spatial
app/
  sim_worker.py  simulation process; publishes world state
  live_app.py    FastAPI server
  static/        frontend (single page, WebGPU renderer)
scripts/         benchmarks, A/B harnesses, verification scripts
var/             runtime state — world snapshot, IPC files, experience log (gitignored)
```

---

## 3. Architecture

**ECS-style structure-of-arrays.** `Individuals` and `PixelArena` are separate component tables
with a foreign-key relationship (1NF), not per-creature Python objects. This replaced a Python
object loop measured at ~25–40ms/tick of pure CPython bookkeeping overhead at ~1000 population.
Variable-length pixel data lives in one shared arena with a free-list allocator; dead individuals
are tombstoned and slot-recycled rather than rebuilding the population array each tick.

**Parallel/sequential split.** Each tick runs a read-only parallel phase (sense → brain forward
pass → thrust → contact forces) via rayon, then a sequential mutation phase (movement integration,
reproduction, collision, death) where cross-individual mutation makes parallelism unsafe.

**Stable IDs everywhere.** Any cross-reference between individuals (`attached_to`, `parent_id`)
stores a stable `u64` id resolved through an `id_to_slot` map, never a raw slot index. Slots get
recycled, and a dangling slot index caused a real bug where an attacker silently began dragging an
unrelated newborn that had reused a freed slot — visible as "creatures suddenly rush somewhere and
die".

**Tick/publish decoupling.** Simulation stepping and state serialisation run on different cadences;
the frontend reconstructs intermediate frames analytically rather than needing a snapshot per frame.

**File-based IPC.** `_reset_request` (sentinel), `_speed_control.json` (latest-value, mtime-checked),
`_food_drops.json` (append-only queue drained each tick). No sockets, no shared memory, no locks.

**Architectural inspiration:** a personal C++/Vulkan engine at `H:\GameEngine` — its `FreeList.hpp`
buddy allocator (the pixel-arena allocation pattern) and its per-pipeline instanced-draw batching in
`Renderers/SceneRenderer` (the WebGPU capsule renderer). Inspired by, not copied from.

---

## 4. Rendering

WebGPU (explicitly chosen over WebGL2), instanced capsule-chain rendering. Each body segment is a
capsule — a thick line segment with round caps, the exact convex hull of two circles, so there is no
pinching at joints. One instance buffer covers every segment of every individual; two draw calls per
frame.

Two hard-won details:

- **Two draw passes, not one.** An outline pass at larger radius, then a fill pass. A single-pass
  per-fragment fill/outline branch produced visible seams, because GPU instance draw order is not
  guaranteed and one segment's outline could rasterise over a neighbour's fill.
- **WebGPU validation errors are asynchronous.** `createShaderModule` / `createRenderPipeline` do not
  throw on a bad shader, so a broken pipeline can sit behind a `gpuReady === true` flag drawing
  nothing. All shader/pipeline creation is wrapped in `pushErrorScope('validation')` /
  `popErrorScope()`. This is how a WGSL reserved-word collision (a variable named `pass`) was caught.

**Client-side forward kinematics.** The frontend ports the engine's exact `world_positions_at`
formula to JS and evaluates it at a continuously-extrapolated `sim_time`, rather than interpolating
published joint positions. Raw interpolation visibly aliased the periodic bend-wave animation once
ticks and publishes were decoupled. A useful side effect: visual smoothness became independent of
publish rate, which is what made the later publish-rate optimisations free.

---

## 5. Biological concepts implemented

Each is a real, named mechanism, researched rather than invented. The pattern throughout: add a
**heritable trait**, a **sensory channel**, and an **economic cost**, then let selection decide —
never an `if` statement checking species identity.

### Locomotion & body
- **Resistive force theory** — thrust computed from per-segment normal/tangential drag anisotropy,
  not an arbitrary "swim speed" stat.
- **Forward kinematics body chains** — a tree of pixels with rest angles, joint flex, per-part size.
  Growth is *uniform inflation* (`size_scale`) of a body plan fixed at birth, never sprouting parts.
- **Crawling** — a distinct locomotion mode near the floor, gated on evolved `crawl_affinity`.
- **Digging** — the sand seafloor is genuinely solid; penetrating it needs evolved `dig_strength`,
  which most founders lack.

### Sensing & cognition
- **Tiny MLP brains** (sense → 12 hidden → act), per-individual weights, evolved by mutation and
  inheritance with a heritable `weight_transmission_rate`. No backpropagation.
- **Recurrent memory** — 4 channels written by the brain, fed back as inputs next tick.
- **Vision** — direction and proximity to the nearest meaningfully *bigger* body (threat), nearest
  meaningfully *smaller* (prey), and nearest *eligible mate*. Added so context-appropriate aggression
  could evolve instead of undifferentiated constant fighting.
- **Chemical gradients** — food, blood, acid, light, pheromone, quorum and territory fields, each
  diffusing with its own rate and decay.

### Social & signalling
- **Kin recognition** — a continuous heritable signature vector, sensed as similarity to the nearest
  neighbour. Deliberately *only* a sensory channel: nothing suppresses aggression toward kin.
- **Inclusive fitness / Hamilton's rule** — a parent is rewarded when its *child* reproduces, so
  there is real selective payoff for keeping offspring alive to breeding age.
- **Quorum sensing** — every individual constitutively emits into a shared density field just by
  existing, mirroring real autoinducer production. Sense-only.
- **Stigmergy** — indirect coordination by modifying the environment. Implemented twice: the
  pheromone trail field and the territory-marking field.

### Defence & conflict
- **Graded combat** — per-pixel health pools, size-scaled effective toughness (fixing "small creature
  one-shots big creature"), headshot multiplier on root hits, energy-scaled attacker damage,
  regeneration.
- **Venom** — a bite injects the attacker's evolved `acid_secretion` into the target's location,
  reusing the acid field as a slower alternative kill path.
- **Camouflage / crypsis** — evolved colour versus local ambient colour gives a real detection
  penalty. Nothing decides which colour is safe; it depends entirely on where the creature lives.
- **Aposematism** — conspicuousness (inverse of camouflage) × real defensive capability (toughness or
  acid) gives a chance to deter an attack. Reuses the camouflage machinery, because warning colours
  and camouflage are the same signal read two ways.
- **Attack costs energy** — reflexive aggression is economically selected against rather than free.

### Life history
- **Sexual reproduction** — requires an eligible opposite-sex mature partner in range, recent
  feeding, and energy above threshold; females have a real post-birth recovery cooldown.
- **Sessile anchoring (sea anemones)** — evolved `anchor_strength` resists thrust, gravity and drift
  *only while resting on solid ground*, plus a metabolic discount, because sessile organisms have far
  lower upkeep. Combined with the pre-existing stickiness/attachment mechanic this produces genuine
  ambush-predator behaviour with **no new combat code**.
- **Day/night cycle** — a real variable, not decoration: it modulates the ambient colour camouflage
  is judged against, and is a brain input, so nocturnal/diurnal strategies can evolve.
- **Weather** — food blooms, cold snaps, fertility surges.

### Spatial ecology & diversity
- **Territoriality / home ranges** — heritable `territoriality` drives passive scent deposits into a
  slow-decaying, low-diffusion field (real scent marks are local and persistent, unlike trail
  pheromones). Each individual has a `home_pos` fixed at its own birth site — a "den" — which is *not*
  inherited, so offspring disperse into new ranges. The brain senses direction-to-home and local mark
  strength. Based on central-place-forager and scent-mediated home-range formation models.
- **Negative frequency-dependent (Red Queen) selection** — specialist pathogen pressure keyed to a
  lineage's *global* share of the population, with an evolvable `disease_resistance` trait that
  carries a real metabolic cost. Counteracts competitive exclusion. See §7 for why this replaced a
  Janzen–Connell design, and for the calibration data.

---

## 6. Performance

All figures measured directly, never assumed.

| Change | Result |
|---|---|
| Python object loop → Rust SoA engine | 131ms → 12.8ms per tick at pop ~1000 |
| Tick/publish decoupling | ~15 → ~50 ticks/sec |
| Spatial grid `nearby_radius` | Fixed a real bug: the 3×3-cell search silently capped *any* caller's radius at ~3–4 units, making `VISION_RANGE = 12` a lie |
| Publish interval + field-grid cadence | 4.5 → 5.25 ticks/sec at pop 6000 |
| Species aggregation moved into Rust | 5.25 → **10.1 ticks/sec** at pop 6000 |

The last is the most transferable lesson. Seven per-individual traits were carried in all ~6000
published dicts *purely* so Python could sum them into per-lineage averages — ~42,000 wasted key
insertions and float boxings per publish. Grepping the frontend showed it never read them
per-individual, only as species-table averages. Moving aggregation into Rust let those fields be
dropped entirely. **Before optimising how expensively a payload is built, check whether the consumer
actually reads all of it.**

A recurring scaling trap: tick/publish decoupling silently stopped working when the world grew
(160→240, pop cap 4000→6000). Once a single tick costs more than the publish interval, a "skip
publishing until enough time has passed" throttle has no room left to skip anything. This will recur
on any future world-size increase — it is a scaling relationship, not a one-time bug.

---

## 7. Findings, including negative ones

**Reproduction was broken in a non-obvious way.** `has_nearby_mate()` was satisfied by *any* nearby
living individual — not checking sex, maturity, or cooldown. Combined with movement costing energy,
this made passive clustering strictly more reproductively successful than exploring: exactly
backwards. Verified with a 30-seed same-sex vs opposite-sex test (0 vs 2415 reproductions).

**Food gradients were useless at range.** A 1-cell finite difference gives zero signal until a
creature already touches a patch — not "smell" but "bump into it and notice too late". This explained
creatures swimming to the surface and starving.

**Two rounding systems must agree exactly.** Terrain generation truncated `floor_height` to `u32`
while physics used the raw float, so anything "resting on the sand surface" was, by the terrain grid's
own classification, standing on empty space. Any boundary computed independently by two systems must
share identical rounding, not merely the same formula.

**A failed test can be a bad test.** The first anchor-strength test showed anchored creatures drifting
*more* than unanchored ones. Cause: the chosen seed placed the creature under a rock column, so a
one-time settling correction on tick 1 — independent of anchoring — dominated the measurement. Adding
a settling period gave the correct result (8.15 vs 0.00 units of drift).

**Janzen–Connell does not transfer to this engine.** The goal was to stop a single lineage reaching
55–69% of the population. The mechanism (specialist pathogens hurting locally-dense conspecifics) was
implemented and measured:

1. *First attempt* reduced lineage counts and crashed population in all three A/B seeds. Cause: the
   response used a `tanh` saturating by density ~3, compressing the signal into a near-flat tax.
2. *Second attempt* sharpened the kin kernel so only genuinely similar neighbours counted. Separation
   between abundant and rare lineages barely moved: **1.38× → 1.37×**.
3. The reason is structural. Janzen–Connell relies on seed dispersal producing a *gradient* of local
   conspecific density. Here offspring are born adjacent to their parents, so **every** lineage clumps
   equally tightly — a 50-member lineage and a 4000-member one have nearly identical *local* density.
   Local density measures clumping, not global rarity, and no response curve recovers a signal that
   is not there.

Replaced with **negative frequency-dependent (Red Queen) selection** keyed on global lineage share.
Measured with *effective* species count (inverse Simpson) rather than raw lineage count, because raw
count is confounded — fewer individuals mechanically means fewer distinct colours:

| seed | rate | pop | lineages | **effective species** | dominant % | top-5 % |
|---|---|---|---|---|---|---|
| 11 | 0.000 | 6000 | 163 | **2.1** | 67.0% | 89.4% |
| 11 | 0.020 | 2224 | 82 | **3.7** | 47.2% | 81.7% |
| 11 | 0.035 | 1972 | 77 | **10.5** | 19.6% | 59.4% |
| 22 | 0.000 | 5960 | 143 | **2.0** | 69.3% | 89.0% |
| 22 | 0.020 | 5630 | 152 | **2.9** | 58.3% | 72.7% |

The striking result is the baseline: **the untreated world is effectively a two-species world**
(2.0–2.1) despite 143–163 nominal lineages. Shipped at `0.02`, which buys real evenness at little
population cost; `0.035` buys far more diversity but costs a third to two thirds of the population.
This is a genuine diversity-versus-population trade-off, not a single correct value.

**The same settings, two completely different worlds.** A scarcity sweep run at identical
parameters landed on wildly different outcomes depending only on the RNG seed:

| regrow | meal | seed | pop | mean parts | >=12 parts | worms |
|---|---|---|---|---|---|---|
| 0.030 | 1.4 | 11 | 1320 | 4.7 | **1.9%** | 35% |
| 0.030 | 1.4 | 22 | 1055 | **13.3** | **63.1%** | 5% |
| 0.006 | 8.0 | 22 | 707 | 12.7 | 47.8% | 6% |

This is the single most important measurement in the project so far, for two reasons. First, **the
complex basin already exists** — the engine can and does produce worlds where nearly two thirds of
all animals carry twelve or more parts and worms are a rounding error. The problem was never that
complexity is impossible; it is that which basin a run falls into is not reliable. Second, it means
**every single-seed measurement in this project is worthless**, including several taken earlier. The
spread between seeds is far larger than any effect being measured. All A/B work from here on runs
multiple seeds and reports the spread.

The likely mechanism is a classic **priority effect / alternative stable state** (Beisner et al.
2003; Fukami 2015): if worms saturate the world first they pin food to zero, and nobody can then
accumulate the energy surplus a large body needs to get established. If a large lineage gets going
first, it eats the worms and holds the space. Making the complex basin *reliable*, rather than a
coin flip, is now the central objective.

**Metabolism was linear in body size, which is biologically wrong and quietly fatal to complexity.**
Upkeep was computed as a straight sum over parts, so a sixteen-part animal paid sixteen times the
running cost of a one-part blob with nothing offsetting it. Complexity was therefore a pure tax and
selection stripped it out as fast as growth added it. Real metabolic rate scales as roughly
*mass^(3/4)* across twenty-seven orders of magnitude of body mass (Kleiber 1932; West, Brown &
Enquist 1997) — large animals get a large per-gram energy *discount*, and that discount is much of
why being big is viable at all. Now implemented, normalised at one part so it only ever makes large
bodies cheaper, never small ones dearer.

**Active organs were being selected against.** Measured on the live world, the passive organs pay
(gut 13.9% of all tissue, tentacle 6.1%) while every *active* organ sits below the ~5% that random
differentiation alone would produce: flipper 3.7%, eye 2.8%, mouth 2.1%. Eyes matter most, because
without eyes there is no perception, and with no perception no amount of brain can help. Two causes
were found, and they were deliberately measured apart: sensors were priced like armour plate, and —
more damning — **foraging was never gated on eyes at all**. The food gradient was sampled out to
range 18 for every creature regardless of anatomy, so a blind animal found distant food exactly as
well as a sighted one and an eye bought nothing but upkeep. Smell is now short-range and sight is
what reaches; chemoreception in water genuinely is diffuse and local while vision is directional and
long-ranged, so this is the honest model as well as the useful one.

**Bodies stacked because collision could not see them.** Animals piling on top of one another was
blamed on behaviour for a long time, and a crowding energy cost was added to punish it. That was
treating the symptom. Contact broad-phased through the spatial grid's default one-cell search --
about three world units around the **root** -- which was perfectly adequate when animals were
three-part blobs. Bodies now reach forty parts and span twenty units or more, so two animals lying
completely across one another were never even tested for contact unless their roots nearly touched.
No penalty can discourage an overlap the engine never detects. Two further defects sat behind it:
the narrow phase tested only the single *nearest* component of the other body, and nearest is not
deepest -- a slightly further but much fatter part can be penetrating while the closest one is not
-- and contact was a pure force, which has to fight mass and damping to undo a penetration, a fight
a heavy body never wins. Fixed by searching each body's real reach, testing every component pair
(which costs nothing extra, since the inner scan already ran over all of the other body's parts),
and adding direct positional correction of a fraction of the penetration per tick, mass-weighted so
the lighter animal yields more. Measuring it correctly took a second attempt. The raw "what fraction of components sit inside
another animal's component" figure is confounded by density, and badly: raising collision stiffness
twelvefold drove overlap from 56% to 95%, which looks like a catastrophic regression and is nothing
of the kind -- the same change doubled the population, and denser worlds overlap more whatever the
solver does. Scoring against a null model (the same bodies, each rigidly displaced to a random
position, preserving its own shape) gives a density-controlled ratio:

| solver | overlap vs random placement | population |
|---|---|---|
| force only (the original) | **1.46** | 401-903 |
| stiffness x12, no correction | 0.98 | 1667-1929 |
| positional correction | **0.99** | 1272-1658 |

So the original solver left bodies clumping about 46% *more* than chance, and both fixes remove
that entirely. Contact resolution now works.

It also revealed a genuine trade-off that no amount of solver work removes. Keeping bodies apart
roughly **doubled the population**, because 93% of deaths here are predation and anything that
stops bodies interpenetrating also stops predators reaching their prey. The world therefore still
*reads* as crowded even though the physics is correct: at twenty thousand components in a 240x240
world, overlap is unavoidable and a ratio of 1.0 is the floor. Crowding is now a density problem,
not a collision problem, and the lever for it is the metabolic economy rather than the solver.

Fixing it correctly also made it **twice as fast**. Contact was broad-phasing through the body
grid, which indexes each animal by its root alone, so the query had to be wide enough to reach the
largest animal in the *world* -- one forty-part giant made every other body, however small, sweep a
radius-twenty neighbourhood of hundreds of mostly empty cells, and then test all of its components
against all of theirs. Indexing the components themselves instead means a component queries only its
own contact reach, about a unit, and finds exactly what could touch it: **72.3 ticks/s at 11589
components, against 34.8 ticks/s at 11991 before**.

The first attempt at that rewrite crashed, and the reason is worth keeping. The position cache and
the part grid are both built *before* the attachment loop, which bites parts off victims and
reallocates their storage -- so a body's offset and count can both move within a single tick. The
original code guarded against this by requiring the cached length to still match the live part
count. The rewrite checked only that the index was inside the cache, which does not reject a body
that shrank, and read past the end of the arena. A cache that is one phase stale is not the same as
a cache that is merely indexed safely.

**"They swim with the tail frontward" -- finally diagnosed, and it was never a brain problem.**
Two measurements together settle this. With the brain frozen out and the body pinned in place, the
mean thrust direction of a single body is *perfectly* locked to that body: identical to five decimal
places across all twelve headings tested, consistency **R = 1.000**. So propulsion is not noisy, and
turning the animal turns its thrust vector exactly as it should. (This is a real improvement --
before rotation was moved to the centre of mass, alignment sat near 81 degrees, i.e. statistically
random.)

But across *different* bodies, the angle between where a body points and where it actually pushes
scatters completely:

| bodies tested | mean offset | consistency |
|---|---|---|
| 15 body plans, 7-20 parts | +141.9 deg | **R = 0.259** |

Individual offsets run +25, +19.6, -10.7, -18.8, -162.1, -103, +154, +166.9, +160.5, -168.7, -171.4,
+113.6, +24.7, +137.8, +129.7 degrees. Several bodies push at 160-170 degrees from their nominal
heading -- genuinely, precisely backwards. So `heading` is simply **not** the direction an animal
swims, and which way any given body goes is a property of the body it happened to grow. The
centroid-derived anterior axis does not capture it.

The decisive part is what came next. The sense vector contained **no self-motion channel at all** --
no velocity, no drift, nothing relating where the animal points to where it is going. An animal
swimming backwards had no way to perceive that it was. This was never a matter of network capacity
or training: no brain of any size could learn to correct a fault it cannot observe, because the
feedback did not exist.

Fixed by adding proprioception -- the animal's own velocity expressed in its own body frame, plus
speed -- and deliberately **not** by rotating thrust to match heading. Correcting the physics would
hand every animal a working body for free; the standing requirement is that motion stays a
consequence of how the body moves and the animal has to learn to use the one it grew. What it gets
is the sense that makes that learnable.

**The food economy: two knobs, and the wrong one was being turned all night.**
"Plankton is too nutritive" was answered by instrumenting where energy actually comes from, which
had never been visible. The answer: **plankton 94% of all energy, predation 1%, scavenging 5%** --
while predation was causing 44% of deaths. Killing was common and nearly worthless, so nothing
could make a living as a predator. Predation now pays out of the prey's own reserves, so a fat
animal is a better meal than a thin one and energy moves *up* the chain instead of being invented
at each link.

Then the calorie value was cut by hand, repeatedly, and the world kept collapsing. The trajectory
showed why: not a slow decline but an immediate crash, 98 animals to 4 within a few hundred ticks,
then a remnant of ten limping along. Each measurement had been taken at a different world age,
which cannot distinguish a wrong food level from a world already dying.

The cause was structural and was my own earlier fix. Charging intake on exposed surface was meant
to stop bigger always winning -- and for a *compact* body it does, since perimeter grows as
sqrt(N). But an **elongated** body's perimeter grows as N, and elongation is exactly what that
pressure selects for, so intake went straight back to scaling as N against upkeep at N^0.75:

| body | intake / upkeep, before |
|---|---|
| 4 parts | **0.98** -- a net loss with food everywhere |
| 12 parts | 1.29 |
| 40 parts | 1.74 |

Founders could not establish at any calorie level. Intake is now sublinear in feeding capacity
(exponent 0.70 against metabolism's 0.75 -- the real filtration exponent), which restores the
ordering for every body shape rather than only compact ones, and gives 1.95 at three parts falling
to 1.67 at sixty-four.

Then a proper sweep, 3 seeds x 14000 ticks, scored on the **minimum population after founding**
rather than an endpoint, because that is where the failure lives:

| calories | production rows | pop | min after founding | starved | eaten |
|---|---|---|---|---|---|
| 0.36 | 6 | **9** | 6 | **97%** | 3% |
| 0.36 | 14 | 217 | 42 | 66% | 30% |
| **0.80** | **6** | 284 | 88 | **46%** | **49%** |
| 0.80 | 14 | 465 | 107 | 41% | 46% |
| 1.60 | 14 | 310 | 154 | 45% | 31% |
| 1.60 | 30 | 615 | 384 | 41% | 38% |

**Total production, not calories per unit, was starving the world**: at 0.36 calories, going from 6
rows to 14 takes the population from 9 to 217. Every hand-tuned change that night adjusted the
wrong knob, and 0.36/6 -- the setting the live world was actually running -- was the single worst
point in the grid. Shipped at 0.80/6: the least total food among the viable settings, and the only
one with a genuinely balanced food web at 46% starvation against 49% predation.

---

## 8. Roadmap / wishlist

Status key: **[ ]** not started · **[~]** partially done · **[?]** needs investigation

### Intelligence (the largest open area)

**Status: the asynchronous GPU training loop is closed and measurably working.** `app/train_encoder.py`
runs as its own process on the RTX 6000, reads the experience chunks the simulation writes, trains a
world model self-supervised (predict the next sense vector and the reward from the current latent and
action), and writes encoder weights atomically; the simulation hot-loads them without ever waiting.
The GPU produces **two** things, and the simulation hot-loads both:

1. **Perception** — the shared encoder, which predicts the world ~3× better than "assume nothing
   changes".
2. **Instinct** — a baseline policy in exactly an individual decoder's shape, learned by
   advantage-weighted regression on the population's *own* successful behaviour. Newborns are pulled
   a quarter of the way toward it at birth and then mutate and evolve from there, so learning reaches
   the population the way instinct does — through births — while every individual still owns its own
   decisions. The blend is partial deliberately: at full strength every creature would start
   identical and the variation selection needs would be gone.

**A result that did NOT replicate — stated plainly.** An earlier measurement found prey-pursuit
alignment improving from −0.0102 to +0.0149 with trained perception and instinct (diff +0.0251
against a ±0.0216 noise band), positive in all four seeds, and it was reported here as
statistically defensible. After the sense space grew (shelter channels, swim effort, turn bias) and
the world changed (forward-direction fix, drifting food blooms), the comparison was re-run with 29
rounds of retraining and larger samples: **all three behaviours now sit within noise** — chase
+0.0041 ±0.0288, flee −0.0087 ±0.0305, mate −0.0058 ±0.0263.

The original effect was marginal (barely clearing its own error bar) and has not held up. The
honest position is that **the training loop is demonstrably learning** — it predicts world dynamics
several times better than a do-nothing baseline and its policy loss falls steadily — but there is
**no reproducible evidence yet that this improves behaviour**. Treat the pipeline as built and
working mechanically, not as validated.

**Original measured starting point** (before the shared encoder and the training loop): evolved
brains were *no better than random ones* at steering toward food (forage alignment −0.012 evolved vs
+0.000 random). Evolution is improving bodies — evolved
populations live longer and hold more energy — but the brains contribute almost nothing to
navigation. Two causes are visible in the logged sense vectors: the four recurrent memory channels
carry the highest variance of all 34 inputs (std 0.76, saturated at ±1), so the network mostly
listens to its own feedback; and with food abundant everywhere and 6.6 fights per birth, foraging
skill barely affects survival, so there is little pressure to be smart. **A sparser, size-structured
world with refuges is therefore a prerequisite for intelligence mattering at all** — the ecology
work below is not a separate track from this one.

- **[~] Composable brain: a sensory component supplies a brain input.** First instance implemented —
  vision is now organ-gated, so a body with no eyes gets zeroed visual inputs instead of free
  universal sight. The brain keeps a fixed set of input slots, but a slot only carries signal if the
  body has the organ feeding it, which makes sensory anatomy an evolutionary decision. Still to do:
  extend this to the other senses (chemoreception, light, pressure/flow) as their own organs, so the
  full sensorium is assembled from parts rather than granted.
- **[ ] Asynchronous brain evaluation.** Network cost is already a real per-tick expense and will
  grow as networks get bigger; it must not sit on the critical path of the simulation loop. The
  intended shape is the same decoupling used elsewhere here: the simulation keeps stepping while
  inference/training happen off the tick loop.
- **[ ] Reconsider the recurrent memory channels.** They dominate the input vector and saturate;
  they may be actively drowning out sensory input rather than providing useful state.
- **[~] Shared brain trained asynchronously on the GPU.** The requested architecture: a larger common
  network trained by experience replay on an RTX 6000 while the simulation keeps running at full
  speed, with improved weights synced back in.
  - *Done:* experience-logging infrastructure. Transitions sampled deterministically (a cheap modulo,
    no RNG, piggybacking the existing sense/decide pass) into a capped Rust buffer, drained via
    `World.drain_experience_log()` into numpy arrays, flushed every 30s as rotating `.npz` chunks
    under `var/experience_log/`. Verified: 4,294 samples covering 3,418 distinct individuals per flush.
  - *Not done:* the replay buffer, the training process itself, weight-sync back into the engine.
- **[ ] Bigger networks.** Brains are currently a 12-unit hidden layer. Explicitly requested and not
  yet addressed; likely coupled to the shared-brain work, since per-individual evolution of large
  weight matrices is not practical.
- **[ ] Decide the architecture question:** does a trained shared network *replace* evolved individual
  brains or sit alongside them? The "evolvable, not hardcoded" rule argues for alongside — a shared
  instinct sub-network plus a still-evolved individual part. Reynolds' camouflage-coevolution work
  (bibliography §9) is the closest published precedent: lifetime learning *plus* evolution.
- **[?] Reward definition.** Logged rows are (state, action, energy). Reward shaping — energy delta,
  death penalty, reproduction bonus — must be computed from consecutive same-id rows by whatever
  consumes the data. Nothing consumes it yet.
- **[ ] Lifetime learning at all.** Currently intelligence changes only between generations. Any
  within-lifetime adaptation would be new.

### User interaction
- **[x] Drop food** by clicking the canvas.
- **[ ] Spawn a registered species.** Explicitly requested alongside drop-food ("drop food or
  registered species for example") and *not* implemented. Would need: a way to save a lineage's full
  genome (traits + brain weights + body plan) to disk, a UI to pick one, and an engine entry point to
  instantiate it at a clicked position. This also unlocks curated match-ups between saved lineages.
- **[x] Pin/highlight a species** by clicking its table row.
- **[ ] Inspect an individual** — click a creature to see its traits, brain, energy, lineage.
- **[ ] Save / load a world**, so an interesting run can be kept or shared.
- **[ ] Draw terrain / place obstacles** interactively.

### Size-structured ecology (the current design direction)

The world currently reaches a high population of uniformly large creatures stacked on top of one
another, which is both visually illegible and ecologically wrong. The target instead is a
**size-structured community**: fewer, larger animals that are genuinely expensive to be, alongside
small ones persisting in places the large ones cannot reach. Each piece below has a real model
behind it in the bibliography (§C predator–prey coevolution, §D niche construction, §B behavioural
ecology).

- **[ ] Large creatures should eat small ones fast, making life rarer.** Partly in (consumption now
  scales with size mismatch), but the population still saturates. Predation should be the main
  regulator of abundance, not an artificial cap.
- **[ ] Gestation should scale with offspring size.** Reproduction is now *priced* by body size, but
  it is still instantaneous. A large animal should also be slow to reproduce — a real gestation
  delay, which is what separates an r-strategist from a K-strategist and is the standard mechanism
  for size-structured population regulation.
- **[ ] Larger bodies must eat proportionally more.** Metabolism already scales with parts and organ
  types; whether it scales *steeply enough* to make being huge a genuine commitment is unmeasured.
- **[ ] Spatial refugia — the key missing mechanism.** Small creatures should survive in caves,
  crevices and quiet zones that large bodies physically cannot enter. Collision is already per-pixel
  against terrain, so a large body genuinely cannot fit through a narrow gap — meaning refugia may
  work *already* if the terrain generator produced fine structure (pockets, crevices, tunnels). It
  currently produces only open water, a sand floor and rock masses. This is the single highest-value
  item here: size-selective refuge is the textbook mechanism allowing predator and prey to coexist
  instead of the predator eating everything, and it produces habitat specialisation for free.
- **[ ] Crowding should be uncomfortable.** Bodies currently overlap freely; creatures stack. Real
  contact forces exist but are evidently too weak to keep bodies apart at high density.
- **[ ] Let specialisation emerge from the above** rather than being scripted: a cave-dwelling small
  grazer and an open-water hunter should be two strategies the same engine produces, not two coded
  creature types.

### Ecology and mechanics
- **[~] Diversity vs. competitive exclusion.** Red Queen pressure is in and calibrated, but the
  diversity/population trade-off is unresolved (§7). Worth exploring whether dispersal, resource
  partitioning, or spatial niches achieve the same end without the population cost.
- **[ ] Active kin defence.** Territory marking exists, but there is no way to sense "a relative is
  under attack right now", which was named as a missing context for realistic aggression.
- **[ ] Generalise niche construction into one primitive.** The single highest-value structural idea
  identified so far. The engine already contains three *unconnected* instances of "organism modifies
  the environment, and that modification changes the fitness landscape": pheromone trails, territory
  scent marks, and sand digging. Treating them as one mechanism would make burrowing, nest building,
  trap making, food caching and shelter construction fall out of the same primitive instead of each
  needing its own bespoke system — which is exactly the shape of open-ended complexity this project
  is after. See bibliography §D.
- **[ ] Bilateral symmetry as a heritable property of a node.** A part carries a symmetry flag; when
  a new component grows on such a node, a mirrored counterpart appears on the same node at the
  reflected angle. This is one of the genuine major body-plan innovations in animal evolution
  (bilateria), and mechanically it is small: `grow_one_pixel` emits a pair instead of a single part
  when the parent node is symmetric. The payoff is large — paired eyes, paired flippers, paired
  tentacles instead of organs scattered at random angles, which is most of what makes a shape read
  as an *animal* rather than a lump. It also makes symmetry itself evolvable: it should win where
  balanced propulsion or stereo sensing pays, and lose where it just doubles the upkeep.
- **[x] Emergent organs** (first pass). Parts differentiate into eye / mouth / gut / tentacle /
  armor / flipper, each with a real function and upkeep, and selection demonstrably acts on the mix.
  Still open: organs that *compose* into higher-order structures, per bibliography §A (Moreno et al.)
  — the current version gives division of labour but not yet anatomy with topology.
- **[ ] Richer weather / seasons / geological change.** Only three weather events exist; a genuinely
  *dynamic* world was requested repeatedly.
- **[ ] Predator/prey coevolutionary cycles** — bibliography §2 and §9 describe long-period cycles
  that this engine has the ingredients for but has not been shown to produce.
- **[ ] Speciation as a first-class concept.** Lineages are currently identified by colour, which is a
  proxy, not a species definition. Reproductive isolation is not modelled.
- **[?] Boom-bust population instability / Allee effect.** Long-standing; partially masked by the
  population cap. Never systematically characterised.
- **[?] ~3% residual rock-collision violations**, mostly from attachment/capture drag not being
  terrain-checked.

### Rendering and presentation
- **[ ] Vision rendering** — depict what creatures actually see (the vision channels exist but are
  invisible).
- **[ ] Distinct per-body-part appearance** — textures or models rather than uniform capsules.
- **[ ] Territory field overlay** — the field is computed and published but deliberately has no UI
  toggle yet, to avoid clutter. Needs a considered design, not another checkbox.
- **[ ] Better lineage visualisation** — a phylogenetic tree rather than a flat table.

### Engineering
- **[ ] Flat-array publish payload.** `individuals_state()` still builds ~6000 Python dicts per
  publish. Returning flat numpy arrays would remove the remaining dominant publish cost, but requires
  changing the frontend's consumption code too.
- **[ ] Automated tests.** There are verification scripts in `scripts/`, but no test suite and no CI.
- **[ ] Open-endedness metrics (MODES).** Measure whether complexity genuinely increases instead of
  assuming that more mechanics means more interesting. Directly relevant given how many mechanics
  have accumulated — see bibliography §10.
- **[ ] Experience-log disk budget.** Rotates at 500 chunks (~450MB) for data nothing reads yet.
- **[ ] Full Verlet/PBD constraint physics** — considered and explicitly deferred; only a lightweight
  joint-angle clamp was taken. Would be a large rewrite.
- **[ ] Continuous performance regression checks.** Optimisation is a standing priority and
  regressions have twice appeared silently as a side effect of unrelated changes.

---


## 8A. Requirements, concerns and ideas from the project owner

Everything the owner has asked for, objected to, or proposed, recorded verbatim in
substance so none of it is lost. Status: **[x]** done · **[~]** partly done · **[ ]** open ·
**[?]** open question.

### The collapse: why complexity appears and then disappears

The single most important finding about this world, and one that invalidated a
lot of earlier reporting. Measured at **6000 ticks** the population looks
healthy: mean 8-12 parts, bodies up to 29, organs at 27-39% of all tissue,
only a few percent unbranched. Measured at **18000-19400 ticks** it is a
different world: **mean 3.5-4.3 parts, 78% of creatures four parts or fewer,
37-53% pure unbranched chains, and organs at 25% -- exactly the rate at which
random differentiation produces them, i.e. selection is not favouring them at
all.**

Complexity emerges and then collapses. Every measurement taken at 6000 ticks
was measuring the emergence and missing the collapse.

The cause is economic. With food regrowing fast enough that a grazed patch
never empties, and a crowded world, a three-part worm that breeds constantly
out-competes anything that invests in a body. And being large was
mathematically unviable: a 20-part predator burns 41 energy per 100 ticks in
upkeep while engulfing a 3-part prey paid 3.1 -- it needed a kill every eight
ticks merely to break even.

Compounding it, six separate size-and-complexity penalties had been added
individually, each defensible alone, without ever checking their combined
effect: per-part reproduction cost, gestation scaling with size, breeding-space
scaling with size, grazing efficiency falling with mass, per-organ metabolic
surcharges, and pathogen pressure. Together they make minimalism optimal.

**Lesson for future work here: measure at 18000+ ticks, not 6000, and check
what a stack of individually-sensible costs does in combination.**

### The central complaint

> *"creatures are still very basic and fail to develop complex structure and behavior, it is
> boring"* · *"natural selection pressure is not high enough, at the end the whole world is
> filled with stacked individuals with no survivability pressure, infinite food, no predation,
> no big predator or whale eating all small organisms"*

This turned out to be correct on every count, and measurement backed each part of it: food was
effectively infinite (mean energy ~96 against a reproduction threshold near 20), bodies were
being actively selected *down* in complexity, half the world was frozen in capture deadlock, and
brains performed no better than random ones.

### Population, pressure and life history

- **[x] Reproduction must not explode.** Cost was a flat fee regardless of offspring size, so
  biomass was conjured from nothing. Now priced per part.
- **[x] Space is a requirement for reproducing.** A crowded patch produces no offspring.
- **[x] Reward on the reproduction act, but it needs space and safety.** Blood in the water
  blocks breeding; reproduction credits a reward into the experience log.
- **[x] Larger bodies should have longer gestation.** Gestation scales with offspring size.
- **[~] Larger bodies must eat more.** Metabolism scales with parts and organ types; grazing
  yield now falls with mass so large animals must hunt. The scaling is now *sublinear* (Kleiber),
  which is the biologically correct shape and was the missing reason large bodies were never
  viable -- see §7.
- **[x] Big creatures should eat many small ones in one sweep.** A sweep budget scaling with
  body mass, plus gape-limited engulfing that swallows small prey whole.
- **[x] Other individuals are food for emergent predators.** 92% of deaths are now predation.
- **[ ] Life should become rarer.** Population self-limits (~600–1400 rather than a 6000 cap)
  but the owner wants fewer, larger animals still.
- **[ ] Creatures should not stack on each other.** Contact now scales with real part size, but
  crowding at high density is not solved.
- **[x] Are reproductive anomalies (mutations) well diversified?** Investigated and the answer is
  yes. Counting distinct body TOPOLOGIES (branching profile, depth profile and organ composition,
  so trivial coordinate differences don't inflate the count): roughly **three quarters of creatures
  have a unique body plan**, distinct plans rise from 393 to 759 over 6000 ticks, and no single
  plan ever exceeds ~2.5% of the population. Morphology is genuinely exploring; mutation is not
  the bottleneck.

### Bodies and structure

- **[x] More basic components.** Eight part types: body, eye, mouth, gut, tentacle, armor, flipper,
  and a filtering mesh. The filter is the interesting one: grazing yield deliberately falls away as
  a body gets heavier, so large animals are forced to hunt and a real food chain exists rather than
  one undifferentiated crowd. That rule had no exception, and nature's most conspicuous exception is
  exactly the animal it forbids -- the enormous filter feeder living on the smallest food in the
  ocean, which is also the "whale eating all small organisms" that was asked for. Filter tissue
  raises the mass at which grazing stops paying, in proportion to the filtering surface carried, so
  there are now two ways to be large instead of one. It is not free: a mesh is a broad face held into
  the flow and drags like one, which is why a filter feeder is slow, and it banks no energy.
- **[x] Bilateral symmetry as a property of a node.** Growing a component on a symmetric node
  emits a mirrored twin at the reflected angle, with mirrored hinge limits. The trait was
  inherited but could not be *expressed*: only a lateral growth can pair, since a direction lying
  on the body axis is its own mirror, yet tip extension runs along exactly that axis. So 35% of
  founders carried the trait and only ~12% of bodies ever showed a pair. Symmetric nodes now grow
  sideways by preference. Measured live: **32.9% of bodies carry a mirrored pair, 45.2% among
  those with six or more parts.**
- **[x] Bounding boxes should apply to all components.** Terrain collision uses each part's own
  radius rather than treating parts as dimensionless points.
- **[~] Organs should be visually distinguishable, not just differently coloured.** Each organ kind
  now has a characteristic girth -- a belly bulges, armour is a slab, an eye is a small lens, a
  tentacle is thin -- and that girth drives collision footprint and hit radius as well as the drawn
  radius, so what is on screen is what the physics uses. Applied to width only: one scalar cannot
  express "long and thin", and folding it into segment length would make tentacles stubby.
- **[ ] Creatures are essentially worms or very basic structures.** Topologically they are *not*
  worms (78% branch, ~1.2 branch points per body), but growth still weights "extend an existing
  tip in the same direction" at **8x** everything else, which biases hard toward elongation. That
  weight is a prime suspect and is **not yet changed**.
- **[ ] Emergent organs that compose into higher-order structures.** Division of labour exists;
  anatomy with real topology does not.

### Motion physics

- **[x] Motion should be a consequence of component movement, not the reverse — the creature must
  learn to use its own body.** This was the deepest architectural correction of the session.
  Heading used to be a variable the brain simply *assigned*, with the body rotated to match.
  Now the brain holds a body curvature, the curved body pushes water asymmetrically, and the
  resulting **torque** rotates it. Rotation is integrated from fluid forces, never assigned.
- **[x] Bodies need a real anterior axis.** `heading` merely rotated whatever shape a creature
  grew into, so the offset between "pointing" and "moving" differed per individual and pooled to
  look like noise. Each body now has an axis from root to centre of mass.
- **[x] "They swim with the tail frontward."** Diagnosed and addressed -- see §7. Thrust is
  perfectly locked to a given body (R=1.000 across headings), but the offset between pointing and
  pushing varies wildly between bodies (R=0.259 over fifteen plans, several pushing almost exactly
  backwards), and animals had no sense channel with which to notice. Proprioception added; the
  physics deliberately left alone.
- **[~] Superseded note on the same problem:** The measured reality was worse than tail-first: the
  angle between heading and actual movement averaged ~91°, i.e. **statistically random**.
  Torque-driven turning plus the anterior axis improved the fastest swimmers to ~58°, but with
  thermal noise removed entirely alignment still sits near 81°. **Propulsion is still not
  reliably axis-locked — this is unresolved and is the most important open physics problem.**
- **[ ] Animals should swim using fins — lateral and rear propulsion.** Flippers exist as an
  organ that multiplies thrust, but propulsion is still whole-body undulation; fins do not
  generate directional thrust from their own motion.

### Environment

- **[x] Rocks should be in patches near the ground, not floating everywhere.** Rock formations are
  now anchored to the seafloor and rise from it within a reef band.
- **[x] Caves are welcome.** Cave networks are carved through the rock by drifting random walks,
  plus eroded crevice pockets.
- **[x] Collision with rock is not working.** Three separate leaks, now fixed; embedded parts fell
  from 7.8–11.6% to 0.24–1.11%.
- **[x] Small creatures should survive in caves and quiet places big ones cannot reach.** Verified:
  small bodies occupy positions with ~80% more surrounding rock than large ones, purely from
  geometry.
- **[ ] More dynamism and diversity in the world and its components.** Day/night, weather and the
  reef exist; the world is still largely static over time.

### Intelligence

- **[x] Improve intelligence with a common brain part trained on the RTX 6000, bigger networks,
  trained asynchronously from experience replay, so the simulation keeps running at full speed.**
  Built and closed end to end.
- **[x] The brain should be composable; a new sensory component means a new input.** Vision is
  organ-gated — no eye, no visual input.
- **[x] A shared latent space of the surroundings, composed from sensors; the encoder may be
  shared, but each decision is per individual.** Exactly the implemented architecture.
- **[x] The neural computation is heavy and must be asynchronous.** Training runs in its own
  process; the simulation never waits.
- **[~] "Intelligence sucks."** Correct when raised, and only partly addressed. Measured: evolved
  brains were no better than random. After the shared encoder and inherited instinct,
  prey-pursuit is meaningfully better (positive in 4/4 seeds), but fleeing and mate-approach
  still show **no learning at all**, and the effect sizes are small.
- **[ ] Bigger networks.** Still a 12-unit hidden layer.
- **[?] Do creatures genuinely learn to swim?** Not demonstrated. See the motion physics section:
  until propulsion is reliably axis-locked, "learning to swim" cannot be claimed.

### Feeding, collision and predation (raised while watching the live world)

- **[x] "It is weird that they can only turn around their head."** Correct, and it was literally
  that: fluid torque was taken about the root and forward kinematics rebuilt the body from the root
  each tick, so every animal swung its whole mass around its nose. Torque is now taken about the
  **centre of mass**, the root is carried around that point as the heading changes, and rotational
  inertia is the real second moment rather than bare mass -- so a long body is genuinely sluggish
  to turn and a compact one is nimble.
- **[x] Eating requires a mouth to be touching the target, and the target part must be smaller
  than the mouth.** Both now enforced. An animal with no mouth cannot eat another animal at all,
  which is what finally makes a mouth worth its upkeep; a gape can only take a part smaller than
  itself. Measured effect: mouths rose from 2.1% to 3.9% of all tissue, and 47.7% of animals carry
  one.
- **[x] Biting a non-leaf part slices the body, leaving dead matter.** The severed limb now falls
  away as carrion at the point it was cut off, with real energy in it. Previously that flesh simply
  vanished.
- **[x] Armour cannot be cut.** A plated part turns a bite outright. It is not blanket
  invulnerability -- blunt combat damage still goes through the normal armour arithmetic -- so
  armour makes an animal tough rather than immortal.
- **[x] A creature should be able to collect many small creatures and bring them to its mouth.**
  Tentacles now sweep smaller animals toward the nearest mouth on the same body, with a reaction
  force on the hauler, so gathering a crowd and working through it is possible and the placement of
  tentacles relative to the mouth is worth evolving. Costs nothing for bodies lacking both organs.
- **[x] Upkeep should scale with surface (in 2D).** Metabolism is charged on body **area** --
  girth squared per part, weighted by tissue type -- rather than on a part count, so an animal
  built from armour slabs is genuinely more expensive than one of the same part count built from
  slender tentacles. See §7 for why the exponent on that area is not 1.0.
- **[x] Less food, but more storable energy; energy stock is a reward.** Food regrowth cut to
  0.009. Energy is now bounded by a storage capacity built from evolved storage tissue and gut,
  weighted by part area -- it was previously unbounded, and an animal was observed sitting on 898
  units of free organ-less buffer. A small per-tick reward tracks how full that larder is, kept far
  below the reproduction reward because rewarding energy directly was previously measured making
  animals hoard instead of breed.
- **[ ] Use Jolt as the physics engine for realism.** Not taken, and worth stating why rather than
  silently skipping: Jolt is a 3D rigid-body engine, while these animals are kinematic chains whose
  motion is *generated* by an undulation wave passing down the body and resolved against fluid drag.
  Handing that to a rigid-body solver would replace the exact mechanism that makes them swim, and
  the thing actually being asked for -- collision between all components -- already exists and has
  been extended: every part collides by its own girth against terrain, against other animals, and
  now for feeding as well. If a real solver is ever wanted here, the candidate is `rapier2d` (native
  Rust, genuinely 2D), not Jolt.

### Standing instructions

- Keep iterating autonomously; do not stop.
- Periodically research real biology and ecology, and mine the bibliography for mechanisms.
- Watch performance continuously, and keep the world legible rather than an unreadable mess.
- Verify with measurement before claiming anything works.

## 9. Development notes

**Debugging the frontend.** Playwright driving the *real installed Chrome*
(`chromium.launch({ channel: 'chrome', headless: true })`) gives a genuine autonomous loop: console
and `pageerror` capture, `page.evaluate()` for direct JS state inspection, screenshots. This is how
the WGSL shader bug was found after code review missed it.

One caveat: the species table re-renders every ~80ms poll, so Playwright's actionability wait
("element is detached from the DOM, retrying") never settles and `locator.click()` times out. Use
`page.evaluate()` with a dispatched `MouseEvent` to test the state transition directly — a
testing-tool limitation, not a real bug at human click speeds.

**Verification discipline.** Every mechanic here was verified with a purpose-built `debug_*` hook and
a standalone script against a fresh `World`, or with a real browser session, before being claimed to
work. Several mechanics only survived because the first measurement contradicted the expectation and
got investigated rather than explained away.

The most useful pattern is the **A/B harness**: identical seeds, one variable changed through a
runtime `debug_set_*` override — because a compile-time constant cannot be compared against itself in
a single run. Equally important is **choosing the right metric**: raw lineage count said the diversity
mechanic was failing; effective species count (inverse Simpson) showed it working. See `scripts/`.

---

## 10. Bibliography

### Consulted during design

**Territoriality and home-range formation** — basis for the territory/scent-marking mechanic.
- [Territorial Dynamics and Stable Home Range Formation for Central Place Foragers](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0034033) — PLOS One
- [A mechanistic, stigmergy model of territory formation in solitary animals](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289346/)
- [Home range formation in wolves due to scent marking](https://link.springer.com/article/10.1006/bulm.2001.0273) — Bulletin of Mathematical Biology
- [The integrated role of resource memory and scent-based territoriality in the emergence of home-ranges](https://www.biorxiv.org/content/10.1101/2021.05.07.443202.full.pdf)
- [How memory of direct animal interactions can lead to territorial pattern formation](https://royalsocietypublishing.org/rsif/article/13/118/20160059/64678/) — J. R. Soc. Interface
- [Territoriality modulates the effect of conspecific encounters on foraging behaviours of a mammalian predator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11885166/)

**Janzen–Connell and density-dependent diversity maintenance** — basis for the diversity work in §7.
- Bagchi et al. (2010), [Testing the Janzen–Connell mechanism: pathogens cause overcompensating density dependence in a tropical tree](https://pubmed.ncbi.nlm.nih.gov/20718845/) — Ecology Letters
- Comita et al. (2014), [Testing predictions of the Janzen–Connell hypothesis: a meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4140603/)
- [Pathogen regulation of plant diversity via effective specialization](https://pubmed.ncbi.nlm.nih.gov/24091206/)
- [Closing the gap in the Janzen–Connell hypothesis: what determines pathogen diversity?](https://onlinelibrary.wiley.com/doi/abs/10.1111/ele.14316) — Ecology Letters 2024
- [Contribution of conspecific negative density dependence to species diversity](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8455644/)

**Metabolic scaling** — basis for the Kleiber discount that makes large bodies viable (§7).
- Kleiber, M. (1932), *Body size and metabolism*, Hilgardia 6:315-353 — the original observation that
  metabolic rate scales as mass^(3/4) rather than in proportion to mass.
- West, G. B., Brown, J. H. & Enquist, B. J. (1997), [A general model for the origin of allometric
  scaling laws in biology](https://www.science.org/doi/10.1126/science.276.5309.122) — Science. Derives
  the 3/4 exponent from the geometry of resource-distribution networks.
- Brown, J. H. et al. (2004), [Toward a metabolic theory of ecology](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/03-9000)
  — Ecology. Metabolic rate as the pacemaker for growth, reproduction and population dynamics.

**Alternative stable states and priority effects** — the framing for the seed-bistability result (§7),
where identical parameters produce either a worm world or a large-bodied one.
- Beisner, B. E., Haydon, D. T. & Cuddington, K. (2003), [Alternative stable states in ecology](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/1540-9295%282003%29001%5B0376%3AASSIE%5D2.0.CO%3B2)
  — Frontiers in Ecology and the Environment.
- Fukami, T. (2015), [Historical contingency in community assembly: integrating niches, species pools,
  and priority effects](https://www.annualreviews.org/doi/10.1146/annurev-ecolsys-110411-160340)
  — Annual Review of Ecology, Evolution, and Systematics.
- Scheffer, M. et al. (2001), [Catastrophic shifts in ecosystems](https://www.nature.com/articles/35098000)
  — Nature.

**Chemoreception vs vision** — basis for making smell short-range and sight long-range (§7).
- Atema, J. (1995), [Chemical signals in the marine environment: dispersal, detection, and temporal
  signal analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC40010/) — PNAS. Odour plumes are diffuse,
  intermittent and give poor directional information compared with vision.

**Physics / rendering**
- [zalo — Constraints (Verlet / PBD)](https://zalo.github.io/blog/constraints/) — considered as a full
  physics rewrite, deliberately out of scope; only a lightweight joint-angle clamp was taken.

**Consulted and rejected**
- [PMC6353876](https://pmc.ncbi.nlm.nih.gov/articles/PMC6353876/) — supplied as inspiration, but it is
  a computational-chemistry paper on measuring molecular complexity via fractal dimension. No
  behavioural, ecological or evolutionary content. Recorded so it is not re-checked later.

**Concept-level sources** (looked up and adapted rather than cited from one paper): quorum sensing,
crypsis and industrial melanism, aposematism, stigmergy, sea-anemone sessility and ambush predation,
resistive force theory, Hamilton's rule and inclusive fitness, the Allee effect, competitive
exclusion, and negative frequency-dependent (Red Queen) selection.

### Reading list for future mechanics

Curated for finding **transposable mechanisms**, not general explanations or code. Organised by the
five research domains that matter most for this simulation. A good starting path through it:
Connelly (pheromones) → Ito (predator–prey) → Framsticks → Moreno (multicellularity) → MODES.

#### A. Evo-devo and morphological development
*Emergent organs, limbs, mouths, digestive systems, segmentation, juvenile vs adult forms.*

- *Morphological Development at the Evolutionary Timescale: Robotic Developmental Evolution*
  (Artificial Life, 2022) — separates genome evolution from **body development**, with part-composed
  structures and evolvable muscles. [MIT Press](https://direct.mit.edu/artl/article/28/1/3/109958/)
- *How morphological development can guide evolution* (Scientific Reports, 2018) — organisms that
  change morphology *while behaving*; directly relevant to juvenile → adult body plans, which this
  engine currently reduces to uniform inflation of a birth-fixed plan.
  [Nature](https://www.nature.com/articles/s41598-018-31868-7)
- Danca et al. (2015), *How morphology of artificial organisms influences their evolution* — avoids
  the "genome → a few statistics → fitness" shortcut; locomotion, foraging and competition all depend
  on the morphology the genome produces.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1476945X15001014)
- Silveira & Massad (1998), *Modeling and Simulating Morphological Evolution in an Artificial Life
  Environment* — resource distribution exerting indirect selection on morphology.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/9561807/)
- *Guideless Artificial Life Model for Reproduction, Development, and Interactions* (Artificial Life,
  2025) — treats reproduction and development as evolvable processes rather than single events.
  [MIT Press](https://direct.mit.edu/artl/article/31/1/31/127798/)
- *Emergence of Organisms* — theoretical treatment of how proto-cells, multicellular organisms and
  increasingly complex organisation can emerge rather than being explicitly represented.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7597334/)
- Moreno et al. (2022), *Exploring Evolved Multicellular Life Histories in an Open-Ended Digital
  Evolution System* — observed emergence of division of labour, resource sharing, offspring
  investment, cell–cell communication, morphological patterning, adaptive apoptosis, and transitions
  to multicellular individuality. The most promising route to making functional **organs** emerge
  instead of hardcoding "stomach = X".
  [Frontiers](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2022.750837/full)

#### B. Behavioural ecology
*Territoriality, mating strategies, parental care, offspring protection, dominance, cooperation.*

- [Territorial Dynamics and Stable Home Range Formation for Central Place Foragers](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0034033) — PLOS One
- [A mechanistic, stigmergy model of territory formation in solitary animals](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289346/)
- [Home range formation in wolves due to scent marking](https://link.springer.com/article/10.1006/bulm.2001.0273) — Bulletin of Mathematical Biology
- [The integrated role of resource memory and scent-based territoriality in the emergence of home-ranges](https://www.biorxiv.org/content/10.1101/2021.05.07.443202.full.pdf)
- [How memory of direct animal interactions can lead to territorial pattern formation](https://royalsocietypublishing.org/rsif/article/13/118/20160059/64678/) — J. R. Soc. Interface
- [Territoriality modulates the effect of conspecific encounters on foraging behaviours of a mammalian predator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11885166/)
- Cairns et al. (2020), *Evolution in interacting species alters predator life-history traits,
  behaviour and morphology in experimental microbial communities* — ~600 generations of real
  experimental evolution; predators evolve changes in **size, speed and movement directionality**.
  A source of plausible selection pressures rather than invented fitness functions.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341940/)

#### C. Predator–prey coevolution
*Hunting, ambush, baiting, camouflage, pursuit, escape, defensive morphology.*

- Ito et al., *Population and Evolutionary Dynamics based on Predator-Prey Relationships in a 3D
  Physical Simulation* — evolvable morphology *and* behaviour in a physical world; observes long
  evolutionary cycles from coevolving defensive strategies.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/26934093/)
- Ito, Pilat, Suzuki & Arita (ECAL 2015), *Evolutionary change precedes extinction in
  eco-evolutionary dynamics based on a 3D virtual predator-prey system*.
  [Link](https://www.cs.york.ac.uk/nature/ecal2015/paper-147.html)
- Craig Reynolds (2025), *Camouflage From Coevolution of Predator and Prey* — prey evolve camouflage
  while predators evolve perception **and learn within their lifetime**. The closest published
  precedent for the planned GPU-trained brain: lifetime learning alongside evolution.
  [MIT Press](https://direct.mit.edu/artl/article/31/2/153/130573/)
- *Evolution of Swarming Behavior Is Shaped by How Predators Attack* (Artificial Life, 2016) — *how*
  a predator attacks determines which prey behaviours evolve; grouping, selfish-herd effects and
  collective defence. [MIT Press](https://direct.mit.edu/artl/article/22/3/299/2845/)
- *Complex eco-evolutionary dynamics induced by the coevolution of predator–prey movement strategies*
  (2021). [Springer](https://doi.org/10.1007/s10682-021-10140-x)
- *Hidden paths to endless forms most wonderful: ecology latently shapes evolution of multicellular
  development in predatory bacteria* (2022) — prey type and environment alter developmental and
  morphological evolution **even when those traits are not directly selected**. A caution against
  assuming fitness must directly select morphology.
  [Nature Comms Biology](https://www.nature.com/articles/s42003-022-03912-w)

#### D. Niche construction and environmental modification
*Digging, burrowing, nest building, dams, traps, territorial marking, food caches, shelters.*

**This is the highest-value direction for this engine.** If organisms can modify the environment,
then digging, nest building, trap making, territory marking and caching need not be separate
programmed behaviours. They all reduce to one primitive:

> organism modifies environment → environment changes the fitness landscape → selection favours
> organisms that exploit that modification.

The engine already contains three unconnected instances of this primitive — pheromone trails,
territory scent marks, and sand digging — without ever having treated them as one mechanism.
Generalising it is probably the single most promising route to open-ended complexity here.

- *What Is Artificial Life Today, and Where Should It Go?* (Artificial Life, 2024) — environmental
  construction and modification, ecological niches, emergent ecosystem interactions, and eusocial
  nest construction as complexity emerging from local interactions.
  [MIT Press](https://direct.mit.edu/artl/article/30/1/1/120293/)

#### E. Collective behaviour and stigmergy
*Colonies, swarms, coordinated hunting, collective defence, communication without central control.*

- Connelly, McKinley & Beckmann (2009), *Evolving Cooperative Pheromone Usage in Digital Organisms* —
  digital organisms evolve pheromone use to coordinate movement; cooperative strategies **emerge**
  rather than being scripted. The most directly relevant single paper for this engine's chemical
  fields. [PDF](https://citeseerx.ist.psu.edu/document?doi=34929424598cf8b708f158311062d8db28e8c4fa&repid=rep1&type=pdf)
- Search terms: *digital organisms pheromone*, *evolved chemical communication*, *evolutionary
  stigmergy*, *pheromone-based collective behavior artificial life*.

#### F. Whole systems, evolved behaviour, and open-endedness

- **Framsticks** — genotype → physical morphology + control system, with evolution, coevolution,
  multiple populations, species and ecosystems. One of the most relevant complete architectures.
  [framsticks.com](https://www.framsticks.com/) ·
  [Komosinski & Ulatowski (1998)](https://www.framsticks.com/files/common/Komosinski_Framsticks_ECML1998.pdf) ·
  [resources](https://www.framsticks.com/node/343)
- *The Surprising Creativity of Digital Evolution* (Artificial Life, 2020) — anecdote collection;
  Tierra alone produced parasitism, immunity, hyperparasitism, cheating and obligate sociality.
  Useful both as inspiration and as a warning about how readily evolution exploits engine bugs.
  [MIT Press](https://direct.mit.edu/artl/article/26/2/274/93255/)
- *A Case Study of the De Novo Evolution of a Complex Odometric Behavior in Digital Organisms* —
  digital organisms evolved genuine internal odometry that nobody programmed.
  [PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0060466)
- *Evolutionary Developmental Robotics: Improving Morphology and Control of Physical Robots*
  (Artificial Life, 2017) — different body morphologies produce distinct emergent gaits.
  [MIT Press](https://direct.mit.edu/artl/article/23/2/169/2866/)
- *Evolutionary Robotics* — ALife encyclopedia entry; a good entry point into evolved locomotion and
  morphology. [alife.org](https://alife.org/encyclopedia/introduction/evolutionary-robotics/)
- *Evolutionary Robotics and Morphological Design* — overview of jointly evolving morphology and
  behaviour. [Nature Index](https://www.nature.com/nature-index/topics/l4/evolutionary-robotics-and-morphological-design)
- *The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems* (2019) — metrics for
  novelty, complexity, ecological and change potential. For actually *measuring* whether the
  simulation grows more complex rather than merely accumulating features.
  [MIT Press](https://direct.mit.edu/artl/article/25/1/50/2915/)
- *Open-Endedness for the Sake of Open-Endedness* (2019) — why piling on more mechanisms does **not**
  necessarily produce open-ended evolution. Worth taking seriously here.
  [MIT Press](https://direct.mit.edu/artl/article/25/2/198/2923/)

> Do not limit the search to ALife systems. Evolutionary robotics, evo-devo, behavioural ecology,
> theoretical ecology, experimental predator/prey evolution, transitions in individuality and
> microbial evolution have all studied these mechanisms in isolation, often in more transposable form.
