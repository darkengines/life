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
  yield now falls with mass so large animals must hunt. Whether the scaling is *steep* enough
  is unverified.
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

- **[x] More basic components.** Seven part types: body, eye, mouth, gut, tentacle, armor, flipper.
- **[x] Bilateral symmetry as a property of a node.** Growing a component on a symmetric node
  emits a mirrored twin at the reflected angle, with mirrored hinge limits.
- **[x] Bounding boxes should apply to all components.** Terrain collision uses each part's own
  radius rather than treating parts as dimensionless points.
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
- **[~] "They swim with the tail frontward."** The measured reality was worse than tail-first: the
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
