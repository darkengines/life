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
python -m uvicorn live_app:app --host 127.0.0.1 --port 8002
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

### Ecology and mechanics
- **[~] Diversity vs. competitive exclusion.** Red Queen pressure is in and calibrated, but the
  diversity/population trade-off is unresolved (§7). Worth exploring whether dispersal, resource
  partitioning, or spatial niches achieve the same end without the population cost.
- **[ ] Active kin defence.** Territory marking exists, but there is no way to sense "a relative is
  under attack right now", which was named as a missing context for realistic aggression.
- **[ ] Emergent organs.** The evolved `storage` trait (a de-facto belly/mouth anchor) hints at what
  is possible. Making functional organs *emerge* rather than hardcoding "stomach = X" is the most
  promising direction in the bibliography (§6, Moreno et al.).
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

Curated for finding **transposable mechanisms**, not general explanations or code.
*Suggested order: 1 → 2 → 4 → 6 → 5 → 3 → 9 → 10.*

**1. Evolved chemical communication / stigmergy**
- Connelly, McKinley & Beckmann (2009), *Evolving Cooperative Pheromone Usage in Digital Organisms* —
  digital organisms evolve pheromone use to coordinate movement; cooperative strategies **emerge**
  rather than being scripted. [PDF](https://citeseerx.ist.psu.edu/document?doi=34929424598cf8b708f158311062d8db28e8c4fa&repid=rep1&type=pdf)
- Search terms: *digital organisms pheromone*, *evolved chemical communication*, *evolutionary
  stigmergy*, *pheromone-based collective behavior artificial life*.

**2. Morphology ↔ behaviour ↔ selection**
- Ito et al., *Population and Evolutionary Dynamics based on Predator-Prey Relationships in a 3D
  Physical Simulation* — evolvable morphology *and* behaviour in a physical world; observes long
  evolutionary cycles from coevolving defensive strategies. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26934093/)
- Ito, Pilat, Suzuki & Arita (ECAL 2015), *Evolutionary change precedes extinction in eco-evolutionary
  dynamics based on a 3D virtual predator-prey system*. [Link](https://www.cs.york.ac.uk/nature/ecal2015/paper-147.html)

**3. Morphology as a real ecological factor**
- Danca et al. (2015), *How morphology of artificial organisms influences their evolution* — avoids the
  "genome → a few statistics → fitness" shortcut; locomotion, foraging and competition all depend on
  the morphology the genome produces. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1476945X15001014)

**4. Framsticks** — genotype → physical morphology + control system, with evolution, coevolution,
multiple populations, species and ecosystems. One of the most relevant complete architectures.
- [framsticks.com](https://www.framsticks.com/) ·
  [Komosinski & Ulatowski (1998)](https://www.framsticks.com/files/common/Komosinski_Framsticks_ECML1998.pdf) ·
  [resources](https://www.framsticks.com/node/343)

**5. Morphological evolution and development**
- Silveira & Massad (1998), *Modeling and Simulating Morphological Evolution in an Artificial Life
  Environment* — resource distribution exerts indirect selection on morphology. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9561807/)
- *Morphological Development at the Evolutionary Timescale: Robotic Developmental Evolution*
  (Artificial Life, 2022) — separates genome evolution from **body development**, with part-composed
  structures and evolvable muscles. [MIT Press](https://direct.mit.edu/artl/article/28/1/3/109958/)

**6. Multicellularity and the emergence of organs**
- Moreno et al. (2022), *Exploring Evolved Multicellular Life Histories in an Open-Ended Digital
  Evolution System* — observed emergence of division of labour, resource sharing, offspring
  investment, cell–cell communication, morphological patterning, adaptive apoptosis, and transitions
  to multicellular individuality. [Frontiers](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2022.750837/full)
- The most promising direction for making functional **organs** emerge instead of hardcoding
  "stomach = X".

**7. Predation → morphological evolution (real biology, for plausible selection pressures)**
- Cairns et al. (2020), *Evolution in interacting species alters predator life-history traits,
  behaviour and morphology in experimental microbial communities* — ~600 generations; predators evolve
  changes in **size, speed and movement directionality**. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341940/)

**8. Predation ↔ development ↔ morphology**
- *Hidden paths to endless forms most wonderful: ecology latently shapes evolution of multicellular
  development in predatory bacteria* (2022) — prey type and environment alter developmental and
  morphological evolution **even when those traits are not directly selected**. A caution against
  assuming fitness must directly select morphology. [Nature Comms Biology](https://www.nature.com/articles/s42003-022-03912-w)

**9. Predator/prey coevolution**
- *Complex eco-evolutionary dynamics induced by the coevolution of predator–prey movement strategies*
  (2021). [Springer](https://doi.org/10.1007/s10682-021-10140-x)
- Craig Reynolds (2025), *Camouflage From Coevolution of Predator and Prey* — prey evolve camouflage
  while predators evolve perception **and learn within their lifetime**. Directly relevant to the
  planned GPU-trained brain, which is exactly lifetime learning alongside evolution. [MIT Press](https://direct.mit.edu/artl/article/31/2/153/130573/)

**10. Open-ended evolution**
- *The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems* (2019) — metrics for
  novelty, complexity, ecological and change potential. Useful for actually *measuring* whether the
  simulation grows more complex rather than merely accumulating features. [MIT Press](https://direct.mit.edu/artl/article/25/1/50/2915/)
- *Open-Endedness for the Sake of Open-Endedness* (2019) — why piling on more mechanisms does **not**
  necessarily produce open-ended evolution. [MIT Press](https://direct.mit.edu/artl/article/25/2/198/2923/)

> Guidance attached to this list: do not limit the search to ALife systems. Eco-evolution,
> experimental predator/prey evolution, stigmergy, evo-devo, transitions in individuality, and
> microbial evolution are probably richer sources of transposable mechanisms.
