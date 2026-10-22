# Self-Driving Car — Genetic Algorithm: Project Documentation

A browser simulation in which a population of cars, each controlled by a small
neural network, learns to drive through traffic and obey traffic lights. No
driving rules are programmed and no car is ever told the right thing to do.
The network weights are found by a **genetic algorithm**: the cars that drive
best breed the next generation, and driving behaviour emerges over time.

Written in plain JavaScript, HTML and CSS using the browser's Canvas 2D API.
No frameworks, no libraries, no build step.

---

## Contents

1. [Purpose](#1-purpose)
2. [Quick start](#2-quick-start)
3. [What is on the screen](#3-what-is-on-the-screen)
4. [System overview](#4-system-overview)
5. [The simulation world](#5-the-simulation-world)
6. [The neural network (the genome)](#6-the-neural-network-the-genome)
7. [The genetic algorithm](#7-the-genetic-algorithm)
8. [Staged training (curriculum)](#8-staged-training-curriculum)
9. [Demo mode: a frozen champion](#9-demo-mode-a-frozen-champion)
10. [Benchmarks and CSV export](#10-benchmarks-and-csv-export)
11. [Parameter reference](#11-parameter-reference)
12. [File-by-file reference](#12-file-by-file-reference)
13. [Testing](#13-testing)
14. [Engineering history: problems found and fixed](#14-engineering-history-problems-found-and-fixed)
15. [Measured results](#15-measured-results)
16. [Limitations and honest framing](#16-limitations-and-honest-framing)
17. [Presenting the project](#17-presenting-the-project)
18. [Glossary](#18-glossary)
19. [Provenance](#19-provenance)

---

## 1. Purpose

The project demonstrates **evolutionary optimisation of a neural network** on a
problem where supervised learning is not possible.

There is no dataset of correct steering decisions and no way to compute an
error gradient: nobody knows what a car *should* have done at frame 412. The
only feedback available is a single score per car at the end of its run. That
rules out backpropagation and makes the problem a natural fit for a genetic
algorithm, which needs nothing but a way to score candidates.

So the project is really about four things:

- **Representation** — a driver's entire behaviour is encoded as 124 numbers
  (the weights and biases of a 10 → 8 → 4 network), which makes it a
  chromosome.
- **Fitness** — how to turn "drove well" into a number that evolution can
  climb, and how easy it is to write one that rewards the wrong thing.
- **Evolution** — selection, crossover, mutation and elitism operating on a
  population of 100 such chromosomes, in three training stages, with the
  whole process made visible on screen so it can be explained by
  demonstration.
- **Evidence** — a frozen-brain demo, benchmarks against untrained networks,
  a mutation-rate comparison, an unseen-track evaluation and a CSV export,
  so the claim "it learned" can be checked rather than taken on trust.

---

## 2. Quick start

### Run

Any static HTTP server will do; `file://` is not reliable because the page uses
`localStorage` and loads images.

```bash
python3 -m http.server 8000      # or: npm start
# open http://localhost:8000
```

### Test

Node.js is required; there are no dependencies.

```bash
npm test
```

### Controls

| Control | Effect |
| --- | --- |
| **Train Population** | Training mode (the default). Also resumes training after a demo or a benchmark |
| **Demo Best Brain** | Loads the saved champion and drives it, frozen, on the fixed demo route (§9) |
| **Reset Demo** | Restarts the identical demo route from the start line |
| **Pause / Resume** | Freezes the simulation *and* the breeding panel |
| **×1 / ×5 / ×20** | Simulation speed. ×20 runs twenty simulation steps per drawn frame; lights still cycle at the correct simulated rate |
| **Save Best Brain** | Writes the current generation's best brain to `localStorage` by hand (saving is normally automatic — §7.8) |
| **Load Saved Brain** | Restarts training seeded from the saved champion, on its track, in its stage |
| **Discard & Restart** | Wipes the saved brain, history and track; generation 1 from random brains on a new track |
| **GA parameters → Apply** | Population, mutation rate/strength, tournament size, elite count, stage (auto / A / B / C). Values are validated and clamped; the active configuration is displayed; changes take effect from the next generation |
| **Benchmarks → Run Evaluation** | Experiment A, B, C or all three (§10) |
| **Export Results (CSV)** | Downloads the training history and every stored benchmark row |
| **Clear stored results** | Deletes stored benchmark rows, after confirmation |

A good demonstration order: ×20 for a few minutes of training; Pause on a
breeding panel to walk through the algorithm; **Demo Best Brain** at ×1 to show
the champion drive, stop at the red light and go on green; **Run Evaluation**
to show the numbers.

---

## 3. What is on the screen

The page has four columns. A **mode banner** at the top of the left column
says which mode is active: `TRAINING — population evolving`, `DEMO — frozen
champion, no evolution`, or `BENCHMARK — evaluating, training paused`. The
road canvas carries the same banner.

### Left column — statistics, metrics and the fitness chart

**All-time Best Distance** — the furthest any car has driven in training.

**Info panel** (training): generation and stage, generation time, active /
alive cars, the camera car's speed and *current action* (e.g. `ACCELERATE +
STEER LEFT`, `BRAKE`), the champion's distance / overtakes / legal stops /
lights, its fitness and the equation behind it, and what the curriculum is
waiting for before promoting the population.

**Info panel** (demo): the champion's provenance, route time and progress,
speed, the five sensor readings, the nearest light's colour and distance, the
action chosen this frame, overtakes / legal stops / lights cleared, collision
status, the **SUCCESS / FAILURE** verdict, and a log of events (overtook car 2,
stopped at red light, light turned green, cleared light 1...).

**Population metrics** table — live values for the current generation beside
the all-time best: best, mean and median fitness; mean distance; mean
overtakes; collision rate; legal-stop rate (legal stops ÷ red lights met);
cars clearing a light (cars that cleared one ÷ cars that met one); genome
diversity (mean per-gene standard deviation; ~0.58 for a random population,
0 for clones). A rate whose denominator is zero is shown as `n/a`, never as 0.

**Fitness by generation** chart — **yellow** best, **blue** mean, **green**
median, one point per generation, persisted across reloads. Orange dashed
lines mark stage changes (fitness before and after a stage change is not
comparable: it is a different track). The best line alone proves little —
elitism makes it non-decreasing by construction — so the mean and median are
drawn with it: those only rise when the whole population is getting better.

### Road canvas

| What you see | What it is |
| --- | --- |
| One **solid blue car with yellow lines** | The leader: the highest-scoring car still active. The camera follows it. The yellow lines are its five sensors |
| Many **faint, translucent blue cars** | The rest of the population (20% opacity) |
| **Brightly coloured cars** | Traffic. No brain; they drive straight at a slow fixed speed, follow the car ahead, and stop for red lights. They are walls, not competitors |
| **Faint green shaded lane** beside a wall | The open lane — the route past that wall |
| **Dark ✕ marks** | Where a car crashed this generation |
| **Traffic light** with a coloured lamp | Green, yellow or red. When red, a **red bar across the road** is drawn; crossing it destroys the car |
| **Chequered FINISH bar** | The end of the demo route (demo mode only) |

### Controls column

Mode, simulation, saved-brain, GA-parameter and benchmark groups, and a
status line that reports every state change (champion saved, stage promoted,
parameters corrected, benchmark finished, why a demo could not start).

### Right column — the brain

The leader's (or, in demo mode, the frozen champion's) network live. Circles
are neurons, lines are weights: **yellow = positive, blue = negative, opacity
= magnitude**. Inputs are labelled along the bottom:

```
L45  L22  FWD  R22  R45  SPD  RED  YEL  GRN  DIST
```

Outputs are labelled `ACC LEFT RIGHT BRAKE`; the output(s) the car actually
acted on this frame after arbitration carry a **green ring**.

Between generations this canvas shows the **breeding panel** for about three
seconds: the ranked top eight with their fitness terms, the elites in green,
four real parent pairs from the tournaments, the mutation rate used, and
**one child's 124 genes** as a grid coloured by parent with mutated genes
outlined in red. If the stagnation escape fired, the panel says so in orange.
While a benchmark runs it shows a progress bar and the results so far; the
full results are printed in a text panel underneath.

---

## 4. System overview

### Architecture

```
index.html ─ loads, in dependency order:
   utils.js            geometry helpers, seeded RNG, mean/median, CSV quoting
   network.js          NeuralNetwork, Level, and the GA operators
   controls.js         control flags (forward, left, right, brake)
   road.js             lanes and borders
   trafficLightSensor.js   a light's stop line
   trafficLight.js     light state machine (per-light durations)
   sensor.js           the five rays
   car.js              physics, arbitration, collision, light sensing, fitness
   visualizer.js       draws a network
   panels.js           fitness chart, breeding panel, progress panel
   world.js            scenarios, buildTrack(seed, scenario), World
   ga.js               Evolution (the GA engine), Curriculum
   evaluate.js         runEpisode, heuristic baseline, experiments A/B/C, runner, CSV
   main.js             modes, persistence, drawing, HUD, buttons
```

Every file is a classic `<script>`; top-level classes and constants are shared
through the global lexical scope. All files are `"use strict"`.

The important separation is between **`World`** (the physics, shared by
everything), **`Evolution`** (the GA, an object with no DOM dependency),
**`runEpisode`** (one frozen brain through one World) and **`main.js`** (the
page). Training, the demo and the benchmarks all drive the same `World.step`,
so a brain is judged by identical rules wherever it runs.

### One simulation step (`World.step`)

1. Advance the world clock by one frame (1/60 s) and update every traffic
   light's colour from the time since the run began.
2. Collect the stop-line polygons of every light that is currently red.
3. Update every traffic car (it may lift off for a light or for a car ahead).
4. Update every active AI car: sense the nearest light ahead; **move**
   (arbitrating the four commands); rebuild the collision polygon; update
   distance, overtake, legal-stop and light counters; check for collision
   (recording the cause); then — if still alive — cast the rays, build the
   10 inputs, run the network and set the four commands for the *next* step.
5. Retire a car that has not made progress (§7.7) or that has crossed the
   finish line of a finite route.

### One drawn frame (`main.js: animate`)

In training mode: if a breeding panel is being held, count the hold down;
otherwise run `simSpeed` steps of `Evolution.step` (stopping early if a
generation ends so the panel is readable). In demo mode: `simSpeed` steps of
the demo world. In benchmark mode: nothing — the experiment runs itself in
`setTimeout` slices. Then draw everything and request the next frame.

The simulation runs on its own fixed-step virtual clock rather than on
`requestAnimationFrame` timestamps. That is what lets ×20 run twenty steps per
drawn frame while lights still cycle at the correct simulated rate, and it is
what makes a run reproducible.

---

## 5. The simulation world

### Road and coordinates

Three lanes, 60px each, on a 180px-wide road centred at x = 110 on a 260px
canvas. Lane centres are at x = 50, 110 and 170. The borders are two vertical
segments; touching either one is a crash. Up the road is **negative y**; cars
start at y = 100 and "distance" is `startY − y`.

### Scenarios and the track

A **scenario** is a recipe for a track; a **seed** decides the exact layout
(`buildTrack(seed, scenario)` in `world.js`). The same seed with a different
scenario gives a different track; the same seed and scenario always give the
identical track, which is what makes fitness a property of the genome.

| Scenario | Walls of traffic | Lights | Used for |
| --- | --- | --- | --- |
| **A** | 40, first 350px out, 360px apart; 60% block one lane, 40% two | none | obstacle avoidance and overtaking |
| **B** | none | 12, first 900px out, 1100px apart | stop at red, wait, go on green |
| **C** | as A | 8, first 1500px out, 1800px apart | combined driving |
| **DEMO** | 6, hand-placed | 2, hand-timed | the fixed demonstration route (§9) |

Track constraints: no two consecutive walls are identical, and the first wall
leaves the centre lane (the start lane) open. Every car in a wall shares one
speed, drawn from 0.8–1.2 px/frame, and speeds are assigned so **a wall is
never slower than the wall behind it**; with the car-following rule this
guarantees walls never catch and merge into each other on open road. The
whole track is live from the first frame, so the world at frame N is identical
in every run.

### Traffic ("dummy") cars

No sensors, no network. `forward` is on unless a rule turns it off: they lift
off (and coast to a stop with friction) when a red or yellow light is within
~117px, and when another car is less than 95px ahead in their lane; when
queueing for a light any stopped car less than 95px ahead counts, whatever its
lane. They never brake hard, never reverse, and cannot be damaged.

### Traffic lights

Each light cycles green 9.0 s → yellow 2.2 s → red 7.0 s (18.2 s) with its own
seeded phase offset, on run-relative time. A light's **stop line** is a
4px-thick bar spanning exactly the road width; it is lethal to AI cars only
while the light is **red**, and it is drawn exactly where it kills. The demo
route's lights use their own durations so the first one is red for a long,
documented window.

### Car physics (`car.js: #move`)

| Quantity | Value |
| --- | --- |
| Acceleration | +0.2 px/frame² while `accelerate` |
| Braking | −0.3 px/frame² while `brake`, never below 0 |
| Friction | 0.05 px/frame² towards zero |
| Top speed | 3 px/frame (AI); 0.8–1.2 (traffic) |
| Reverse | none — the car cannot drive backwards |
| Steering | ±0.03 rad/frame while moving; a stationary car cannot pivot |
| Start speed | AI cars roll onto the road at 1.5 px/frame |

**Action arbitration.** The network can raise all four commands at once, so
they are resolved into one consistent action before anything moves: brake
overrides accelerate; left and right together cancel (the car holds its
heading); braking stops at zero. The resolved action is stored as
`car.decision` and shown in the interface, so what you see is what the car
did, not what the four raw outputs requested.

A car with a `driver` function instead of a brain (the hand-coded baseline)
goes through exactly the same arbitration and physics.

### Sensors (`sensor.js`)

Five rays over 90° (−45°, −22.5°, 0°, +22.5°, +45°), each **280px** long. Each
reports the nearest intersection with a border, a traffic car or a red stop
line as an offset 0–1; the input is `1 − offset` (0 = nothing in range). 280px
against traffic closing at up to ~2.2px/frame gives ~130 frames of warning for
a ~76-frame lane change.

---

## 6. The neural network (the genome)

### Inputs — 10

| # | Label | Value |
| --- | --- | --- |
| 1–5 | `L45 L22 FWD R22 R45` | `1 − offset` of each sensor ray |
| 6 | `SPD` | `speed / maxSpeed`, 0 to 1 |
| 7 | `RED` | 1 if the nearest light within 260px is red, else 0 |
| 8 | `YEL` | 1 if it is yellow, else 0 |
| 9 | `GRN` | 1 if it is green, else 0 |
| 10 | `DIST` | `1 − distance / 260` to that light (1 at the line), else 0 |

**With no light inside 260px, inputs 7–10 are all 0.** The colour is one-hot
and the distance is a separate input, so "a green light close ahead" and "a
red light far ahead" are different patterns rather than one blended number.
The previous 7-input network fed a single value that mixed colour and distance
(`1 − d/260` for red or yellow, 0 for green or none) — so a green light was
invisible and "red but far" looked like "yellow but near".

### Layers — 10 → 8 → 4

Fully connected; each neuron computes `tanh(Σ inputᵢ·weightᵢ + bias)`. The
four outputs are thresholded at zero to produce the four boolean commands.

### Outputs — 4

```
outputs[0] > 0  →  accelerate
outputs[1] > 0  →  steer left
outputs[2] > 0  →  steer right
outputs[3] > 0  →  brake
```

### The genome is 124 numbers

| Level | Weights | Biases |
| --- | --- | --- |
| 10 → 8 | 10 × 8 = 80 | 8 |
| 8 → 4 | 8 × 4 = 32 | 4 |
| **Total** | **112** | **12** → **124 genes** |

All genes live in [−1, 1]. Genes are indexed in one fixed order (per level:
biases, then weights row by row); crossover and mutation record their work in
that order, which is what the breeding panel's gene grid shows.

### Brain format

A saved brain is tagged `v3/10x8x4/rays280/light260/brake`. The tag names
everything the genome's meaning depends on — shape, ray length, light
look-ahead, and the brake control. A brain with a different tag, a different
shape, or any gene that is not a finite number is **refused on load** with a
message saying it must be retrained; an old 7-input `bestBrain` entry is
detected and reported the same way. Nothing is ever loaded into an
incompatible network.

The network is **not trained by gradient descent**. There is no
backpropagation, no loss function and no learning rate anywhere in the
project.

---

## 7. The genetic algorithm

The GA is the `Evolution` class in `ga.js`; the operators it calls live in
`network.js`; the fitness function lives in `car.js`. An `Evolution` takes a
`random` function, so every stochastic decision — initial genes, selection,
crossover, mutation — is reproducible from a seed.

### 7.1 Fitness (`car.js: get fitness`)

```
fitness = maxDistance
        + 120 × carsPassed
        + 100 × legalStops
        + 150 × lightsCleared
        − 100   if the car crashed
        −  50   if the crash was running a red light
```

- **maxDistance** is the furthest point reached, not the current position.
- **carsPassed** counts traffic cars the car has got ahead of, each once.
- **legalStops** counts, once per light, the car coming to rest (speed <
  0.1) with its front bumper within 120px before a red light's stop line. It
  is paid once, so sitting there earns nothing further.
- **lightsCleared** counts lights passed while not red (crossing a red one is
  a crash). It pays for the patience of waiting and makes resuming on green
  worth more than staying stopped — together with all the distance after it.
- The **crash penalty is deliberately small.** Crashing already costs every
  metre the car would have driven afterwards; a large extra penalty just
  double-counts it, and at 300 it priced every steering experiment out of the
  search. Running a red light is a crash with the stop line plus a further 50,
  and is reported separately from other collisions.

Why the overtake bonus exists, and why it alone was not enough, is in §14.3
(cause 4) and §14.5.

### 7.2 Selection — tournament, k = 3

Pick three cars uniformly at random (with replacement); the fittest of the
three is a parent. Do it twice for two parents. Tournament selection was chosen
over a roulette wheel because fitness can be negative once the crash penalty
applies. k = 3 (not 5) is deliberate: higher pressure made the population
clonal within ten generations.

### 7.3 Crossover — uniform

The child is built gene by gene; each of its 124 genes is taken from parent A
or parent B by a fair coin flip. Both parents are left untouched.

### 7.4 Mutation — per gene

Each gene independently has probability **0.2** of changing; a changed gene
moves by a uniform random amount up to **±0.5** and is clamped to [−1, 1].
About 25 of the 124 genes change in a typical child, which is high on purpose
(§14.4). The rate and strength are configurable in the interface.

### 7.5 Elitism

The **2** fittest brains are cloned into the next generation unchanged. On a
fixed track this makes the champion's fitness monotone non-decreasing within a
stage: a generation can never come out strictly worse than the one before.
The evolution test asserts this for every generation of its run.

### 7.6 Stagnation escape

If the champion's fitness has not changed (to within 1) for **6** consecutive
generations, mutation rate and strength are multiplied by 1.5 and **20%** of
the children are replaced by random "immigrants" until the champion moves.

### 7.7 Generation lifecycle and the stall rule

```
startGeneration(brains)
   the world is rebuilt from (track seed, scenario) — byte-identical each time
   N cars at the start line, one brain each, rolling at speed 1.5
         │
         ▼  step() × frames, each car until it is
   crashed (wall, traffic or red stop line), OR
   retired: it gained < 1.3 px/frame over the last 420 frames and did not
            wait at a red light in that window, OR
   the generation reaches 3600 frames (60 s)
         │
         ▼
endGeneration()
   summarise: best / mean / median fitness, mean distance, mean overtakes,
   collision rate, legal-stop rate, light-clearing rate, diversity, ...
   autosave the champion if it beats the saved one (§7.8)
   count stagnant generations; curriculum.observe(summary)
   buildNextPopulation(): rank → elites → children (→ immigrants if stagnant)
   show the breeding panel for ~2.8 s
         │
         ▼
advance() → startGeneration(next)
```

Each car's episode is independent: **a car is retired individually** for not
making progress, so one parked car can no longer end the generation for
everyone else. The minimum speed (1.3 px/frame) sits just above the fastest
wall (1.2) and far below cruising speed (3). A car that waited at a red light
during the window is exempt for that window. Why this rule matters more than
any fitness constant is in §14.5.

### 7.8 Determinism and persistence

The track, traffic speeds and light phases come from one seed; the GA's own
random draws come from a second seed; the simulation is fixed-step. The world
is byte-identical at the same frame of every generation (a test asserts it),
fitness is a deterministic function of the genome, and a whole training run is
reproducible from `(track seed, GA seed, parameters)` — both seeds are shown
under the GA parameters.

`localStorage` keys:

| Key | Contents |
| --- | --- |
| `champion` | One record: `format`, the genome, `fitness`, `stage`, `trackSeed`, `gaSeed`, `generation`, `savedAt`, and the fitness breakdown |
| `trainingState` | Generation, seeds, stage and curriculum mode, GA parameters, all-time best distance, the last 400 history points |
| `experimentResults` | Stored benchmark rows (up to 5000) |

**Champion replacement policy.** A candidate from a later stage always
replaces the saved champion (the later stage is the harder task); within the
same stage only a higher fitness replaces it. So a lucky early champion is
never overwritten by a worse late one, and every replacement is announced in
the status line. On reload, generation 1 is seeded as the saved brain
unchanged + 69 mutated copies + 30 random immigrants.

---

## 8. Staged training (curriculum)

The algorithm does not have to discover every behaviour at once. The
`Curriculum` class (`ga.js`) moves the population through three stages, and
promotion is decided by **measured behaviour over consecutive generations**,
never by elapsed time:

| Stage | Track | Promotion rule (must hold for 2 generations running) |
| --- | --- | --- |
| **A** obstacles & overtaking | walls, no lights | champion overtakes ≥ 4 cars **and** population mean overtakes ≥ 0.5 |
| **B** traffic lights | lights, no walls | champion makes ≥ 2 legal stops and clears ≥ 3 lights **and** population mean legal stops ≥ 0.15 |
| **C** combined | walls and lights | final stage |

The population mean thresholds are low on purpose: with 20% per-gene mutation
most children are broken, by design, so a population mean is dominated by
them. The two-generation requirement stops a single lucky generation from
promoting.

The same track seed is used for every stage, so a stage change is a change of
recipe, not a new roll of the dice. The brains carry over: the network shape
is the same in every stage, so nothing has to be retrained from scratch. The
stagnation counter resets on promotion (fitness on the new track is not
comparable), the chart marks the change, and the champion policy (§7.8)
prefers the later stage. The stage can be pinned to A, B or C in the
interface for experiments.

Training and evaluation are separate code paths: `Evolution` never evaluates a
saved brain, and `runEpisode` never breeds.

---

## 9. Demo mode: a frozen champion

**Demo Best Brain** loads the saved champion (refusing anything incompatible
with a clear message), resets the demo route, creates one car with a *clone*
of the genome and drives it. Selection, crossover, mutation and population
replacement are not running; the training population is left exactly where it
was and **Train Population** resumes it. At the end the panel compares the
car's genome with the one that was loaded and reports `brain unchanged during
playback: yes`.

The route (`DEMO_SCENARIO` in `world.js`) is hand-written, so it is the same
every time and can be described exactly:

```
  350px  wall in lane 0        passes without steering (the start lane is open)
  700px  wall in lane 1        blocks the start lane — a lane change is required
 1050px  wall in lanes 1 & 2   only lane 0 is open
 2350px  light 1               green 0–4s, yellow 4–6.2s, RED 6.2–22.2s, then green
                               a car averaging over ~1.8px/frame arrives on red
 2800px  wall in lane 1
 3150px  wall in lanes 0 & 2
 3500px  wall in lane 1
 3900px  light 2               red 18–25s, green from 25s to 45s
 4200px  FINISH
```

Traffic drives at 0.9px/frame, inside the training band. The verdict is
**SUCCESS** only if the car crosses the finish line; otherwise **FAILURE**
with the reason (crashed into traffic / wall / ran the red light, stalled, or
timed out at 90 s) and the distance reached. Nothing steers the car except its
network: there is no scripted path, no recorded playback and no safety
override. A champion that fails the route is reported as failing — go back to
training.

The saved record carries enough to reproduce the conditions the brain was bred
under: the brain format, the stage, the track seed, the GA seed and the
generation.

---

## 10. Benchmarks and CSV export

All three experiments live in `evaluate.js` as generators and are run by
`ExperimentRunner` in ~40 ms time slices between frames, so the page stays
usable; a progress bar shows where they are. Training is paused meanwhile and
neither the population nor the saved champion is touched. Every episode runs
`runEpisode`, which clones the brain, drives it through a `World` with a
6000px finish line (4200px for the demo route) and a 90 s cap, and verifies
afterwards that the brain is byte-identical to what went in.

**Experiment A — GA versus an untrained baseline.** 20 freshly randomised
networks (seeded), the saved champion plus the five current elites, and a
hand-coded rule driver, all on the training track under the current stage's
rules. Reports, per group: trials, mean fitness, mean distance, mean
overtakes, completions, collision rate, red-light violations, legal stops of
red encounters, lights cleared. The rule driver is labelled *hand-coded
baseline (rules, NOT the GA)* everywhere and is never used for anything else.

**Experiment B — mutation rate.** Rates 0.05, 0.10, 0.20, 0.30 with every other
parameter held constant (population 30, 10 generations of 1200 frames,
tournament 3, elites 2), two independent seeded trials each, on the training
track. Reports the final best and mean fitness per rate with each trial's
value, so the spread between trials is visible next to the difference between
rates. Two trials is a small sample, and the panel says so.

**Experiment C — generalisation.** The frozen champion on the training track,
on five fixed seeds it never trained on (101, 202, 303, 404, 505 — any equal
to the training seed is dropped), and on the demo route, each reported
separately. A brain that does well on its own track and poorly on unseen
ones has learned *that track*; the panel does not call it a general driver.

**CSV export.** `Export Results (CSV)` writes every per-generation training
row and every stored benchmark row with one fixed set of columns:

```
experiment, experiment_id, source, timestamp, track_type, track_seed, scenario,
stage, group, trial, generation, population_size, elite_count, tournament_size,
mutation_rate, mutation_strength, ga_seed, best_fitness, mean_fitness,
median_fitness, mean_distance_px, mean_overtakes, collisions, collision_rate,
legal_stops, red_encounters, legal_stop_rate, lights_cleared,
red_light_violations, diversity, fitness, distance_px, overtakes, frames,
sim_seconds, outcome, brain_unchanged, notes
```

Fields are RFC 4180 quoted. Each experiment run gets its own `experiment_id`
and timestamp, so two runs with different parameters are never confused.
Stored rows survive reloads and are only removed by **Clear stored results**,
which asks first.

---

## 11. Parameter reference

| Constant | File | Value | Meaning |
| --- | --- | --- | --- |
| `POP_SIZE` | ga.js | 100 | Cars per generation (UI: 4–400) |
| `ELITE_COUNT` | ga.js | 2 | Brains copied through unchanged (UI: 0–20) |
| `TOURNAMENT_SIZE` | ga.js | 3 | Candidates per selection tournament (UI: 1–20) |
| `MUTATION_RATE` | ga.js | 0.2 | Probability a gene mutates (UI: 0–1) |
| `MUTATION_STRENGTH` | ga.js | 0.5 | Maximum change to a mutated gene (UI: 0.01–1) |
| `STAGNATION_LIMIT` | ga.js | 6 | Flat generations before the escape fires |
| `STAGNATION_MUTATION_BOOST` | ga.js | 1.5 | Multiplier on rate and strength while stagnant |
| `STAGNATION_IMMIGRANT_SHARE` | ga.js | 0.2 | Share of children replaced by random brains while stagnant |
| `IMMIGRANT_SHARE` | ga.js | 0.3 | Random share when resuming from a saved brain |
| `MAX_GENERATION_FRAMES` | ga.js | 3600 | Hard cap, 60 s |
| `STALL_FRAMES` | world.js | 420 | Progress-check window, 7 s |
| `STALL_MIN_SPEED` | world.js | 1.3 | Minimum average px/frame over the window |
| `START_SPEED` | world.js | 1.5 | Speed cars enter the road at |
| `DUMMY_MIN_SPEED` / `MAX` | world.js | 0.8 / 1.2 | Traffic speed range |
| `SCENARIOS` | world.js | — | Stage recipes (rows, spacing, lights) |
| `DEMO_SCENARIO` | world.js | — | The fixed demo route |
| `CURRICULUM` | ga.js | — | Promotion rules per stage |
| `CRASH_PENALTY` | car.js | 100 | Fitness penalty for crashing |
| `RED_LIGHT_PENALTY` | car.js | 50 | Extra penalty for running a red light |
| `LIGHT_BONUS` | car.js | 150 | Fitness per light cleared |
| `OVERTAKE_BONUS` | car.js | 120 | Fitness per car overtaken |
| `LEGAL_STOP_BONUS` | car.js | 100 | Fitness per legal stop |
| `STOP_ZONE` / `STOP_SPEED` | car.js | 120 / 0.1 | Legal-stop zone before the line, and "at rest" |
| `BRAKE_FORCE` | car.js | 0.3 | Speed scrubbed per frame while braking |
| `LIGHT_LOOKAHEAD` | car.js | 260 | Range of the light inputs |
| `Sensor.RAY_COUNT` / `RAY_LENGTH` | sensor.js | 5 / 280 | Rays per car, reach in px |
| `LIGHT_GREEN_MS` / `YELLOW` / `RED` | trafficLight.js | 9000 / 2200 / 7000 | Default light phases |
| `BREEDING_HOLD_FRAMES` | main.js | 170 | Drawn frames the breeding panel is held |
| `DEMO_MAX_FRAMES` | main.js | 5400 | Demo time-out, 90 s |
| `UNSEEN_TRACK_SEEDS` | evaluate.js | 101…505 | Generalisation seeds |

---

## 12. File-by-file reference

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `index.html` | Page layout and script order | — |
| `style.css` | Layout, panels, mode banner | — |
| `main.js` | **The page.** Modes (train / demo / benchmark), persistence, champion policy, config panel, HUD, drawing, buttons, CSV export | `restartTraining`, `onGenerationEnd`, `startDemo`, `resetDemo`, `runEvaluation`, `buildResultsCsv`, `exportResults`, `applyConfig`, `draw`, `animate` |
| `ga.js` | **The GA engine** and the curriculum | `Evolution` (`validateConfig`, `seedPopulation`, `step`, `summarize`, `endGeneration`, `advance`, `tournamentSelect`, `buildNextPopulation`), `Curriculum`, `CURRICULUM` |
| `world.js` | Scenarios, track building, the simulation world, the stall rule | `SCENARIOS`, `DEMO_SCENARIO`, `buildTrack`, `World` (`reset`, `spawnCar`, `step`) |
| `evaluate.js` | Frozen evaluation, baseline driver, experiments, runner, CSV columns | `runEpisode`, `heuristicDriver`, `experimentBaseline`, `experimentMutation`, `experimentGeneralization`, `ExperimentRunner`, `CSV_COLUMNS` |
| `network.js` | The genome and the GA operators | `NeuralNetwork.feedForward / mutate / crossover / clone / equals / genes / geneCount / diversity / toJSON / fromJSON`, `Level` |
| `car.js` | Physics and arbitration, collision with cause, light sensing and the 10 inputs, overtake / legal-stop / light counting, **the fitness function**, dummy behaviour | `Car`, `get fitness`, `fitnessBreakdown`, `actionLabel`, `#move`, `#buildInputs`, `#senseLight`, `#trackLegalStop`, `#assessDamage` |
| `sensor.js` | The five raycasts with distance culling | `Sensor` |
| `road.js` | Lanes, borders, lane centres, viewport-clamped drawing | `Road` |
| `trafficLight.js` | Light state machine with per-light durations | `TrafficLight.update / stateAt / isRed / isStopping` |
| `trafficLightSensor.js` | A light's stop line | `TrafficLightSensor` |
| `visualizer.js` | Draws a network with labelled inputs/outputs and active-output rings | `Visualizer.drawNetwork` |
| `panels.js` | Fitness chart (best / mean / median, stage marks); breeding panel; progress panel | `Panels.drawFitnessHistory / drawBreeding / drawProgress` |
| `controls.js` | Control flags; keyboard listeners for a `KEYS` car (unused by the GA) | `Controls` |
| `utils.js` | `lerp`, `clamp`, `randomGene`, `createSeededRandom`, `shuffled`, `mean`, `median`, intersections, `csvEscape`, `toCsv` | — |
| `tests/` | Node harness and four suites (§13) | — |

---

## 13. Testing

The game files are plain browser scripts, so the tests load them unmodified
into a Node `vm` context with just enough DOM stubbed out (`document`, `Image`,
`localStorage`, `requestAnimationFrame`, `setTimeout`, a no-op canvas
context). `Math.random` is replaced by a seeded generator so a run is
reproducible. The harness can rewrite top-level constants per test, which is
how the evolution test runs 18 short generations in about two minutes.

| Suite | Checks | Covers |
| --- | --- | --- |
| `tests/operators.test.js` | 55 | 10 → 8 → 4 shape and 124 genes; finite outputs, NaN inputs neutralised; save/load round-trip; old 7-input and malformed/NaN genomes rejected; mutation per gene, clamped, reproducible; crossover from both parents, parents untouched; elites copied unchanged; population size; tournament picks the fittest of its sample; selection pressure; stagnation escape; parameter validation and clamping; **brake reduces speed to zero, never reverses; brake overrides accelerate; left + right cancel; stationary car cannot pivot**; obstacle and border detection; one-hot light inputs, zero with no light; red line kills and is a violation; green crossing not penalised and counted; custom light durations; **legal stop counted once, then cleared on green**; stop far short / on green not counted; fitness terms; overtake counted once; stalled car retired; curriculum promotion and pinning |
| `tests/panels.test.js` | 41 | Breeding hold and snapshot (all fitness terms, 124-gene trace, diversity); pause freezes countdown and clock; hold release; history points with stage and all metrics, null rates where there is no denominator; persistence; HUD and metrics table render; speed multiplier; open-lane data; config applied / corrected / displayed; population size change at next generation; stage pinning rebuilds the world; crash marks |
| `tests/demo.test.js` | 38 | No-champion and incompatible-brain messages; demo loads the saved genome (a clone), fixed route with two lights and finish line, red window of light 1; runs to a verdict; **brain unchanged**; population and champion untouched; status panel; reset restores identical conditions and a re-run is identical; Train resumes; frozen evaluation reports real outcomes, leaves the brain unchanged, is reproducible; hand-coded baseline drives; two seeded GA runs identical; experiments A / B / C row counts, groups, constants held, training/unseen split, demo included; page benchmark completes asynchronously and stores rows; CSV header, field counts, training and benchmark rows, values match measurements, quoting, export without `Blob` |
| `tests/evolution.test.js` | 19 | Loop never dies; no NaN; generations advance; same fixed track; stage A has no lights, B and C do; world byte-identical at the same frame of every generation; traffic never overlaps; no three-lane wall on open road; finite fitness; champion saved with format / stage / seed / generation; refuse other formats; champion never replaced by a worse one; **mean population fitness improves substantially; champion fitness never decreases** |

`npm test` runs all four (153 checks). The harness runs the simulation
roughly 15× slower than a browser does, so timings taken in tests are not
representative of the page. The tests exercise logic, not rendering: nothing
here can see pixels, so layout and drawing must be checked in a browser.

---

## 14. Engineering history: problems found and fixed

Every diagnosis below was made by *measuring* — instrumenting the simulation
and counting — after reasoning about it had repeatedly given the wrong answer.

### 14.1 The original project was not a genetic algorithm

| GA component | As cloned |
| --- | --- |
| Population | 100 cars — yes |
| Fitness | Implicit only: "furthest car", never stored or ranked |
| Selection | A human clicked *Save Best Brain* |
| Crossover | Absent |
| Mutation | Every gene nudged 10% towards a random value |
| Generation loop | A page reload; progress lost unless saved by hand |
| Elitism | Broken — the saved brain was mutated along with everything else |

It was a mutation-only hill climber with a person in the loop. The README also
described it as "deep learning"; there was no backpropagation or loss function.

### 14.2 Defects in the original code

Thirty-one issues were found. The significant ones: a traffic-light kill zone
500px wide on a 180px road; lights that switched on a global timer the network
had no input for; a stuck-car detector whose condition was always false; a
`NaN` first frame; crashed cars that kept running sensors forever; ~31,000
intersection tests per frame with no culling; the canvas resized every frame;
100 duplicate image objects; invalid HTML.

### 14.3 Why the cars would not learn to steer — six root causes

1. **All-zero-input deadlock.** A car at rest in an empty lane has almost
   all-zero inputs, so every output is `tanh(bias)` — a constant. 45 of 100
   cars were frozen every generation. *Fix:* cars enter the road at 1.5.
2. **Traffic was unreachable.** Dummies at speed 2 against a population
   averaging 1.57 ran away. *Fix:* slower traffic.
3. **Walls smeared.** Per-car random speeds scattered each wall and left a
   straight channel 76% of the time. *Fix:* one speed per wall.
4. **Fitness rewarded parking.** With distance as the only reward, evolution
   found "drive to the first wall and tailgate it". *Fix then:* overtake
   bonus; crash penalty 300 → 100. *Fix now:* see §14.5 — the bonus was
   necessary but not sufficient.
5. **A random world made fitness noisy.** The same brain scored 2000 or 800
   depending on the layout. *Fix:* fixed seeded track.
6. **The reaction window was too short.** 150px sensors gave ~65 frames of
   warning for a ~76-frame lane change. *Fix:* 280px sensors, traffic 0.8–1.2.

### 14.4 Premature convergence

Tournament k = 5 with mutation 0.12 made the population clonal within ~10
generations. *Fix:* k = 3, mutation 0.2 / 0.5, stagnation escape. A side
effect worth knowing when reading the metrics: with ~25 genes changed per
child, most children are broken, so population means stay far below the
champion and the median is often near zero. That is the price of keeping the
search exploring.

### 14.5 The audit that motivated this version

Before this rewrite the simulation was instrumented and run for 21 full
generations (population 60, 2400 frames). The findings:

- **No car ever met a traffic light.** `atLight` was 0 in every generation.
  The first light was 3000px out and the champion reached ~2800px. Stopping
  at lights was not learned because it was never trained — an exposure
  problem, not a learning problem. *Fix:* the curriculum (§8): stage B is
  nothing but lights, 1100px apart.
- **The light input conflated colour and distance** and could not represent
  green at all. *Fix:* one-hot colour plus a separate distance (§6).
- **"Brake" was reverse.** The same command that slowed a car also drove it
  backwards, and forward + reverse could be raised together. *Fix:* a real
  brake that stops at zero, and arbitration (§5).
- **Nothing rewarded stopping.** A car that stopped for red and one that
  crawled through the zone were indistinguishable until the line. *Fix:*
  `legalStops`, counted once per light (§7.1).
- **Elitism was not monotone** — the champion fell from 3067 to 2287 between
  two generations. The generation ended when the *fitness leader* stalled,
  and the leader was a car parked behind a wall; that cut every other car's
  run short, including the elites'. Fitness was therefore not a pure
  function of the genome. *Fix:* cars are retired individually (§7.7).
- **The parking valley was deeper than the overtake bonus.** After the brake
  change, two full-size runs (population 100, 3600 frames) sat for 15
  generations at 1–2 overtakes with the champion covering ~3460px in 3600
  frames — 0.96px/frame, exactly the speed of the wall it was following.
  Following a wall for 60 s earns ~3450px for free; passing one wall and
  crashing at the next earns ~1500px. No single mutation can cross that gap,
  so none is selected. Raising the bonus does not fix a valley that deep.
  *Fix:* the stall rule's minimum speed (1.3px/frame over 7 s): a car that
  tucks in behind a wall is retired after seven seconds with ~1700px, and
  every extra wall passed becomes the best thing a car can do. On the same
  two tracks the champion then reached 4 overtakes by generation 5 (was 12)
  and 7 by generation 4 (was stuck at 3 for 17 generations).

---

## 15. Measured results

All numbers below were measured with the Node harness, which runs the real
game files. Settings are given with each table; "gen" is generation.

### 15.1 Stage B — learning to stop at red and go on green

Population 60, 2400-frame generations, track seed 3, GA seed 1, stage pinned
to B (12 lights, no traffic):

| gen | champion | legal stops | lights cleared | crashed | population mean legal stops | legal-stop rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 830px | 0 | 0 | no | 0.00 | 0.00 |
| 2 | 1029px | 0 | 1 | yes (ran red) | 0.03 | 0.11 |
| 3 | **4995px** | **4** | **4** | no | 0.08 | 0.25 |
| 12 | 4995px | 4 | 4 | no | 0.17 | 0.40 |

By generation 3 the champion stops at every red light it meets, waits, and
goes on green, and never crashes. 4995px is the ceiling for a 2400-frame
generation with four red waits (4 × 7 s ≈ 1260px lost), which is why the
champion's score is then flat: it is optimal for that track length, not
stuck. The population's legal-stop rate climbs from 0 to 0.40 over the run.

### 15.2 Stage A — learning to overtake, before and after the stall rule

Population 60, 2400-frame generations, stage pinned to A:

| track seed | rule | champion overtakes by gen 4 | first generation with ≥ 4 overtakes |
| --- | --- | --- | --- |
| 3 | old (1px progress) | 1 | 12 |
| 3 | new (1.3px/frame) | 1 | **5** |
| 11 | old | 3 | 18 (reached 5) |
| 11 | new | **7** | **4** |

### 15.3 Automatic curriculum (population 60, 2400 frames, track 11, GA seed 2)

Stage A for 19 generations (champion 3 → 5 overtakes), promoted to B;
generation 22 champion cleared 4 lights, generation 23 made 3 legal stops and
cleared 5; promoted to C at generation 26; by generation 29 the stage C
champion passed 5 cars and cleared 2 lights. (This run used the old stall rule
and the stricter B → C bar; it is included to show the promotion mechanism.)

### 15.4 Full-size runs through the page (population 100, 3600 frames, auto curriculum)

FULL_RUN_RESULTS

### 15.5 Cost

With 100 cars and ~55 traffic cars live, one simulation step costs about
0.6–1 ms in a browser-equivalent context (the Node harness is ~15× slower),
so ×20 runs at roughly 50–60 drawn frames per second on an ordinary laptop in
stage A/C and faster in stage B. Experiment B (4 rates × 2 trials × 10
generations × 30 cars × 1200 frames) takes about half a minute.

---

## 16. Limitations and honest framing

- **It is not deep learning.** The network is trained by evolution, not by
  gradient descent. Say "a neural network whose weights are found by a genetic
  algorithm".
- **It learns one track.** The champion learns *its* course. Experiment C
  measures how much of that carries to unseen tracks; report what it says.
  A good result on the training track is not a general driver.
- **The demo can fail.** The demo route is not the training track. The
  verdict is whatever the frozen brain actually does; nothing is scripted and
  nothing overrides it. If it fails, train more and try again.
- **Learning is slow and ugly to watch at ×1.** Early generations are straight
  lines into walls. The learning is between generations, not within a car's
  life. Judge progress by the mean and median lines, the metrics table, and
  how far up the road the ✕ marks have moved.
- **Plateaus are normal.** On a fixed track the champion's score is exact and
  stays flat until a mutation finds the next wall. The stagnation escape
  mitigates this but does not eliminate it.
- **Population means are low by design.** With 25 genes mutated per child,
  most children are broken; that is what keeps the search exploring. The
  mean and median rising is the signal; their absolute level is not.
- **Controls are binary and the policy is reactive.** Steering is
  all-or-nothing at 0.03 rad/frame, the network has no memory, and one
  frame passes between sensing and acting.
- **Traffic obeys simplified rules.** Dummies brake for lights and for the
  car ahead, never reverse, cannot crash, and can run a yellow.
- **Two trials per mutation rate is a small sample.** Experiment B is a
  comparison, not a proof; differences smaller than the spread between trials
  mean nothing.
- **Not road-safe.** It demonstrates evolutionary optimisation of a
  controller in a toy environment and nothing more.

---

## 17. Presenting the project

### A two-minute explanation

> This is a neural network trained by a genetic algorithm. No libraries — the
> maths is written from scratch.
>
> Each car's brain is a small network: 10 inputs, 8 hidden neurons, 4 outputs.
> The inputs are five distance sensors, the car's own speed, the colour of the
> nearest traffic light as three on/off inputs, and how far away it is. The
> outputs are accelerate, steer left, steer right and brake. The whole network
> is 124 numbers.
>
> There is no correct answer to train on — nobody knows what the car should
> have done at any moment — so backpropagation is impossible. Instead those
> 124 numbers are treated as a chromosome and evolved.
>
> A hundred cars drive at once on the same track. Each gets a fitness score:
> distance, plus bonuses for overtaking, for stopping at a red light and for
> clearing a light legally, minus a crash penalty. Then the next generation is
> bred: the two best brains are copied through unchanged — elitism. The rest
> come from tournament selection — pick three at random, keep the fittest,
> twice — then uniform crossover, where each of the 124 genes is taken from one
> parent or the other by coin flip, and finally mutation, where each gene has a
> 20% chance of being nudged.
>
> Training is staged: first obstacles only, then lights only, then both. The
> population is promoted when its measured behaviour clears a bar, not after a
> fixed time.
>
> *(pause on the breeding panel)* This shows the breeding step for real: the
> ranked population, the elites in green, the parent pairs, and one child's
> genes coloured by which parent each came from, with mutations outlined in red.
>
> *(Demo Best Brain)* This is the best brain, frozen — evolution is off. Watch
> it change lanes around the walls, stop at the red light, wait, and go when
> it turns green. The panel shows what it senses and what it decided each
> frame, and confirms the brain was not changed during playback.
>
> *(Run Evaluation)* And this is the evidence: untrained networks against the
> evolved one on the same track, the mutation-rate comparison, and the frozen
> champion on tracks it never saw — exported as CSV.

### Questions to expect

**Why a genetic algorithm rather than normal neural-network training?**
Because there is no labelled target. Backpropagation needs the right answer to
compute an error gradient; here the only feedback is one score at the end of a
run.

**What is the chromosome?** The 124 weights and biases of one car's network,
in a fixed order, each in [−1, 1].

**Why tournament selection and not a roulette wheel?** Fitness can be
negative once the crash penalty applies; a roulette wheel needs non-negative
weights. And a small tournament (k = 3) keeps selection pressure low enough to
avoid the population going clonal.

**What does elitism do?** Copying the top two through unchanged guarantees the
best brain so far survives. On a fixed, deterministic track it makes the
champion's score monotone non-decreasing within a stage — and a test checks
that in every generation.

**Why does the champion's fitness stay exactly the same for several
generations?** Because the world is deterministic: the same brain on the same
track scores the same number. Nothing changes until a mutation finds the way
past the next wall; then it jumps. The flat stretches are the search failing,
the jumps are it succeeding.

**Why is the fitness function more than just distance?** Because distance
alone rewarded tailgating the first wall forever. And even with an overtake
bonus the "free distance" of following a wall for 60 seconds outweighed any
single improvement, so a rule was added that retires a car averaging less than
1.3 px/frame over 7 seconds unless it is waiting at a red light. Designing the
fitness function and the episode rules is most of the work.

**How do you know the car learned to stop at lights and did not just get
lucky?** Legal stops are counted once per light, only when the car is at rest
within 120px of a red line. The stage B champion makes four of them on four
lights and clears all four on green, every run, because the run is
deterministic; the population's legal-stop rate rises over generations; and
the frozen demo shows the stop happening with the sensor and action readout
visible.

**Is the demo scripted?** No. The demo car is driven by its network alone;
there is no recorded path and no override. The panel verifies the genome is
unchanged after playback, and the demo reports failure when the brain fails.

**Why stages?** So the algorithm does not have to discover steering and
stopping at the same time. Stage A has no lights; stage B has nothing but
lights; stage C combines them. Promotion is on measured behaviour over two
consecutive generations.

**Does it generalise?** Experiment C answers that for the specific champion:
training track, five unseen tracks and the demo route, reported separately.
Quote the numbers, not a feeling.

**What would you change with more time?** Continuous steering outputs, a
recurrent or history input so the policy is not purely reactive, more trials
in Experiment B, and training on several track seeds at once so the champion
is selected for generalisation rather than for one track.

**Is this how real self-driving cars work?** No. It demonstrates evolutionary
optimisation of a controller in a toy environment.

---

## 18. Glossary

| Term | Meaning here |
| --- | --- |
| **Genome / chromosome** | The 124 weights and biases of one car's network |
| **Gene** | One of those 124 numbers |
| **Population** | The cars evaluated together in a generation (100 by default) |
| **Generation** | One evaluation of the population followed by one breeding step |
| **Fitness** | The score used to compare cars (§7.1) |
| **Selection** | Choosing parents in proportion to fitness — here by tournament |
| **Crossover** | Combining two parents' genes into a child — here uniformly |
| **Mutation** | Random small changes to a child's genes |
| **Elitism** | Carrying the best individuals into the next generation unchanged |
| **Immigrant** | A brand-new random brain injected into the population |
| **Stagnation** | The champion's fitness not improving for several generations |
| **Stage / curriculum** | The three training tracks A, B, C and the rules for moving between them |
| **Legal stop** | Coming to rest within 120px before a red stop line, counted once per light |
| **Retired** | A car withdrawn from the run for not making progress; it keeps its score |
| **Champion** | The highest-fitness car in a generation; the *saved champion* is the one in `localStorage` |
| **Leader** | The highest-fitness car *still active*; the camera follows it |
| **Frozen** | A brain that is driven but never mutated, crossed or replaced (demo, benchmarks) |
| **Dummy** | A traffic car with no brain |
| **Wall / row** | One group of traffic cars at the same position, blocking one or two lanes |
| **Diversity** | Mean per-gene standard deviation across the population |

---

## 19. Provenance

The simulation core — car, road, sensor, network and visualiser — follows the
structure of Radu Mariescu-Istodor's freely published "Self-driving car — no
libraries" JavaScript course. The original repository added traffic lights, a
score display and car sprites on top of that, with a Turkish project report
(`README.md`, now translated). The genetic algorithm, the ten-input network
and brake, the fixed track and scenarios, the curriculum, the frozen demo, the
benchmarks and CSV export, the metrics, the breeding and chart panels,
persistence and the test suite were written for this version.
