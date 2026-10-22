# Self-Driving Car — Genetic Algorithm: Project Documentation

A browser simulation in which a population of cars, each controlled by a small
neural network, learns to drive through traffic and obey traffic lights. No
driving rules are programmed and no car is ever told the right thing to do.
The network weights are found by a **genetic algorithm**: the cars that drive
furthest breed the next generation, and driving behaviour emerges over time.

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
8. [Parameter reference](#8-parameter-reference)
9. [File-by-file reference](#9-file-by-file-reference)
10. [Testing](#10-testing)
11. [Engineering history: problems found and fixed](#11-engineering-history-problems-found-and-fixed)
12. [Measured results](#12-measured-results)
13. [Limitations and honest framing](#13-limitations-and-honest-framing)
14. [Presenting the project](#14-presenting-the-project)
15. [Glossary](#15-glossary)
16. [Provenance](#16-provenance)

---

## 1. Purpose

The project demonstrates **evolutionary optimisation of a neural network** on a
problem where supervised learning is not possible.

There is no dataset of correct steering decisions and no way to compute an
error gradient: nobody knows what a car *should* have done at frame 412. The
only feedback available is a single score per car at the end of its run. That
rules out backpropagation and makes the problem a natural fit for a genetic
algorithm, which needs nothing but a way to score candidates.

So the project is really about three things:

- **Representation** — a driver's entire behaviour is encoded as 100 numbers
  (the weights and biases of a 7 → 8 → 4 network), which makes it a chromosome.
- **Fitness** — how to turn "drove well" into a number that evolution can
  climb, and how easy it is to write one that rewards the wrong thing.
- **Evolution** — selection, crossover, mutation and elitism operating on a
  population of 100 such chromosomes, with the whole process made visible on
  screen so it can be explained by demonstration.

The deliverable is a running simulation in which the learning itself can be
watched: the population's fitness climbs generation by generation, the breeding
step is shown between generations, and the best car's brain is drawn live.

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
| **Pause / Resume** | Freezes the simulation *and* the breeding panel, so the panel can be held on screen while it is explained |
| **×1 / ×5 / ×20** | Simulation speed. ×20 runs twenty simulation steps per drawn frame; traffic lights still cycle at the correct simulated rate |
| **Save Best Brain** | Writes the current leader's network to `localStorage` by hand (saving is normally automatic — see §7.8) |
| **Load Saved Brain** | Restarts training seeded from the saved brain, on the track it was bred on |
| **Discard & Restart** | Wipes the saved brain, history and track; starts generation 1 from random brains on a freshly generated track |

For a demonstration: press **×20** for a minute to let the fitness chart build,
then **Pause** on a breeding panel to walk through the algorithm, then **×1** to
watch the best car drive.

---

## 3. What is on the screen

The page has three columns.

### Left column — statistics and the fitness chart

**All-time Best Distance** — the furthest any car has ever driven, across all
generations and sessions.

**About Game** panel:

| Line | Meaning |
| --- | --- |
| Generation | Which generation is currently being evaluated |
| Generation time | Seconds of simulated time this generation has run |
| Best car velocity | Current speed of the car the camera follows |
| Alive cars | How many of the 100 have not crashed yet |
| Best distance | Furthest the camera car has reached this generation |
| Cars passed | How many traffic cars the camera car has overtaken |
| Best fitness | The highest fitness score in this generation |
| *fitness breakdown* | The equation behind that score, e.g. `1450 dist + 3×120 passed + 1×150 lights` |

**Fitness by generation** chart — one point per completed generation. The
**yellow** line is the champion's fitness; the **blue** line is the population
mean. It is persisted, so it is already drawn when the page is reopened. This
chart is the evidence that learning is happening: the yellow line climbs in a
staircase, and the blue line following it upwards shows that the *whole
population* improves, not one lucky individual.

### Middle column — the road

| What you see | What it is |
| --- | --- |
| One **solid blue car with yellow lines** | The leader: the highest-scoring car still alive. The camera follows it. The yellow lines are its five sensors |
| Many **faint, translucent blue cars** | The other 99 cars of the population. All 100 drive simultaneously; they are drawn at 20% opacity so the leader is visible through them |
| **Brightly coloured cars** (pink, purple, orange, green, cyan) | Traffic. These have no brain; they drive straight at a slow fixed speed and are obstacles. The colours are random, purely to tell them apart |
| **Faint green shaded lane** beside a wall of traffic | The open lane — the route past that wall |
| **Dark ✕ marks** | Where a car crashed this generation. Later generations visibly die further up the road |
| **Traffic light** with a coloured lamp above it | Green, yellow or red. When red, a **red bar across the road** is drawn; crossing it destroys the car. Yellow is the warning to brake |
| Black banner *"leader crashed — following next best car"* | The camera has switched to another car because the one it was following was destroyed |

The important distinction: **the coloured cars are not competitors, they are
walls.** Only the blue cars are being trained.

### Right column — the brain

Normally shows the leader's neural network live. Circles are neurons, lines are
weights. **Yellow = positive weight, blue = negative; opacity = magnitude.**
Inputs are labelled along the bottom:

```
L45  L22  FWD  R22  R45  SPD  RED
```

(the five sensor rays from left to right, then the car's own speed, then the
red-light signal). Outputs are labelled `↑ ← → ↓`.

Between generations this canvas is replaced for about three seconds by the
**breeding panel**, which shows the actual breeding step that just happened:

1. **Score and rank by fitness** — the top eight cars with their fitness
   breakdown; the elites marked in green.
2. **Elitism** — how many brains are copied through unchanged.
3. **Tournament selection + crossover + mutation** — four real parent pairs
   with their ranks and scores.
4. **One child's 100 genes** — a 20×5 grid, each cell coloured by which parent
   that gene came from, mutated genes outlined in red.

If the stagnation escape (§7.6) fired, the panel says so in orange.

---

## 4. System overview

### Architecture

```
index.html ─ loads, in dependency order:
   utils.js            geometry helpers, seeded RNG
   network.js          NeuralNetwork, Level, and the GA operators
   visualizer.js       draws a network
   panels.js           fitness chart and breeding panel
   controls.js         keyboard / dummy control flags
   road.js             lanes and borders
   trafficLightSensor.js   a light's stop line
   trafficLight.js     light state machine
   sensor.js           the five rays
   car.js              physics, collision, fitness accounting
   main.js             the GA engine, the world, the frame loop, the UI
```

Every file is a classic `<script>`; top-level classes and constants are shared
through the global lexical scope. All files are `"use strict"`.

### One simulation step (`main.js: step()`)

1. Advance the simulation clock by one frame (1/60 s) and update every traffic
   light's colour from the time since the generation began.
2. Collect the stop-line polygons of every light that is currently red.
3. Update every traffic car (it may brake for a light or for a car ahead).
4. Update every AI car: move, rebuild its collision polygon, update distance
   and overtake counters, check for collision, then — if still alive — cast its
   rays, build its 7 inputs, run the network and set its four controls for the
   *next* step.
5. Pick the camera target (highest fitness among survivors) and the champion
   (highest fitness overall). Record crash marks.
6. Update the stall timer. End the generation if everyone is dead, the leader
   has made no progress for 7 seconds (not counting time waiting at a red
   light), or 60 simulated seconds have passed.

### One drawn frame (`main.js: animate()`)

If not paused: if a breeding panel is being held, count down the hold;
otherwise run `simSpeed` simulation steps (stopping early if a generation ends
so the panel is readable). Then draw everything and request the next frame.

The simulation runs on its own fixed-step virtual clock rather than on
`requestAnimationFrame` timestamps. That is what lets ×20 run twenty steps per
drawn frame while traffic lights still cycle at the correct simulated rate, and
it is what makes a run reproducible.

---

## 5. The simulation world

### Road

Three lanes, 60px each, on a 180px-wide road centred at x = 110 on a 260px
canvas. Lane centres are at x = 50, 110 and 170. The borders are two vertical
segments; touching either one is a crash. Only the visible slice of the road
is drawn each frame.

### Coordinates

Up the road is **negative y**. Cars start at y = 100 and drive towards
increasingly negative y. "Distance" is `startY − y`.

### The track

The course is **one fixed track, built from a seed** (`buildTrack`). Every
generation drives the identical track, so a brain's fitness is a pure function
of its genome rather than of the random layout it happened to meet. The seed
is stored in `localStorage`; *Discard & Restart* rolls a new one.

The track is 40 walls of traffic, the first 350px ahead of the start and the
rest 360px apart, so the last is about 14,300px away — further than any car can
drive in a 60-second generation at 3px/frame (10,800px). Each wall blocks
either one lane (60%) or two lanes (40%), chosen at random with two
constraints: no two consecutive walls are identical, and the first wall leaves
the centre lane (where the cars start) open.

Every car in a wall shares one speed, drawn from 0.8–1.2 px/frame, and the
speeds are assigned so that **a wall is never slower than the wall behind it**.
Together with the car-following rule below this guarantees walls never catch
and merge into each other on open road.

The whole track is live from the first frame — nothing streams in or out —
so the world state at frame N is identical in every generation regardless of
where the leader is.

### Traffic ("dummy") cars

- No sensors, no network. `forward` is on unless a rule turns it off.
- Lift off the throttle (and coast to a stop with friction) when a red or
  yellow light is within ~117px, and when another car is less than 95px ahead
  in their lane. When queueing for a light, *any* stopped car less than 95px
  ahead counts, whatever its lane, so a wall queues behind the wall in front
  of it rather than pulling up alongside and merging into a three-lane block.
- Never reverse.
- Cannot be damaged by traffic or stop lines (a wrecked dummy would sit on the
  road for the rest of the generation).

### Traffic lights

Four lights, 3,000px apart, the first 3,000px from the start. Each cycles
green 9.0 s → yellow 2.2 s → red 7.0 s (18.2 s total) with its own seeded
phase offset, on generation-relative time.

A light's **stop line** is a 4px-thick bar spanning exactly the road width at
the light's y. It is lethal to AI cars only while the light is **red**. It is
drawn red when red and yellow when yellow, exactly where it kills, so a car is
never destroyed outside the bar it can see.

### Car physics (`car.js: #move`)

| Quantity | Value |
| --- | --- |
| Acceleration | 0.2 px/frame² while `forward`; −0.2 while `reverse` |
| Friction | 0.05 px/frame² towards zero |
| Top speed | 3 px/frame (AI); 0.8–1.2 (traffic) |
| Reverse limit | half of top speed |
| Steering | ±0.03 rad/frame while moving, sign flipped when reversing |
| Start speed | AI cars roll onto the road at 1.5 px/frame |

The car body is a 4-point polygon (30×50px) rotated by its heading. Collision
is segment–segment intersection between polygons (`polysIntersect`). A cheap
distance pre-check skips any traffic car whose centre is more than 80px away.

### Sensors (`sensor.js`)

Five rays, spread over 90° (−45°, −22.5°, 0°, +22.5°, +45° from the heading),
each **280px** long. Each ray reports the nearest intersection with a road
border, a traffic car's polygon, or a red stop line, as an *offset* 0–1 along
the ray. The network input is `1 − offset`: 0 means nothing in range, close to
1 means something is right at the bumper.

Ray length is not cosmetic. The car's **reaction window** is
`rayLength / closingSpeed`. At 150px against traffic closing at ~2px/frame a
car saw a wall ~65 frames out, but a lane change at full speed takes ~76
frames — it was physically unable to react in time whatever its brain decided.
280px gives ~140 frames.

---

## 6. The neural network (the genome)

### Inputs — 7

| # | Label | Value |
| --- | --- | --- |
| 1–5 | `L45 L22 FWD R22 R45` | `1 − offset` of each sensor ray |
| 6 | `SPD` | `speed / maxSpeed`, so −0.5 to 1 |
| 7 | `RED` | 0 if no red/yellow light within 260px ahead; otherwise rises linearly to 1 as the light gets closer |

Inputs 6 and 7 were added in the rewrite. Without them the network had no way
of knowing it was moving, or that a light existed, let alone its colour, so
"stop at red, go on green" was not learnable.

### Layers — 7 → 8 → 4

Fully connected. Each neuron computes `tanh(Σ inputᵢ·weightᵢ + bias)`.

The original project compared the weighted sum *against* the bias and emitted 0
or 1. That discarded all magnitude information between layers (a hidden neuron
could only say "yes" or "no") and inverted the usual meaning of a bias. `tanh`
keeps the signal continuous; the four final outputs are thresholded at zero to
produce the boolean controls.

### Outputs — 4

```
outputs[0] > 0  →  forward
outputs[1] > 0  →  left
outputs[2] > 0  →  right
outputs[3] > 0  →  reverse (brake)
```

### The genome is 100 numbers

| Level | Weights | Biases |
| --- | --- | --- |
| 7 → 8 | 7 × 8 = 56 | 8 |
| 8 → 4 | 8 × 4 = 32 | 4 |
| **Total** | **88** | **12** → **100 genes** |

All genes live in [−1, 1]. Change those 100 numbers and you change how the car
drives; that is the entire representation the genetic algorithm works on.

The network is **not trained by gradient descent**. There is no
backpropagation, no loss function and no learning rate anywhere in the
project. The network is a fixed parameterised function; all learning happens
through the genetic algorithm described next.

---

## 7. The genetic algorithm

The GA lives in `main.js`; the operators it calls live in `network.js`; the
fitness function lives in `car.js`.

### 7.1 Fitness (`car.js: get fitness`)

```
fitness = maxDistance
        + 120 × carsPassed
        + 150 × lightsCleared
        − 100   if the car crashed
```

- **maxDistance** is the furthest point reached, not the current position, so
  a car cannot lose credit by drifting backwards.
- **carsPassed** counts traffic cars the car has got ahead of. It exists
  because distance alone reliably evolved a car that drives up to the first
  wall and tailgates it forever at the wall's own speed — safe, and worth more
  than any attempt to pass that ends in a crash. Paying for each overtake puts
  a gradient into that valley.
- **lightsCleared** counts lights passed while not red (crossing a red one is
  a crash, so a surviving crossing is a legal one). It pays for the patience of
  waiting at red.
- The **crash penalty is deliberately small.** Crashing already costs the car
  every metre it would have driven afterwards; a large extra penalty just
  double-counts it, and at 300 it priced every steering experiment out of the
  search.

### 7.2 Selection — tournament, k = 3 (`tournamentSelect`)

Pick three cars uniformly at random; the fittest of the three is a parent. Do
it twice for two parents. Tournament selection was chosen over a roulette
wheel because fitness can be negative once the crash penalty applies, and
roulette needs non-negative weights. k = 3 (not 5) is deliberate: higher
pressure made the population clonal within ten generations.

### 7.3 Crossover — uniform (`NeuralNetwork.crossover`)

The child is built gene by gene; each of its 100 genes is taken from parent A
or parent B by a fair coin flip. The operator can record which parent each
gene came from (`originsOut`), which is what the breeding panel's gene grid
shows.

### 7.4 Mutation — per gene (`NeuralNetwork.mutate`)

Each gene independently has probability **0.2** of changing; a changed gene
moves by a uniform random amount up to **±0.5** and is clamped to [−1, 1]. The
original implementation nudged *every* gene 10% towards a fresh random value,
which cannot preserve a good gene while exploring others. The operator can
record which gene indices changed (`mutatedOut`) for the gene grid.

### 7.5 Elitism

The **2** fittest brains are cloned into the next generation unchanged. On a
fixed track this makes the champion's fitness monotone non-decreasing: a
generation can never come out strictly worse than the one before.

### 7.6 Stagnation escape

If the champion's fitness has not changed (to within 1) for **6** consecutive
generations the search is sitting in a local optimum — typically the best
brain reaches one particular wall and nothing in the population gets past it.
For as long as that lasts, mutation rate and strength are multiplied by 1.5
(to 0.3 and 0.75) and **20%** of the children are replaced by brand-new random
"immigrants". The breeding panel announces this in orange when it fires.

### 7.7 Generation lifecycle

```
startGeneration(brains)
   100 cars at the start line, each given one brain, rolling at speed 1.5
   the fixed track is rebuilt (same seed → same traffic, same light phases)
         │
         ▼  step() × N, until:
   every car crashed, OR
   the leader made no progress for 420 frames (paused while waiting at red), OR
   3600 frames (60 s) elapsed
         │
         ▼
endGeneration()
   score every car; autosave the champion if it beats the best ever saved
   record (best, mean) for the chart
   count stagnant generations
   buildNextPopulation(): rank → 2 elites → 98 children (or 78 + 20 immigrants)
   show the breeding panel for ~2.8 s (170 drawn frames)
         │
         ▼
startGeneration(next)
```

### 7.8 Determinism and persistence

Because the track, traffic speeds and light phases all come from one seed, and
the simulation runs on a fixed-step clock from a known start, **the world is
byte-identical at the same frame of every generation** (a test asserts this).
Fitness is therefore a deterministic function of the genome. This is why the
champion's fitness plateaus at an exact value for several generations and then
jumps: nothing has changed until a mutation finds the way past the next wall.

`localStorage` keys:

| Key | Contents |
| --- | --- |
| `bestBrain` | JSON of the best brain ever seen (by fitness) |
| `bestBrainFormat` | `7x8x4/rays280/light260` — the brain is refused on load if this does not match the current sensors |
| `bestBrainFitness` | Fitness of the saved brain, so a worse later champion never overwrites it |
| `generation` | Generation counter |
| `_gameBestScore` | All-time best distance |
| `fitnessHistory` | Up to 400 `(generation, best, mean)` points for the chart |
| `trackSeed` | The seed the current track was built from |

On a reload with a saved brain, generation 1 is seeded as: the saved brain
unchanged, 69 mutated copies of it, and 30 random immigrants (so a single
ancestor cannot dominate a population that has lost all diversity).

---

## 8. Parameter reference

All tunables are top-level constants.

| Constant | File | Value | Meaning |
| --- | --- | --- | --- |
| `POP_SIZE` | main.js | 100 | Cars per generation |
| `ELITE_COUNT` | main.js | 2 | Brains copied through unchanged |
| `TOURNAMENT_SIZE` | main.js | 3 | Candidates per selection tournament |
| `MUTATION_RATE` | main.js | 0.2 | Probability a gene mutates |
| `MUTATION_STRENGTH` | main.js | 0.5 | Maximum change to a mutated gene |
| `STAGNATION_LIMIT` | main.js | 6 | Flat generations before the escape fires |
| `STAGNATION_MUTATION_BOOST` | main.js | 1.5 | Multiplier on rate and strength while stagnant |
| `STAGNATION_IMMIGRANT_SHARE` | main.js | 0.2 | Share of children replaced by random brains while stagnant |
| `IMMIGRANT_SHARE` | main.js | 0.3 | Random share when resuming from a saved brain |
| `STALL_FRAMES` | main.js | 420 | Leader frames without progress that end a generation |
| `MAX_GENERATION_FRAMES` | main.js | 3600 | Hard cap, 60 s |
| `BREEDING_HOLD_FRAMES` | main.js | 170 | Drawn frames the breeding panel is held |
| `START_SPEED` | main.js | 1.5 | Speed cars enter the road at |
| `DUMMY_MIN_SPEED` / `MAX` | main.js | 0.8 / 1.2 | Traffic speed range |
| `TRACK_ROWS` | main.js | 40 | Walls of traffic |
| `TRACK_FIRST_ROW_AHEAD` | main.js | 350 | Distance to the first wall |
| `TRACK_ROW_SPACING` | main.js | 360 | Distance between walls |
| `TRACK_TWO_LANE_CHANCE` | main.js | 0.4 | Probability a wall blocks two lanes |
| `TRACK_LIGHT_SPACING` | main.js | 3000 | Distance between lights |
| `TRACK_FIRST_LIGHT_AHEAD` | main.js | 3000 | Distance to the first light |
| `CRASH_PENALTY` | car.js | 100 | Fitness penalty for crashing |
| `LIGHT_BONUS` | car.js | 150 | Fitness per light cleared |
| `OVERTAKE_BONUS` | car.js | 120 | Fitness per car overtaken |
| `LIGHT_LOOKAHEAD` | car.js | 260 | Range of the red-light input |
| `DUMMY_FOLLOW_DISTANCE` | car.js | 95 | Gap at which traffic lifts off behind a car |
| `Sensor.RAY_COUNT` | sensor.js | 5 | Rays per car |
| `Sensor.RAY_LENGTH` | sensor.js | 280 | Ray reach in px |
| `LIGHT_GREEN_MS` / `YELLOW` / `RED` | trafficLight.js | 9000 / 2200 / 7000 | Light phases |

---

## 9. File-by-file reference

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `index.html` | Page layout: stats column, chart canvas, road canvas, control buttons, network canvas; loads scripts in dependency order | — |
| `style.css` | Layout and HUD styling | — |
| `main.js` | **The GA engine and the world.** Constants, track generation, population seeding, selection, breeding, generation lifecycle, the step loop, drawing, HUD, controls, persistence | `buildTrack`, `resetWorld`, `seedPopulation`, `tournamentSelect`, `buildNextPopulation`, `startGeneration`, `endGeneration`, `step`, `draw`, `animate`, `restartTraining` |
| `network.js` | The genome and the GA operators | `NeuralNetwork.feedForward / mutate / crossover / clone / fromJSON`, `Level` |
| `car.js` | Physics, collision polygon, sensor→brain→controls wiring, overtake and light counting, **the fitness function**, dummy behaviour, shared sprite/mask cache | `Car`, `get fitness`, `#move`, `#assessDamage`, `#senseLight`, `#driveDummy`, `#blockedAhead` |
| `sensor.js` | The five raycasts with distance culling | `Sensor`, `RAY_COUNT`, `RAY_LENGTH` |
| `road.js` | Lanes, borders, lane centres, viewport-clamped drawing | `Road.getLaneCenter`, `Road.draw` |
| `trafficLight.js` | Light state machine and drawing (lamp colour over the static sprite) | `TrafficLight.update`, `isRed`, `isStopping` |
| `trafficLightSensor.js` | A light's stop line: a polygon spanning exactly the road, drawn exactly where it kills | `TrafficLightSensor` |
| `visualizer.js` | Draws a network with labelled inputs/outputs and a weight-colour key | `Visualizer.drawNetwork` |
| `panels.js` | Fitness-by-generation chart; the breeding panel with the gene grid | `Panels.drawFitnessHistory`, `Panels.drawBreeding` |
| `controls.js` | Control flags; keyboard listeners for a `KEYS` car (unused by the GA) | `Controls` |
| `utils.js` | `lerp`, `clamp`, `randomGene`, `createSeededRandom`, `shuffled`, segment/polygon intersection, `getRGBA`, `getRandomColor` | — |
| `tests/` | Node test harness and three suites (see §10) | — |
| `package.json` | `npm start` (static server) and `npm test` | — |
| `README.md` | The original project report, translated to English, with the current implementation summarised at the top | — |

---

## 10. Testing

The game files are plain browser scripts, so the tests load them unmodified
into a Node `vm` context with just enough DOM stubbed out (`document`, `Image`,
`localStorage`, `requestAnimationFrame`, a no-op canvas context). `Math.random`
is replaced by a seeded generator so an evolutionary run is reproducible rather
than flaky. The harness can rewrite top-level constants per test, which is how
the evolution test runs 18 short generations in well under a minute.

| Suite | Checks | Covers |
| --- | --- | --- |
| `tests/operators.test.js` | 19 | Brain save/load round-trip and format rejection; mutation is per-gene and clamped; crossover genes come from both parents and parents are untouched; clone independence (elitism); red stop line kills, green does not, kill zone equals the road; light cycle; light input rises on red; fitness shaping |
| `tests/panels.test.js` | 31 | Breeding hold and snapshot contents; real gene-origin and mutation traces; pause freezes the countdown and the clock; speed multiplier; history recording and persistence; track rows record genuinely open lanes; first wall leaves the start lane open; stagnation escape produces immigrants only when stagnant; crash marks |
| `tests/evolution.test.js` | 16 | Loop never dies; no NaN; generations advance without reload; same fixed track every generation; **world byte-identical at the same frame of every generation**; **traffic never overlaps**; **no three-lane wall on open road**; brain autosaved and tagged with its sensor format; **mean population fitness improves substantially**; champion fitness does not collapse |

`npm test` runs all three (66 checks). Note that the harness runs the
simulation roughly 17× slower than a browser does — Node's `vm` routes every
global lookup through an interceptor — so timings taken in tests are not
representative of the page.

---

## 11. Engineering history: problems found and fixed

This section records what was wrong with the project as cloned and what went
wrong during the rewrite, because the failures are as instructive as the
result. Every diagnosis below was made by *measuring* — instrumenting the
simulation and counting — after reasoning about it had repeatedly given the
wrong answer.

### 11.1 The original project was not a genetic algorithm

| GA component | As cloned |
| --- | --- |
| Population | 100 cars — yes |
| Fitness | Implicit only: "furthest car", never stored or ranked |
| Selection | A human clicked *Save Best Brain* |
| Crossover | Absent |
| Mutation | Every gene nudged 10% towards a random value |
| Generation loop | A page reload (`location.reload()`); progress lost unless saved by hand |
| Elitism | Broken — the saved brain was mutated along with everything else |

It was a mutation-only hill climber with a person in the loop. The README also
described it as "deep learning"; there was no backpropagation or loss function.

### 11.2 Defects in the original code

Thirty-one issues were found. The significant ones: a traffic-light kill zone
500px wide on a 180px road that killed cars outside the visible bar; lights
that switched on a global 10-second timer the network had no input for, so a
car mid-crossing died at random; a stuck-car detector whose condition was
provably always false; `-bestCar.y.toFixed(2)` (precedence bug); an implicit
global; a `NaN` first frame; crashed cars that kept running sensors forever;
~31,000 intersection tests per frame with no culling; the canvas resized every
frame; dashed lines drawn across a 2,000,000px path; 100 duplicate image
objects; one source file saved as UTF-16; invalid HTML; a p5.js claim with no
p5.js.

### 11.3 Why the cars would not learn to steer — six root causes

Each of these produced the same symptom: the best car drove in a straight line.

1. **All-zero-input deadlock.** A car at rest in an empty lane has almost
   all-zero inputs, so every output is `tanh(bias)` — a constant. If the
   constant said "don't accelerate" the car never moved, so its inputs never
   changed, so it never accelerated. 45 of 100 cars were frozen every
   generation. *Fix:* cars enter the road at speed 1.5.
2. **Traffic was unreachable.** Dummies at speed 2 against a population
   averaging 1.57 ran away; cars never met an obstacle. *Fix:* slower traffic.
3. **Walls smeared.** Per-car random speeds made the two cars of a wall drift
   apart; scattered single cars left a straight channel 76% of the time.
   *Fix:* one speed per wall.
4. **Fitness rewarded parking.** With distance as the only reward, evolution
   found "drive to the first wall and tailgate it" — safe, and better than a
   crash. Overtakes pinned at exactly 2 for 20+ generations is the signature.
   *Fix:* overtake bonus; crash penalty 300 → 100.
5. **A random world made fitness noisy.** The same brain scored 2000 or 800
   depending on the layout. *Fix:* fixed seeded track; lights on generation
   time; whole track live from frame 0.
6. **The reaction window was too short.** 150px sensors against ~2.3px/frame
   closing speed gave ~65 frames of warning for a ~76-frame lane change. The
   champion sat at exactly 1180px for 27 generations. *Fix:* 280px sensors,
   traffic 0.8–1.2, walls 360px apart. Champion went to 3178px with 8
   overtakes and a light cleared.

### 11.4 Defects introduced during the rewrite, and fixed

- **Traffic merged into impassable walls.** Walls of different speeds, with no
  dummy-to-dummy collision, drove through one another; 61% of frames had a
  three-lane block somewhere, and walls stacked on top of each other at red
  lights. *Fix:* walls ordered by speed, car-following, queue-behind at lights.
  Overlaps measured 73% → 0%.
- **Stale brains re-loaded.** A brain saved with 150px sensors passed the
  shape check and seeded the whole population with a driver that had learned
  the wrong world. *Fix:* brains are tagged with their sensor format.
- **Autosave kept the latest champion, not the best.** *Fix:* best-ever by
  fitness.
- **Dummies reversed into waiting cars** at red lights. *Fix:* never reverse.
- **Premature convergence.** Tournament k = 5 with mutation 0.12 made the
  population clonal within ~10 generations. *Fix:* k = 3, mutation 0.2/0.5,
  stagnation escape.

---

## 12. Measured results

Measured with the test harness (population 60, generations capped at 1600
frames so 30 generations fit in a few minutes; the real page uses 100 cars and
3600 frames). "a → b" is the mean of the first seven generations → the mean of
the last seven.

**Final code**, two tracks, 30 generations each:

| Track seed | Champion distance | Cars overtaken | Cars that steer (of 60) | Generation-30 champion |
| --- | --- | --- | --- | --- |
| 3 | 2088 → 3764 | 3.0 → 9.9 | 29 → 33 | 4693px, 12 cars passed, 1 red light cleared |
| 11 | 1499 → 2127 | 1.7 → 4.0 | 32 → 31 | 2302px, 4 cars passed |

Both show the staircase pattern: seed 3 sat at 2527px from generation 17 to 29
and then jumped to 4693px in a single generation. Seed 11 — the track that had
previously got stuck — now improves steadily and keeps its steering cars (32 →
31) instead of breeding them out.

**Before the traffic and stagnation fixes** (§11.4), for comparison:

| Track seed | Champion distance | Cars overtaken | Cars that steer (of 60) |
| --- | --- | --- | --- |
| 3 | 1523 → 3178 | 3.1 → 8.0 | 32 → 37 |
| 42 | 1837 → 3015 | 3.9 → 7.0 | 27 → 39 |
| 11 | 2405 → 2784 | 4.4 → 3.0 | 34 → 9 |

Seed 11 was then the honest example of premature convergence: it settled on a
near-straight driver and steering collapsed from 34 cars to 9. The merged,
impassable traffic walls described in §11.4 were present on that track at the
time; with them removed and the stagnation escape added, the same track learns.

The automated evolution test asserts, on every run, that mean population
fitness rises substantially over 18 short generations.

Simulation cost in a browser-equivalent context is about **0.65 ms per step**
with 100 cars and ~55 traffic cars live, so ×20 runs at a full 60 fps.

---

## 13. Limitations and honest framing

- **It is not deep learning.** The network is trained by evolution, not by
  gradient descent. Describe it as "a neural network whose weights are found by
  a genetic algorithm".
- **It learns one track.** Because the track is fixed, the champion learns
  *this* course. It is a demonstration of optimisation, not of general driving.
  A new track (*Discard & Restart*) starts learning over.
- **Learning is slow and ugly to watch at ×1.** Early generations are straight
  lines into walls, and the camera follows whichever car happens to be best so
  far — which, early on, is a lucky straight-line driver. The learning is
  between generations, not within a car's life. Judge progress by the chart,
  "Cars passed", and how far up the road the ✕ marks have moved.
- **It can get stuck.** Plateaus of 10+ generations are normal; a population
  can converge on a local optimum. The stagnation escape mitigates this but does
  not eliminate it.
- **Controls are binary.** Steering is all-or-nothing at 0.03 rad/frame and
  the network has no memory, so the policy is purely reactive.
- **Traffic obeys simplified rules.** Dummies brake for lights and for the car
  ahead, never reverse, cannot crash, and can run a yellow.

---

## 14. Presenting the project

### A 90-second explanation

> This is a neural network trained by a genetic algorithm. No libraries — the
> maths is written from scratch.
>
> Each car's brain is a small network: 7 inputs, 8 hidden neurons, 4 outputs.
> The inputs are five distance sensors, the car's own speed, and whether a red
> light is ahead. The outputs are accelerate, left, right, brake. The whole
> network is 100 numbers.
>
> There is no correct answer to train on — nobody knows what the car should
> have done at any moment — so backpropagation is impossible. Instead those
> 100 numbers are treated as a chromosome and evolved.
>
> A hundred cars drive at once on the same track. Each gets a fitness score:
> distance, plus bonuses for overtaking and clearing lights, minus a crash
> penalty. Then the next generation is bred: the two best brains are copied
> through unchanged — elitism. The rest come from tournament selection — pick
> three at random, keep the fittest, twice — then uniform crossover, where each
> of the 100 genes is taken from one parent or the other by coin flip, and
> finally mutation, where each gene has a 20% chance of being nudged.
>
> *(pause on the breeding panel)* This shows that step for real: the ranked
> population, the elites in green, the parent pairs, and one child's 100 genes
> coloured by which parent each came from, with mutations outlined in red.
>
> *(point at the chart)* And this is the proof it works: the blue line is the
> population's average fitness climbing over generations. Not one lucky car —
> the whole population getting better.

### Questions to expect

**Why a genetic algorithm rather than normal neural-network training?**
Because there is no labelled target. Backpropagation needs the right answer to
compute an error gradient; here the only feedback is one score at the end of a
run.

**Why tournament selection and not a roulette wheel?** Fitness can be
negative once the crash penalty applies; a roulette wheel needs non-negative
weights.

**What does elitism do?** Without it a generation can be worse than its
parents and progress is lost. Copying the top two through guarantees the best
brain so far survives — and on a fixed track makes the champion's score
monotone.

**Why does the champion's fitness stay exactly the same for several
generations?** Because the world is deterministic: the same brain on the same
track scores the same number. Nothing changes until a mutation finds the way
past the next wall; then it jumps. The flat stretches are the search failing,
the jumps are it succeeding.

**Why is the fitness function more than just distance?** Because distance
alone rewarded tailgating the first wall forever. Designing a fitness function
that rewards what you actually want is most of the work — see §11.3.

**Is this how real self-driving cars work?** No. It demonstrates evolutionary
optimisation of a controller in a toy environment.

---

## 15. Glossary

| Term | Meaning here |
| --- | --- |
| **Genome / chromosome** | The 100 weights and biases of one car's network |
| **Gene** | One of those 100 numbers |
| **Population** | The 100 cars evaluated together in a generation |
| **Generation** | One evaluation of the population followed by one breeding step |
| **Fitness** | The score used to compare cars (§7.1) |
| **Selection** | Choosing parents in proportion to fitness — here by tournament |
| **Crossover** | Combining two parents' genes into a child — here uniformly |
| **Mutation** | Random small changes to a child's genes |
| **Elitism** | Carrying the best individuals into the next generation unchanged |
| **Immigrant** | A brand-new random brain injected into the population |
| **Stagnation** | The champion's fitness not improving for several generations |
| **Local optimum** | A strategy that beats all nearby alternatives without being the best possible |
| **Champion** | The highest-fitness car in a generation |
| **Leader** | The highest-fitness car *still alive*; the camera follows it |
| **Dummy** | A traffic car with no brain |
| **Wall / row** | One group of traffic cars at the same position, blocking one or two lanes |

---

## 16. Provenance

The simulation core — car, road, sensor, network and visualiser — follows the
structure of Radu Mariescu-Istodor's freely published "Self-driving car — no
libraries" JavaScript course. The original repository added traffic lights, a
score display and car sprites on top of that, with a Turkish project report
(`README.md`, now translated). The genetic algorithm, fixed track, traffic
behaviour, fitness function, breeding and chart panels, speed controls,
persistence model and the test suite were written for this version.
