"use strict";

// The simulation world: the road, the traffic, the lights, and the fixed
// track they are built from. Training, the frozen-brain demo and the headless
// benchmarks all drive a World, so a brain is judged by exactly the same
// physics and rules wherever it runs.

// The simulation runs on its own fixed-step clock rather than on wall-clock
// timestamps, so the speed control can run several steps per drawn frame and
// the traffic lights still cycle at the correct simulated rate.
const SIM_FRAME_MS = 1000 / 60;

const START_Y = 100;

// Cars roll onto the road already moving. From a standstill in an empty lane
// almost every network input is zero, so the output collapses to tanh(bias) —
// a constant. A car whose constant says "do not accelerate" never moves, so
// its inputs never change, so it never accelerates: a permanent deadlock that
// froze about half of every generation.
const START_SPEED = 1.5;

// Dummy traffic speed trades off two things through the CLOSING speed. Too
// fast (the original flat 2) and a mediocre population never catches a wall,
// so nothing selects for steering. Too slow (0.5) and a car at full speed
// closes so fast that, with 280px sensors, it sees a wall only ~100 frames
// out and cannot complete a lane change before hitting it. This band gives
// a full-speed car a ~140-frame reaction window and a wall every ~180 frames.
const DUMMY_MIN_SPEED = 0.8;
const DUMMY_MAX_SPEED = 1.2;

const LIGHT_SIZE = 44;

// The stall rule. Every STALL_FRAMES frames each car must have gained at
// least STALL_FRAMES × STALL_MIN_SPEED pixels since the last check, unless it
// spent part of that window waiting at a red light; otherwise it is retired:
// it keeps its score and stops being simulated.
//
// Two things hide in this rule. Retiring cars one by one, rather than ending
// the whole generation when the leader stalls, makes a car's fitness depend
// only on its own genome — one parked car can no longer cut every other car's
// run short (that bug made the champion's fitness fall from 3067 to 2287
// between generations despite elitism). And the minimum speed, 1.3px/frame,
// sits just above the fastest wall of traffic (1.2) and far below the car's
// cruising speed (3): a car that tucks in behind a wall and follows it is
// retired after seven seconds instead of collecting ~3450px of free distance
// over a 60-second generation. That free distance was the "parking" local
// optimum: with it, passing one wall and crashing at the next (~1500px) could
// never beat following the first wall forever, so nothing ever learned to
// overtake. Without it, every extra wall passed is the best thing a car can do.
const STALL_FRAMES = 420;
const STALL_MIN_SPEED = 1.3;

// Most walls block one lane, some block two. All-two-lane walls with the gap
// always moving were simply too hard to bootstrap: there was no easy first
// lesson, so the population settled on parking behind the first wall.
const TRACK_TWO_LANE_CHANCE = 0.4;

// Training stages. Each is a recipe for a track; the seed decides the exact
// layout. The tracks are longer than any car can drive in a generation
// (40 rows x 360px against a 60s cap at 3px/frame = 10800px).
//
// Stage A teaches steering: walls of traffic, no lights. Stage B teaches the
// lights: lights every 1100px, no traffic, so the only way to score is to
// brake for red and go on green. Stage C puts both together. The curriculum
// (ga.js) moves the population from one to the next on measured behaviour.
const SCENARIOS = Object.freeze({
    A: Object.freeze({
        key: "A",
        name: "Stage A — obstacle avoidance & overtaking",
        rows: 40, rowSpacing: 360, firstRowAhead: 350, twoLaneChance: TRACK_TWO_LANE_CHANCE,
        lights: 0, lightSpacing: 0, firstLightAhead: 0,
        routeLength: 6000
    }),
    B: Object.freeze({
        key: "B",
        name: "Stage B — traffic lights",
        rows: 0, rowSpacing: 360, firstRowAhead: 350, twoLaneChance: 0,
        lights: 12, lightSpacing: 1100, firstLightAhead: 900,
        routeLength: 6000
    }),
    C: Object.freeze({
        key: "C",
        name: "Stage C — combined driving",
        rows: 40, rowSpacing: 360, firstRowAhead: 350, twoLaneChance: TRACK_TWO_LANE_CHANCE,
        lights: 8, lightSpacing: 1800, firstLightAhead: 1500,
        routeLength: 6000
    })
});

const STAGE_ORDER = ["A", "B", "C"];

// The demonstration route. Hand-written rather than seeded, so it is the
// same every time and can be described exactly:
//
//   350px  wall in lane 0           — passes without steering (start lane open)
//   700px  wall in lane 1           — blocks the start lane: a lane change is required
//  1050px  wall in lanes 1 and 2    — only lane 0 is open
//  2350px  traffic light 1          — green 0–4s, yellow 4–6.2s, RED 6.2–22.2s, green again
//                                      a car averaging over ~1.8px/frame arrives on red
//  2800px  wall in lane 1
//  3150px  wall in lanes 0 and 2
//  3500px  wall in lane 1
//  3900px  traffic light 2          — red 18–25s, green from 25s to 45s
//  4200px  finish line
//
// Dummies drive at 0.9px/frame, inside the training band. The car is driven
// by its network alone; nothing here steers it.
const DEMO_SCENARIO = Object.freeze({
    key: "DEMO",
    name: "Demo route",
    fixed: {
        rows: [
            { ahead: 350, lanes: [0] },
            { ahead: 700, lanes: [1] },
            { ahead: 1050, lanes: [1, 2] },
            { ahead: 2800, lanes: [1] },
            { ahead: 3150, lanes: [0, 2] },
            { ahead: 3500, lanes: [1] }
        ],
        speed: 0.9,
        lights: [
            { ahead: 2350, phaseMs: 0, durations: { green: 4000, yellow: 2200, red: 16000 } },
            { ahead: 3900, phaseMs: 4200, durations: { green: 20000, yellow: 2200, red: 7000 } }
        ]
    },
    routeLength: 4200
});

function scenarioByKey(key) {
    if (key === DEMO_SCENARIO.key) {
        return DEMO_SCENARIO;
    }
    return SCENARIOS[key] || SCENARIOS.C;
}

// The track is pure data — lane layout, speed, colours and light phases —
// derived entirely from the seed and the scenario, so it can be rebuilt
// identically on reload and on every generation.
function buildTrack(seed, scenario = SCENARIOS.C) {
    const random = createSeededRandom(seed);
    const rows = [];
    const lights = [];

    if (scenario.fixed) {
        for (const row of scenario.fixed.rows) {
            rows.push({
                y: START_Y - row.ahead,
                lanes: [...row.lanes],
                colors: row.lanes.map(() => getRandomColor(random)),
                speed: scenario.fixed.speed
            });
        }
        for (const light of scenario.fixed.lights) {
            lights.push({
                y: START_Y - light.ahead,
                phaseMs: light.phaseMs,
                durations: light.durations
            });
        }
        return { seed, scenario: scenario.key, rows, lights, routeLength: scenario.routeLength };
    }

    let previousKey = "";
    for (let i = 0; i < scenario.rows; i++) {
        let lanes;
        do {
            const blockedCount = random() < scenario.twoLaneChance ? 2 : 1;
            lanes = shuffled([0, 1, 2], random).slice(0, blockedCount).sort();
            // No two consecutive walls are identical, and the first one leaves
            // the start lane open so the cars get rolling before they must steer.
        } while (lanes.join() === previousKey || (i === 0 && lanes.includes(1)));
        previousKey = lanes.join();

        rows.push({
            y: START_Y - scenario.firstRowAhead - i * scenario.rowSpacing,
            lanes,
            colors: lanes.map(() => getRandomColor(random))
        });
    }

    // Every car in a wall shares one speed, and a wall is never slower than
    // the wall behind it, so on open road no wall can catch and merge into
    // another. Per-car speeds smeared each wall into scattered cars; per-wall
    // speeds in random order let fast walls drive into slow ones, and the
    // merged pile blocked all three lanes for 61% of a generation.
    const speeds = rows
        .map(() => lerp(DUMMY_MIN_SPEED, DUMMY_MAX_SPEED, random()))
        .sort((a, b) => a - b);
    rows.forEach((row, i) => {
        row.speed = speeds[i];
    });

    for (let i = 0; i < scenario.lights; i++) {
        lights.push({
            y: START_Y - scenario.firstLightAhead - i * scenario.lightSpacing,
            phaseMs: random() * LIGHT_CYCLE_MS,
            durations: null
        });
    }

    return { seed, scenario: scenario.key, rows, lights, routeLength: scenario.routeLength };
}

class World {
    // `routeLength` null means an open-ended run (training): the generation
    // cap ends it. A finite length (evaluation, demo) adds a finish line.
    constructor(road, track, { stallFrames = STALL_FRAMES, routeLength = track.routeLength } = {}) {
        this.road = road;
        this.track = track;
        this.stallFrames = stallFrames;
        this.routeLength = routeLength;
        this.lightX = road.right + LIGHT_SIZE * 0.65;
        this.traffic = [];
        this.trackRows = [];
        this.lights = [];
        this.stopLines = [];
        this.elapsedMs = 0;
        this.frame = 0;
        this.reset();
    }

    // Every row is live from the first frame. Nothing streams in or out, so
    // the world at frame N is the same in every run regardless of where the
    // leader happens to be.
    reset() {
        this.traffic = [];
        this.trackRows = [];
        this.elapsedMs = 0;
        this.frame = 0;

        for (const row of this.track.rows) {
            const cars = row.lanes.map((lane, i) => {
                const dummy = new Car(
                    this.road.getLaneCenter(lane), row.y, 30, 50, "DUMMY", row.speed, row.colors[i]
                );
                dummy.speed = row.speed;
                return dummy;
            });
            this.traffic.push(...cars);
            this.trackRows.push({
                cars,
                freeLanes: [0, 1, 2].filter(lane => !row.lanes.includes(lane))
            });
        }

        this.lights = this.track.lights.map(light =>
            new TrafficLight(this.lightX, light.y, LIGHT_SIZE, LIGHT_SIZE, this.road,
                light.phaseMs, light.durations || undefined)
        );
        for (const light of this.lights) {
            light.update(0);
        }
        this.stopLines = this.#collectStopLines();
    }

    // A fresh AI car on the start line, rolling at START_SPEED.
    spawnCar(brain) {
        const car = new Car(this.road.getLaneCenter(1), START_Y, 30, 50, "AI");
        car.brain = brain;
        car.speed = START_SPEED;
        return car;
    }

    #collectStopLines() {
        const stopLines = [];
        for (const light of this.lights) {
            if (light.isRed) {
                stopLines.push(light.stopLine.polygon);
            }
        }
        return stopLines;
    }

    // Advance the world and the given driven cars by one fixed step.
    step(cars) {
        this.elapsedMs += SIM_FRAME_MS;
        this.frame++;

        // Lights run on world time, so each light shows the same colour at the
        // same point of every run.
        for (const light of this.lights) {
            light.update(this.elapsedMs);
        }
        this.stopLines = this.#collectStopLines();

        const aiWorld = { borders: this.road.borders, traffic: this.traffic, lights: this.lights, stopLines: this.stopLines };
        // Dummies see the traffic so they can follow the car ahead; they cannot
        // be wrecked by it (see Car.#assessDamage) and they ignore the stop lines.
        const dummyWorld = { borders: this.road.borders, traffic: this.traffic, lights: this.lights, stopLines: [] };

        for (const dummy of this.traffic) {
            dummy.update(dummyWorld);
        }
        for (const car of cars) {
            if (!car.active) {
                continue;
            }
            car.update(aiWorld);
            if (car.damaged) {
                continue;
            }
            if (this.routeLength && car.maxDistance >= this.routeLength) {
                car.finished = true;
                continue;
            }
            if (car.framesAlive - car.checkpointFrame >= this.stallFrames) {
                const gained = car.maxDistance - car.checkpointDistance;
                if (!car.waitedSinceCheckpoint && gained < this.stallFrames * STALL_MIN_SPEED) {
                    car.retired = true;
                }
                car.checkpointDistance = car.maxDistance;
                car.checkpointFrame = car.framesAlive;
                car.waitedSinceCheckpoint = false;
            }
        }
    }
}
