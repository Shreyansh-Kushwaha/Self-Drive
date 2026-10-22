"use strict";

const ROAD_VIEW_WIDTH = 260;
const NETWORK_VIEW_WIDTH = 420;
const CHART_WIDTH = 300;
const CHART_HEIGHT = 190;
const ROAD_CENTER_X = 110;
const ROAD_WIDTH = 180;
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

// Genetic algorithm parameters.
const POP_SIZE = 100;
const ELITE_COUNT = 2;

// Selection pressure is kept deliberately low and mutation high. With a
// tournament of 5 and a 0.12 mutation rate the population went clonal within
// about ten generations — mean fitness converged on the champion's and the
// search stopped exploring, locking in the first safe strategy it found.
const TOURNAMENT_SIZE = 3;
const MUTATION_RATE = 0.2;
const MUTATION_STRENGTH = 0.5;

// Stagnation escape. When the champion's fitness has not moved for this many
// generations the search is sitting in a local optimum — typically the best
// brain reaches one particular wall and nothing in the population gets past
// it. Mutation is boosted and a share of the children are replaced by random
// immigrants until the champion moves again.
const STAGNATION_LIMIT = 6;
const STAGNATION_MUTATION_BOOST = 1.5;
const STAGNATION_IMMIGRANT_SHARE = 0.2;

// A generation ends when every car is wrecked, when the leader stops making
// progress, or when it has simply run long enough.
const STALL_FRAMES = 420;
const MAX_GENERATION_FRAMES = 60 * 60;

// The simulation runs on its own fixed-step clock rather than on wall-clock
// timestamps, so the speed control can run several steps per drawn frame and
// the traffic lights still cycle at the correct simulated rate.
const FRAME_MS = 1000 / 60;
const SIM_SPEEDS = [1, 5, 20];

// How long the breeding panel is held between generations. 0 breeds instantly;
// the tests override it to 0 so a long evolutionary run stays quick.
const BREEDING_HOLD_FRAMES = 170;

const HISTORY_LIMIT = 400;

// The course is one fixed track, built from a seed and driven identically by
// every generation. A randomly streamed world made fitness noisy: the same
// brain scored 2000 on an easy layout and 800 on a hard one, so selection
// was partly a lottery and the elites' "guaranteed" survival meant little.
// On a fixed track fitness is a pure function of the genome, elitism really
// is monotonic, and a mutation that steers correctly at one wall is rewarded
// every single time instead of only when the dice agree. Discard & Restart
// rolls a new track; Load Saved Brain keeps the one the brain was bred on.
//
// The track is longer than any car can drive in a generation (40 rows x 360px
// against a 60s cap at 3px/frame = 10800px), so it never runs out.
const TRACK_ROWS = 40;
const TRACK_FIRST_ROW_AHEAD = 350;
const TRACK_ROW_SPACING = 360;

// Most walls block one lane, some block two. All-two-lane walls with the gap
// always moving were simply too hard to bootstrap: there was no easy first
// lesson, so the population settled on parking behind the first wall.
const TRACK_TWO_LANE_CHANCE = 0.4;

// Lights are deliberately sparse. At one every 1500px the first one arrived
// before the cars had met any traffic, every run ended braking at it, and
// nothing ever selected for weaving down a long stretch of road.
const TRACK_LIGHT_SPACING = 3000;
const TRACK_FIRST_LIGHT_AHEAD = 3000;

const STORE_BRAIN = "bestBrain";
const STORE_BRAIN_FORMAT = "bestBrainFormat";
const STORE_BEST_FITNESS = "bestBrainFitness";
const STORE_GENERATION = "generation";
const STORE_BEST_SCORE = "_gameBestScore";
const STORE_HISTORY = "fitnessHistory";
const STORE_TRACK_SEED = "trackSeed";

const carCanvas = document.getElementById("carCanvas");
const networkCanvas = document.getElementById("networkCanvas");
const chartCanvas = document.getElementById("chartCanvas");
const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");
const chartCtx = chartCanvas.getContext("2d");

const hud = {
    generation: document.querySelector("#generationInfo"),
    time: document.querySelector("#gameTime"),
    velocity: document.querySelector("#carVelocity"),
    alive: document.querySelector("#liveCarCount"),
    distance: document.querySelector("#bestCarDistance"),
    passed: document.querySelector("#carsPassedInfo"),
    fitness: document.querySelector("#bestFitness"),
    breakdown: document.querySelector("#fitnessBreakdown"),
    bestScore: document.querySelector("#gameBestScore")
};

const road = new Road(ROAD_CENTER_X, ROAD_WIDTH);
const LIGHT_SIZE = 44;
const LIGHT_X = road.right + LIGHT_SIZE * 0.65;
const BRAIN_SHAPE = Car.brainShape(Sensor.RAY_COUNT);

// A saved brain is only meaningful for the sensors it was bred with. The
// shape check alone let a brain trained on 150px rays reload onto 280px rays
// — same input count, every learned threshold miscalibrated — and seed the
// whole population from a driver that had learned the wrong world.
const BRAIN_FORMAT = `${BRAIN_SHAPE.join("x")}/rays${Sensor.RAY_LENGTH}/light${LIGHT_LOOKAHEAD}`;

let generation = 1;
let allTimeBest = 0;
let cars = [];
let traffic = [];
let lights = [];
let bestCar = null;
let champion = null;
let aliveCount = 0;
let frameInGeneration = 0;

let trackSeed = 0;
let track = null;
let trackRows = [];
let savedBestFitness = -Infinity;
let stagnantGenerations = 0;
let lastChampionFitness = null;
let leaderNote = null;
let framesSinceImprovement = 0;
let bestDistanceThisGen = 0;

let simClockMs = 0;
let generationStartClockMs = 0;
let simSpeed = 1;
let paused = false;

let fitnessHistory = [];
let crashMarks = [];
let recordedCrashes = new Set();
let breedingSnapshot = null;
let breedingHoldFrames = 0;
let pendingBrains = null;

// localStorage throws in private-browsing modes; the simulation should still
// run, just without remembering anything between sessions.
function readStore(key) {
    try {
        return localStorage.getItem(key);
    } catch {
        return null;
    }
}

function writeStore(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch {
        /* persistence unavailable */
    }
}

function clearStore(key) {
    try {
        localStorage.removeItem(key);
    } catch {
        /* persistence unavailable */
    }
}

function resizeCanvases() {
    carCanvas.width = ROAD_VIEW_WIDTH;
    carCanvas.height = window.innerHeight;
    networkCanvas.width = NETWORK_VIEW_WIDTH;
    networkCanvas.height = window.innerHeight;
    chartCanvas.width = CHART_WIDTH;
    chartCanvas.height = CHART_HEIGHT;
}

function loadHistory() {
    try {
        const parsed = JSON.parse(readStore(STORE_HISTORY) || "[]");
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed.filter(point => point &&
            Number.isFinite(point.generation) &&
            Number.isFinite(point.best) &&
            Number.isFinite(point.mean));
    } catch {
        return [];
    }
}

function recordHistory(best, mean) {
    fitnessHistory.push({
        generation,
        best: Math.round(best),
        mean: Math.round(mean)
    });
    if (fitnessHistory.length > HISTORY_LIMIT) {
        fitnessHistory.shift();
    }
    writeStore(STORE_HISTORY, JSON.stringify(fitnessHistory));
}

function newTrackSeed() {
    return Math.floor(Math.random() * 0xFFFFFFFF);
}

// The track is pure data — lane layout, speed, colours and light phases —
// derived entirely from the seed, so it can be rebuilt identically on reload.
function buildTrack(seed) {
    const random = createSeededRandom(seed);
    const rows = [];
    let previousKey = "";

    for (let i = 0; i < TRACK_ROWS; i++) {
        let lanes;
        do {
            const blockedCount = random() < TRACK_TWO_LANE_CHANCE ? 2 : 1;
            lanes = shuffled([0, 1, 2], random).slice(0, blockedCount).sort();
            // No two consecutive walls are identical, and the first one leaves
            // the start lane open so the cars get rolling before they must steer.
        } while (lanes.join() === previousKey || (i === 0 && lanes.includes(1)));
        previousKey = lanes.join();

        rows.push({
            y: START_Y - TRACK_FIRST_ROW_AHEAD - i * TRACK_ROW_SPACING,
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

    const lights = [];
    const trackEnd = START_Y - TRACK_FIRST_ROW_AHEAD - TRACK_ROWS * TRACK_ROW_SPACING;
    for (let y = START_Y - TRACK_FIRST_LIGHT_AHEAD; y > trackEnd; y -= TRACK_LIGHT_SPACING) {
        lights.push({ y, phaseMs: random() * LIGHT_CYCLE_MS });
    }

    return { rows, lights };
}

// Every row is live from the first frame. Nothing streams in or out, so the
// world at frame N is the same in every generation regardless of where the
// leader happens to be.
function resetWorld() {
    traffic = [];
    trackRows = [];

    for (const row of track.rows) {
        const cars = row.lanes.map((lane, i) => {
            const dummy = new Car(
                road.getLaneCenter(lane), row.y, 30, 50, "DUMMY", row.speed, row.colors[i]
            );
            dummy.speed = row.speed;
            return dummy;
        });
        traffic.push(...cars);
        trackRows.push({
            cars,
            freeLanes: [0, 1, 2].filter(lane => !row.lanes.includes(lane))
        });
    }

    lights = track.lights.map(light =>
        new TrafficLight(LIGHT_X, light.y, LIGHT_SIZE, LIGHT_SIZE, road, light.phaseMs)
    );
}

function loadSavedBrain() {
    const saved = readStore(STORE_BRAIN);
    if (!saved || readStore(STORE_BRAIN_FORMAT) !== BRAIN_FORMAT) {
        return null;
    }
    try {
        return NeuralNetwork.fromJSON(JSON.parse(saved), BRAIN_SHAPE);
    } catch {
        return null;
    }
}

// When resuming from a saved brain, this share of the population is random
// "immigrants" rather than mutants of the saved brain. A population descended
// from a single ancestor has almost no diversity, so if that ancestor sits in
// a local optimum the search can take many generations to escape it.
const IMMIGRANT_SHARE = 0.3;

// Generation 1: either a fresh random population, or the saved champion kept
// intact plus mutated copies of it plus some random immigrants.
function seedPopulation() {
    const parent = loadSavedBrain();
    const brains = [];

    if (parent) {
        const descendants = Math.round(POP_SIZE * (1 - IMMIGRANT_SHARE));
        brains.push(NeuralNetwork.clone(parent));
        while (brains.length < descendants) {
            const child = NeuralNetwork.clone(parent);
            NeuralNetwork.mutate(child, MUTATION_RATE, MUTATION_STRENGTH);
            brains.push(child);
        }
        while (brains.length < POP_SIZE) {
            brains.push(new NeuralNetwork(BRAIN_SHAPE));
        }
    } else {
        while (brains.length < POP_SIZE) {
            brains.push(new NeuralNetwork(BRAIN_SHAPE));
        }
    }

    return brains;
}

// Tournament selection: sample a few individuals and take the best of them.
// Fitness can go negative once a crash penalty applies, which rules out
// roulette-wheel selection without rescaling.
function tournamentSelect(population) {
    let winner = population[Math.floor(Math.random() * population.length)];
    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
        const challenger = population[Math.floor(Math.random() * population.length)];
        if (challenger.fitness > winner.fitness) {
            winner = challenger;
        }
    }
    return winner;
}

function buildNextPopulation(population) {
    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const brains = [];
    const pairs = [];
    let geneTrace = null;

    const stagnant = stagnantGenerations >= STAGNATION_LIMIT;
    const mutationRate = stagnant
        ? Math.min(1, MUTATION_RATE * STAGNATION_MUTATION_BOOST)
        : MUTATION_RATE;
    const mutationStrength = stagnant
        ? Math.min(1, MUTATION_STRENGTH * STAGNATION_MUTATION_BOOST)
        : MUTATION_STRENGTH;
    const immigrants = stagnant ? Math.round(POP_SIZE * STAGNATION_IMMIGRANT_SHARE) : 0;

    // Elitism: the best brains cross over untouched, so a generation can
    // never come out strictly worse than the one that produced it.
    for (let i = 0; i < ELITE_COUNT && i < ranked.length; i++) {
        brains.push(NeuralNetwork.clone(ranked[i].brain));
    }

    while (brains.length < POP_SIZE - immigrants) {
        const parentA = tournamentSelect(ranked);
        const parentB = tournamentSelect(ranked);

        // The first child's gene origins and the first few pairings are
        // recorded so the breeding panel shows what actually happened rather
        // than an illustration of it.
        const origins = geneTrace === null ? [] : null;
        const child = NeuralNetwork.crossover(parentA.brain, parentB.brain, origins);
        const mutated = origins ? [] : null;
        NeuralNetwork.mutate(child, mutationRate, mutationStrength, mutated);

        if (origins) {
            geneTrace = { origins, mutated };
        }
        if (pairs.length < 4) {
            pairs.push({
                aRank: ranked.indexOf(parentA) + 1,
                aFitness: Math.round(parentA.fitness),
                bRank: ranked.indexOf(parentB) + 1,
                bFitness: Math.round(parentB.fitness)
            });
        }

        brains.push(child);
    }

    for (let i = 0; i < immigrants; i++) {
        brains.push(new NeuralNetwork(BRAIN_SHAPE));
    }

    return { brains, pairs, geneTrace, ranked, immigrants, mutationRate };
}

function buildBreedingSnapshot(next, meanFitness) {
    const { ranked, pairs, geneTrace, immigrants, mutationRate } = next;
    return {
        generation,
        eliteCount: ELITE_COUNT,
        childCount: POP_SIZE - ELITE_COUNT - immigrants,
        tournamentSize: TOURNAMENT_SIZE,
        mutationPercent: Math.round(mutationRate * 100),
        immigrants,
        stagnantGenerations,
        meanFitness: Math.round(meanFitness),
        pairs,
        geneTrace,
        rows: ranked.slice(0, 8).map((car, index) => ({
            rank: index + 1,
            fitness: Math.round(car.fitness),
            distance: Math.round(car.maxDistance),
            carsPassed: car.carsPassed,
            lightsCleared: car.lightsCleared,
            crashed: car.damaged
        }))
    };
}

function startGeneration(brains) {
    cars = brains.map(brain => {
        const car = new Car(road.getLaneCenter(1), START_Y, 30, 50, "AI");
        car.brain = brain;
        car.speed = START_SPEED;
        return car;
    });
    bestCar = cars[0];
    champion = cars[0];
    aliveCount = cars.length;
    crashMarks = [];
    recordedCrashes = new Set();
    leaderNote = null;
    resetWorld();
    frameInGeneration = 0;
    framesSinceImprovement = 0;
    bestDistanceThisGen = 0;
    generationStartClockMs = simClockMs;
}

function endGeneration() {
    let best = cars[0];
    let total = 0;
    for (const car of cars) {
        total += car.fitness;
        if (car.fitness > best.fitness) {
            best = car;
        }
    }
    const meanFitness = total / cars.length;

    // Autosave the best brain ever seen, not merely this generation's. The old
    // build reloaded the page when everything crashed and threw the whole
    // generation away unless you had clicked Save in time; the first rewrite
    // then overwrote the save every generation, so a lucky early champion
    // could be replaced by a worse late one.
    if (best.fitness > savedBestFitness) {
        savedBestFitness = best.fitness;
        writeStore(STORE_BRAIN, JSON.stringify(best.brain));
        writeStore(STORE_BRAIN_FORMAT, BRAIN_FORMAT);
        writeStore(STORE_BEST_FITNESS, String(best.fitness));
    }
    recordHistory(best.fitness, meanFitness);

    if (lastChampionFitness !== null && Math.abs(best.fitness - lastChampionFitness) < 1) {
        stagnantGenerations++;
    } else {
        stagnantGenerations = 0;
    }
    lastChampionFitness = best.fitness;

    const next = buildNextPopulation(cars);
    breedingSnapshot = buildBreedingSnapshot(next, meanFitness);

    generation++;
    writeStore(STORE_GENERATION, String(generation));

    if (BREEDING_HOLD_FRAMES > 0) {
        pendingBrains = next.brains;
        breedingHoldFrames = BREEDING_HOLD_FRAMES;
    } else {
        startGeneration(next.brains);
    }
}

function step() {
    simClockMs += FRAME_MS;

    // Lights run on generation time, not global time, so each light shows the
    // same colour at the same point of every run.
    const generationMs = simClockMs - generationStartClockMs;
    for (const light of lights) {
        light.update(generationMs);
    }

    const stopLines = [];
    for (const light of lights) {
        if (light.isRed) {
            stopLines.push(light.stopLine.polygon);
        }
    }

    const aiWorld = { borders: road.borders, traffic, lights, stopLines };
    // Dummies see the traffic so they can follow the car ahead; they cannot be
    // wrecked by it (see Car.#assessDamage) and they ignore the stop lines.
    const dummyWorld = { borders: road.borders, traffic, lights, stopLines: [] };

    for (const car of traffic) {
        car.update(dummyWorld);
    }
    for (const car of cars) {
        car.update(aiWorld);
    }

    // The camera follows the leading survivor; it used to follow whichever car
    // was furthest up the road even if that car was a wreck.
    const previousLeader = bestCar;
    aliveCount = 0;
    let leader = null;
    champion = cars[0];
    for (const car of cars) {
        if (car.damaged) {
            if (!recordedCrashes.has(car)) {
                recordedCrashes.add(car);
                crashMarks.push({ x: car.x, y: car.y });
            }
        } else {
            aliveCount++;
            if (leader === null || car.fitness > leader.fitness) {
                leader = car;
            }
        }
        if (car.fitness > champion.fitness) {
            champion = car;
        }
    }
    bestCar = leader !== null ? leader : champion;

    // The camera jumping to another car with no explanation looked like the
    // car had teleported; say what happened.
    if (previousLeader && previousLeader.damaged && bestCar !== previousLeader && aliveCount > 0) {
        leaderNote = { text: "leader crashed — following next best car", frames: 110 };
    }

    if (champion.maxDistance > allTimeBest) {
        allTimeBest = champion.maxDistance;
        writeStore(STORE_BEST_SCORE, String(allTimeBest));
    }

    // Waiting at a red light is not being stuck, so the stall timer pauses.
    const waitingAtLight = bestCar.lightSignal > 0 && Math.abs(bestCar.speed) < 0.2;
    if (bestCar.maxDistance > bestDistanceThisGen + 1) {
        bestDistanceThisGen = bestCar.maxDistance;
        framesSinceImprovement = 0;
    } else if (waitingAtLight) {
        framesSinceImprovement = 0;
    } else {
        framesSinceImprovement++;
    }
    frameInGeneration++;

    if (aliveCount === 0 ||
        framesSinceImprovement > STALL_FRAMES ||
        frameInGeneration > MAX_GENERATION_FRAMES) {
        endGeneration();
    }
}

function updateHud() {
    const elapsedMs = simClockMs - generationStartClockMs;
    hud.generation.textContent = "Generation: " + generation;
    hud.time.textContent = "Generation time: " + (elapsedMs / 1000).toFixed(1) + " s";
    hud.velocity.textContent = "Best car velocity: " + bestCar.speed.toFixed(2);
    hud.alive.textContent = "Alive cars: " + aliveCount + " / " + cars.length;
    hud.distance.textContent = "Best distance: " + bestCar.maxDistance.toFixed(0);
    hud.passed.textContent = "Cars passed: " + bestCar.carsPassed;
    hud.fitness.textContent = "Best fitness: " + champion.fitness.toFixed(0);

    // The fitness FUNCTION is more interesting than the score, so show the sum
    // it came from. Values come from the constants so this can never drift.
    hud.breakdown.textContent =
        `${champion.maxDistance.toFixed(0)} dist` +
        ` + ${champion.carsPassed}×${OVERTAKE_BONUS} passed` +
        ` + ${champion.lightsCleared}×${LIGHT_BONUS} lights` +
        (champion.damaged ? ` − ${CRASH_PENALTY} crash` : "");

    hud.bestScore.textContent = String(Math.round(allTimeBest));
}

// A faint cross wherever a car died this generation. Later generations
// visibly die further up the road.
function drawCrashMarks(viewTop, viewBottom) {
    carCtx.strokeStyle = "rgba(0,0,0,0.35)";
    carCtx.lineWidth = 2;
    for (const mark of crashMarks) {
        if (mark.y < viewTop - 40 || mark.y > viewBottom + 40) {
            continue;
        }
        carCtx.beginPath();
        carCtx.moveTo(mark.x - 6, mark.y - 6);
        carCtx.lineTo(mark.x + 6, mark.y + 6);
        carCtx.moveTo(mark.x + 6, mark.y - 6);
        carCtx.lineTo(mark.x - 6, mark.y + 6);
        carCtx.stroke();
    }
}

// Shade the open lane(s) in each wall of traffic, so it is obvious at a
// glance that the car's task is to find the gap.
function drawGaps(viewTop, viewBottom) {
    const laneWidth = road.width / road.laneCount;
    carCtx.fillStyle = "rgba(120,240,150,0.13)";
    for (const row of trackRows) {
        const y = row.cars[0].y;
        if (y < viewTop - 60 || y > viewBottom + 60) {
            continue;
        }
        for (const lane of row.freeLanes) {
            const centre = road.getLaneCenter(lane);
            carCtx.fillRect(centre - laneWidth / 2, y - 28, laneWidth, 56);
        }
    }
}

function draw() {
    carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height);
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

    const viewTop = bestCar.y - carCanvas.height * 0.7;
    const viewBottom = viewTop + carCanvas.height;

    carCtx.save();
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7);

    road.draw(carCtx, viewTop, viewBottom);
    drawGaps(viewTop, viewBottom);
    drawCrashMarks(viewTop, viewBottom);

    for (const light of lights) {
        if (light.y > viewTop - 200 && light.y < viewBottom + 200) {
            light.draw(carCtx);
        }
    }
    for (const car of traffic) {
        if (car.y > viewTop - 60 && car.y < viewBottom + 60) {
            car.draw(carCtx);
        }
    }

    carCtx.globalAlpha = 0.2;
    for (const car of cars) {
        if (car !== bestCar) {
            car.draw(carCtx);
        }
    }
    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, true);

    carCtx.restore();

    if (leaderNote) {
        carCtx.fillStyle = "rgba(0,0,0,0.72)";
        carCtx.fillRect(0, 8, carCanvas.width, 24);
        carCtx.fillStyle = "white";
        carCtx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
        carCtx.textAlign = "center";
        carCtx.textBaseline = "middle";
        carCtx.fillText(leaderNote.text, carCanvas.width / 2, 20);
        // Counted in drawn frames, not simulation steps, so it stays readable
        // at x20 as well.
        leaderNote.frames--;
        if (leaderNote.frames <= 0) {
            leaderNote = null;
        }
    }

    if (breedingHoldFrames > 0 && breedingSnapshot) {
        Panels.drawBreeding(networkCtx, breedingSnapshot);
    } else {
        networkCtx.lineDashOffset = -simClockMs / 50;
        Visualizer.drawNetwork(networkCtx, bestCar.brain);
    }

    Panels.drawFitnessHistory(chartCtx, fitnessHistory);
    updateHud();
}

function animate() {
    // Pause freezes the breeding countdown as well as the simulation, so the
    // breeding panel can be held on screen for as long as it takes to talk
    // through it. Drawing continues either way.
    if (!paused) {
        if (breedingHoldFrames > 0) {
            breedingHoldFrames--;
            if (breedingHoldFrames === 0 && pendingBrains) {
                startGeneration(pendingBrains);
                pendingBrains = null;
            }
        } else {
            for (let i = 0; i < simSpeed; i++) {
                step();
                // A generation just ended and the breeding panel is up: stop
                // fast-forwarding so it stays readable.
                if (breedingHoldFrames > 0) {
                    break;
                }
            }
        }
    }

    draw();
    requestAnimationFrame(animate);
}

function restartTraining(fromSaved) {
    if (!fromSaved) {
        clearStore(STORE_BRAIN);
        clearStore(STORE_BRAIN_FORMAT);
        clearStore(STORE_BEST_FITNESS);
        clearStore(STORE_GENERATION);
        clearStore(STORE_BEST_SCORE);
        clearStore(STORE_HISTORY);
        generation = 1;
        allTimeBest = 0;
        savedBestFitness = -Infinity;
        fitnessHistory = [];
        // A fresh start is also a fresh track; a saved brain is only reloaded
        // onto the track it was bred on.
        trackSeed = newTrackSeed();
        writeStore(STORE_TRACK_SEED, String(trackSeed));
        track = buildTrack(trackSeed);
    }
    breedingHoldFrames = 0;
    breedingSnapshot = null;
    pendingBrains = null;
    startGeneration(seedPopulation());
}

const pauseButton = document.querySelector("#pauseButton");

function setPaused(value) {
    paused = value;
    pauseButton.textContent = paused ? "Resume" : "Pause";
}

function setSpeed(value) {
    simSpeed = value;
    for (const button of document.querySelectorAll(".speedButton")) {
        button.classList.toggle("activeSpeed", Number(button.dataset.speed) === value);
    }
}

document.querySelector("#saveButton").addEventListener("click", () => {
    savedBestFitness = bestCar.fitness;
    writeStore(STORE_BRAIN, JSON.stringify(bestCar.brain));
    writeStore(STORE_BRAIN_FORMAT, BRAIN_FORMAT);
    writeStore(STORE_BEST_FITNESS, String(bestCar.fitness));
});

document.querySelector("#loadButton").addEventListener("click", () => {
    restartTraining(true);
});

document.querySelector("#discardButton").addEventListener("click", () => {
    restartTraining(false);
});

pauseButton.addEventListener("click", () => setPaused(!paused));

for (const button of document.querySelectorAll(".speedButton")) {
    button.addEventListener("click", () => setSpeed(Number(button.dataset.speed)));
}

window.addEventListener("resize", resizeCanvases);

generation = Number(readStore(STORE_GENERATION)) || 1;
allTimeBest = Number(readStore(STORE_BEST_SCORE)) || 0;
fitnessHistory = loadHistory();

const storedFitness = readStore(STORE_BEST_FITNESS);
savedBestFitness = storedFitness === null ? -Infinity : Number(storedFitness);

trackSeed = Number(readStore(STORE_TRACK_SEED)) || newTrackSeed();
writeStore(STORE_TRACK_SEED, String(trackSeed));
track = buildTrack(trackSeed);

resizeCanvases();
setPaused(false);
setSpeed(SIM_SPEEDS[0]);
startGeneration(seedPopulation());
requestAnimationFrame(animate);
