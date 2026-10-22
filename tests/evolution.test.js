"use strict";

// End-to-end test: runs the real animation loop for a number of shortened
// generations and checks that the population actually evolves, that the loop
// never dies, and that no state goes non-finite.
//
// Generations and population are shrunk so this finishes in well under a
// minute; the seeded RNG in the harness makes the run reproducible.

const { loadGame, createReporter } = require("./harness");

const GENERATIONS_WANTED = 18;
const FRAMES_PER_GENERATION = 700;

const { read, runFrames, store } = loadGame({
    seed: 7,
    overrides: {
        POP_SIZE: 40,
        MAX_GENERATION_FRAMES: FRAMES_PER_GENERATION,
        STALL_FRAMES: 240,
        // Breed instantly instead of holding the breeding panel on screen.
        BREEDING_HOLD_FRAMES: 0
    }
});

const report = createReporter("evolution test");

// Pinned to stage A: 700-frame generations are too short to reach the first
// light of stage C, so the lights would only add a hazard the run cannot
// learn; the stage C track itself is checked separately below.
read(`applyConfig({}, "A"); restartTraining(false); applyConfig({}, "A");`);

const SAMPLE = `(() => {
    let best = -Infinity, sum = 0, alive = 0, cleared = 0, distance = 0, passed = 0;
    for (const car of evolution.cars) {
        if (car.fitness > best) best = car.fitness;
        if (car.maxDistance > distance) distance = car.maxDistance;
        if (car.lightsCleared > cleared) cleared = car.lightsCleared;
        if (car.carsPassed > passed) passed = car.carsPassed;
        if (!car.damaged) alive++;
        sum += car.fitness;
    }
    return {
        best, mean: sum / evolution.cars.length, alive, distance, passed,
        lightsCleared: cleared, traffic: evolution.world.traffic.length, lightCount: evolution.world.lights.length
    };
})()`;

const history = [];
// World snapshot at the same frame of each generation. On a fixed track these
// must be identical, otherwise fitness is partly luck.
const layouts = {};
let previousGeneration = read("evolution.generation");
let sample = null;
let nonFiniteAt = -1;
let loopDied = false;

// Traffic integrity, sampled every few frames. Dummies cannot collide with
// each other, so the only thing stopping a faster wall from driving through
// a slower one is the track design and the car-following rule.
let trafficScans = 0;
let overlapFrames = 0;
let fullWallFrames = 0;
const TRAFFIC_SCAN = `(() => {
    const traffic = evolution.world.traffic, lights = evolution.world.lights;
    let overlap = false, fullWall = false;
    for (let i = 0; i < traffic.length && !overlap; i++) {
        for (let j = i + 1; j < traffic.length; j++) {
            if (Math.abs(traffic[i].y - traffic[j].y) < 50 && Math.abs(traffic[i].x - traffic[j].x) < 30) {
                overlap = true;
                break;
            }
        }
    }
    const laneOf = (x) => Math.round((x - road.left) / (road.width / 3) - 0.5);
    for (const d of traffic) {
        // Three abreast is legitimate only while queued at a light.
        if (lights.some(l => Math.abs(l.y - d.y) < 200)) continue;
        const lanes = new Set(traffic.filter(o => Math.abs(o.y - d.y) < 60).map(o => laneOf(o.x)));
        if (lanes.size === 3) { fullWall = true; break; }
    }
    return { overlap, fullWall };
})()`;

const totalFrames = GENERATIONS_WANTED * (FRAMES_PER_GENERATION + 2);
try {
    runFrames(totalFrames, (frame) => {
        if (frame % 12 === 0) {
            const scan = read(TRAFFIC_SCAN);
            trafficScans++;
            if (scan.overlap) overlapFrames++;
            if (scan.fullWall) fullWallFrames++;
        }
        const generation = read("evolution.generation");
        if (generation !== previousGeneration) {
            history.push({ generation: previousGeneration, ...sample });
            previousGeneration = generation;
        }
        if (read("evolution.frameInGeneration") === 150) {
            layouts[generation] = read(`
                evolution.world.traffic.map(d => Math.round(d.y) + ":" + Math.round(d.x)).join("|") +
                "/" + evolution.world.lights.map(l => Math.round(l.y) + ":" + l.state).join("|")`);
        }
        sample = read(SAMPLE);
        const state = read("[evolution.bestCar.x, evolution.bestCar.y, evolution.bestCar.speed, evolution.bestCar.fitness, evolution.bestCar.lightSignal]");
        if (nonFiniteAt < 0 && state.some((value) => !Number.isFinite(value))) {
            nonFiniteAt = generation;
        }
    });
} catch (error) {
    loopDied = true;
    console.log(`  loop error: ${error.message}\n${error.stack}`);
}

report.section("stability");
report.check("the animation loop kept running for every frame", !loopDied);
report.check("no car state ever became NaN or Infinity", nonFiniteAt < 0, `(first at generation ${nonFiniteAt})`);
report.check("generations advanced without a page reload", history.length >= GENERATIONS_WANTED - 2,
    `(saw ${history.length})`);
report.check("every generation drives the same fixed track",
    history.length > 0 && history[0].traffic > 0 &&
    history.every((h) => h.traffic === history[0].traffic),
    `(traffic counts ${[...new Set(history.map((h) => h.traffic))].join(", ")})`);
report.check("stage A has no lights; stages B and C put lights on the same seed's track",
    history.every((h) => h.lightCount === 0) &&
    read("buildTrack(trackSeed, SCENARIOS.B).lights.length") > 0 &&
    read("buildTrack(trackSeed, SCENARIOS.C).lights.length") > 0 &&
    read("buildTrack(trackSeed, SCENARIOS.C).rows.length") > 0);
report.check("traffic cars never overlap each other",
    trafficScans > 0 && overlapFrames === 0,
    `(${overlapFrames} of ${trafficScans} sampled frames)`);
report.check("no wall on open road ever blocks all three lanes",
    trafficScans > 0 && fullWallFrames === 0,
    `(${fullWallFrames} of ${trafficScans} sampled frames)`);
report.check("every fitness in the history is finite",
    history.every(h => Number.isFinite(h.best) && Number.isFinite(h.mean)));

const layoutGens = Object.keys(layouts).map(Number).sort((a, b) => a - b);
report.check("the world is identical at the same frame of every generation",
    layoutGens.length >= 3 &&
    layoutGens.every((g) => layouts[g] === layouts[layoutGens[0]]),
    `(compared generations ${layoutGens.join(", ")})`);

report.section("persistence");
report.check("the champion brain is autosaved", store.has("champion"));
report.check("the saved champion records its format, stage, track seed and generation", (() => {
    const record = JSON.parse(store.get("champion"));
    return record.format === read("BRAIN_FORMAT") && record.stage === "A" &&
        record.trackSeed === read("trackSeed") && Number.isFinite(record.generation) &&
        Number.isFinite(record.fitness) && record.brain && record.brain.neuronCounts.join() === "10,8,4";
})());
report.check("the training state is persisted with the generation counter",
    JSON.parse(store.get("trainingState")).generation === read("evolution.generation"));
report.check("the saved brain reloads into a usable network", read(`(() => {
    const revived = loadChampionRecord().brain;
    if (!revived) return false;
    const outputs = NeuralNetwork.feedForward(new Array(BRAIN_SHAPE[0]).fill(0.4), revived);
    return outputs.length === 4 && outputs.every(Number.isFinite);
})()`));
report.check("a brain saved under a different format is refused with a clear message", read(`(() => {
    const original = localStorage.getItem("champion");
    const record = JSON.parse(original);
    record.format = "7x8x4/rays150/light260";
    localStorage.setItem("champion", JSON.stringify(record));
    const refused = loadChampionRecord();
    localStorage.setItem("champion", original);
    return refused.brain === null && /older network/.test(refused.message) && loadChampionRecord().brain !== null;
})()`));
report.check("the champion is never replaced by a worse brain of the same stage", read(`(() => {
    const fitnesses = fitnessHistory.map(p => p.best);
    return JSON.parse(localStorage.getItem("champion")).fitness >= Math.max(...fitnesses) - 1;
})()`));

report.section("evolution");
const third = Math.floor(history.length / 3);
const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const firstThird = history.slice(0, third);
const lastThird = history.slice(-third);
const meanStart = mean(firstThird.map((h) => h.mean));
const meanEnd = mean(lastThird.map((h) => h.mean));
const bestStart = mean(firstThird.map((h) => h.best));
const bestEnd = mean(lastThird.map((h) => h.best));

report.check("mean population fitness improves over the run", meanEnd > meanStart,
    `(${meanStart.toFixed(1)} -> ${meanEnd.toFixed(1)})`);
report.check("the improvement is substantial, not noise", meanEnd > meanStart * 1.5 + 50,
    `(${meanStart.toFixed(1)} -> ${meanEnd.toFixed(1)})`);
// Checked on the page's own end-of-generation summaries: the per-frame samples
// above are taken one frame before the generation-ending step, so they can be
// a pixel or two short.
const recordedBests = read("fitnessHistory.map(p => p.best)");
report.check("champion fitness never decreases (elitism on a fixed track)",
    recordedBests.length >= GENERATIONS_WANTED - 2 &&
    recordedBests.every((best, i) => i === 0 || best >= recordedBests[i - 1]),
    `(${recordedBests.join(" ")})`);
report.check("champion fitness does not collapse (elitism holds)", bestEnd >= bestStart * 0.8,
    `(${bestStart.toFixed(1)} -> ${bestEnd.toFixed(1)})`);

console.log("\nper-generation results");
for (const h of history) {
    console.log(`  gen ${String(h.generation).padStart(3)}` +
        `  best ${h.best.toFixed(0).padStart(7)}` +
        `  mean ${h.mean.toFixed(0).padStart(7)}` +
        `  survivors ${String(h.alive).padStart(3)}` +
        `  passed ${h.passed}` +
        `  lights cleared ${h.lightsCleared}` +
        `  traffic ${h.traffic}`);
}
console.log(`\nmean population fitness: ${meanStart.toFixed(1)} -> ${meanEnd.toFixed(1)}`);
console.log(`champion fitness:        ${bestStart.toFixed(1)} -> ${bestEnd.toFixed(1)}`);

report.finish();
