"use strict";

// Exercises the page's visual code paths and controls, which the evolution
// test skips because it breeds instantly (BREEDING_HOLD_FRAMES: 0).
const { loadGame, createReporter } = require("./harness");

const { read, runFrames, store } = loadGame({
    seed: 5,
    overrides: { POP_SIZE: 30, MAX_GENERATION_FRAMES: 300, STALL_FRAMES: 200 }
});

const report = createReporter("visual / panel verification");

report.section("breeding panel");

// Run until the first generation ends and the breeding hold engages.
let heldAt = -1;
for (let i = 0; i < 2000; i++) {
    runFrames(1);
    if (read("breedingHoldFrames") > 0) { heldAt = i; break; }
}
report.check("a breeding hold engages when a generation ends", heldAt >= 0, `(frame ${heldAt})`);
report.check("the hold is the configured length", read("breedingHoldFrames") === read("BREEDING_HOLD_FRAMES"));

const snap = read("JSON.parse(JSON.stringify(evolution.breedingSnapshot))");
report.check("snapshot ranks the population", Array.isArray(snap.rows) && snap.rows.length === 8);
report.check("snapshot rows are sorted by fitness descending",
    snap.rows.every((r, i) => i === 0 || snap.rows[i - 1].fitness >= r.fitness));
report.check("snapshot rows carry every fitness term",
    snap.rows.every(r => ["distance", "carsPassed", "legalStops", "lightsCleared", "crashed"].every(k => k in r)));
report.check("snapshot names the elites", snap.eliteCount === read("evolution.config.eliteCount"));
report.check("snapshot records real parent pairs",
    snap.pairs.length === 4 && snap.pairs.every(p => p.aRank >= 1 && p.bRank >= 1));
report.check("snapshot child count matches the population",
    snap.childCount === read("evolution.config.popSize") - read("evolution.config.eliteCount"));
report.check("snapshot reports the gene count and diversity",
    snap.geneCount === 124 && Number.isFinite(snap.diversity) && Number.isFinite(snap.medianFitness));

report.section("crossover gene trace");
const genes = read(`({
    origins: evolution.breedingSnapshot.geneTrace.origins.length,
    mutated: evolution.breedingSnapshot.geneTrace.mutated.length,
    distinct: new Set(evolution.breedingSnapshot.geneTrace.origins).size,
    expected: NeuralNetwork.geneCount(BRAIN_SHAPE),
    maxIndex: Math.max(...evolution.breedingSnapshot.geneTrace.mutated, -1)
})`);
report.check("one origin recorded per gene in the genome (124)",
    genes.origins === genes.expected && genes.expected === 124, `(${genes.origins} vs ${genes.expected})`);
report.check("genes come from both parents", genes.distinct === 2);
report.check("mutated indices stay inside the genome", genes.maxIndex < genes.origins);
report.check("roughly the configured share of genes mutated",
    genes.mutated > 0 && genes.mutated < genes.origins * 0.6,
    `(${genes.mutated}/${genes.origins}, rate ${read("evolution.config.mutationRate")})`);

report.section("pause freezes the breeding panel");
read("setPaused(true)");
const heldFrames = read("breedingHoldFrames");
runFrames(20);
report.check("the breeding countdown does not run while paused",
    read("breedingHoldFrames") === heldFrames, `(${heldFrames} -> ${read("breedingHoldFrames")})`);
read("setPaused(false)");

report.section("hold releases into the next generation");
const genBefore = read("evolution.generation");
// The frame that takes the counter to 0 starts the next generation and runs
// no simulation step, so the cars are untouched at that exact point.
runFrames(read("breedingHoldFrames"));
report.check("the hold clears", read("breedingHoldFrames") === 0);
report.check("a fresh generation is running",
    read("evolution.cars.every(c => c.active && c.maxDistance === 0)"));
report.check("generation counter already advanced", read("evolution.generation") === genBefore);

report.section("fitness history and metrics");
for (let i = 0; i < 4000 && read("fitnessHistory.length") < 4; i++) {
    runFrames(1);
}
const history = read("JSON.parse(JSON.stringify(fitnessHistory))");
report.check("history gains one point per generation", history.length >= 4, `(${history.length})`);
report.check("history points carry generation, stage, best, mean and median",
    history.every(p => Number.isFinite(p.generation) && typeof p.stage === "string" &&
        Number.isFinite(p.best) && Number.isFinite(p.mean) && Number.isFinite(p.median)));
report.check("history points carry the behaviour metrics",
    history.every(p => Number.isFinite(p.meanDistance) && Number.isFinite(p.meanOvertakes) &&
        Number.isFinite(p.collisionRate) && Number.isFinite(p.diversity) &&
        (p.legalStopRate === null || Number.isFinite(p.legalStopRate))));
report.check("rates with no denominator are null, not fabricated",
    history.every(p => (p.redEncounters === 0) === (p.legalStopRate === null)));
report.check("history generations are consecutive",
    history.every((p, i) => i === 0 || p.generation === history[i - 1].generation + 1));
report.check("best >= median and best >= mean in every generation",
    history.every(p => p.best >= p.mean && p.best >= p.median));
report.check("training state is persisted for reload", store.has("trainingState"));
report.check("the HUD rendered without throwing and shows the generation",
    /Generation \d+/.test(read("ui.header.textContent")));
report.check("the metrics table lists best/mean/median and the behaviour rates",
    /best fitness[\s\S]*median fitness[\s\S]*legal-stop rate[\s\S]*genome diversity/.test(read("ui.metricsBody.textContent")));

report.section("speed and pause");
read("restartTraining(true)");
read("setSpeed(20)");
report.check("speed control sets the step multiplier", read("simSpeed") === 20);
const clockBefore = read("simClockMs");
runFrames(1);
const advanced = read("simClockMs") - clockBefore;
report.check("x20 advances the sim clock ~20 steps in one frame",
    advanced > 15 * (1000 / 60), `(${advanced.toFixed(1)}ms)`);

read("restartTraining(true)");
read("setSpeed(1)");
read("setPaused(true)");
const pausedClock = read("simClockMs");
runFrames(30);
report.check("pause freezes the simulation clock", read("simClockMs") === pausedClock);
read("setPaused(false)");
runFrames(5);
report.check("resume restarts the clock", read("simClockMs") > pausedClock);

report.section("gap shading data");
report.check("every traffic row records which lanes are open",
    read(`evolution.world.trackRows.length > 0 && evolution.world.trackRows.every(row =>
        row.freeLanes.length >= 1 && row.freeLanes.length <= 2 &&
        row.cars.length + row.freeLanes.length === 3)`));
report.check("the recorded open lanes are genuinely empty in that row",
    read(`(() => {
        for (const row of evolution.world.trackRows) {
            for (const lane of row.freeLanes) {
                const gapX = road.getLaneCenter(lane);
                if (row.cars.some(c => Math.abs(c.x - gapX) < 1)) return false;
            }
        }
        return true;
    })()`));
report.check("the first wall leaves the start lane open",
    read(`evolution.world.trackRows[0].freeLanes.includes(1)`));
report.check("no two consecutive walls are identical",
    read(`evolution.world.trackRows.every((row, i) => i === 0 ||
        row.freeLanes.join() !== evolution.world.trackRows[i - 1].freeLanes.join())`));

report.section("configuration panel");
const applied = read(`applyConfig({ popSize: 12, mutationRate: 0.33, tournamentSize: 4, eliteCount: 1 }, "auto")`);
report.check("valid parameters are applied", applied.warnings.length === 0 &&
    applied.config.popSize === 12 && applied.config.mutationRate === 0.33);
report.check("the active configuration is displayed",
    /pop 12 .* mutation 0.33/.test(read("ui.configSummary.textContent")));
const bad = read(`applyConfig({ popSize: 9999, mutationRate: "nope" })`);
report.check("invalid parameters are corrected and reported",
    bad.config.popSize === 400 && bad.config.mutationRate === 0.33 && bad.warnings.length === 2 &&
    /corrections/.test(read("ui.status.textContent")));
read(`applyConfig({ popSize: 30 })`);
report.check("a new population size takes effect at the next generation", (() => {
    read(`applyConfig({ popSize: 14 })`);
    const gen = read("evolution.generation");
    for (let i = 0; i < 3000 && read("evolution.generation") === gen; i++) runFrames(1);
    runFrames(read("breedingHoldFrames") + 1);
    return read("evolution.cars.length") === 14;
})(), `(${read("evolution.cars.length")})`);
read(`applyConfig({ popSize: 30 })`);

report.section("stage selection");
read(`applyConfig({}, "B")`);
report.check("pinning stage B rebuilds the world with lights and no traffic at the next generation", (() => {
    const gen = read("evolution.generation");
    for (let i = 0; i < 3000 && read("evolution.generation") === gen; i++) runFrames(1);
    runFrames(read("breedingHoldFrames") + 1);
    return read("evolution.world.lights.length") > 0 && read("evolution.world.traffic.length") === 0 &&
        read("curriculum.stage") === "B";
})());
report.check("the mode banner and header name the stage",
    /stage B/.test(read("ui.header.textContent")) && /TRAINING/.test(read("ui.modeBanner.textContent")));
read(`applyConfig({}, "auto")`);

report.section("crash markers");
report.check("crashes are recorded once each, with coordinates",
    read(`evolution.crashMarks.length === evolution.recordedCrashes.size &&
          evolution.crashMarks.every(m => Number.isFinite(m.x) && Number.isFinite(m.y))`));

report.finish();
