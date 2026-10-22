// Exercises the new visual code paths, which the normal test suite skips
// because it breeds instantly (BREEDING_HOLD_FRAMES: 0).
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

const snap = read("JSON.parse(JSON.stringify(breedingSnapshot))");
report.check("snapshot ranks the population", Array.isArray(snap.rows) && snap.rows.length === 8);
report.check("snapshot rows are sorted by fitness descending",
    snap.rows.every((r, i) => i === 0 || snap.rows[i - 1].fitness >= r.fitness));
report.check("snapshot names the elites", snap.eliteCount === read("ELITE_COUNT"));
report.check("snapshot records real parent pairs",
    snap.pairs.length === 4 && snap.pairs.every(p => p.aRank >= 1 && p.bRank >= 1));
report.check("snapshot child count matches the population",
    snap.childCount === read("POP_SIZE") - read("ELITE_COUNT"));

report.section("crossover gene trace");
const genes = read(`({
    origins: breedingSnapshot.geneTrace.origins.length,
    mutated: breedingSnapshot.geneTrace.mutated.length,
    distinct: new Set(breedingSnapshot.geneTrace.origins).size,
    expected: BRAIN_SHAPE[0]*BRAIN_SHAPE[1] + BRAIN_SHAPE[1] + BRAIN_SHAPE[1]*BRAIN_SHAPE[2] + BRAIN_SHAPE[2],
    maxIndex: Math.max(...breedingSnapshot.geneTrace.mutated, -1)
})`);
report.check("one origin recorded per gene in the genome",
    genes.origins === genes.expected, `(${genes.origins} vs ${genes.expected})`);
report.check("genes come from both parents", genes.distinct === 2);
report.check("mutated indices stay inside the genome", genes.maxIndex < genes.origins);
report.check("roughly the configured share of genes mutated",
    genes.mutated > 0 && genes.mutated < genes.origins * 0.6,
    `(${genes.mutated}/${genes.origins}, rate ${read("MUTATION_RATE")})`);

report.section("pause freezes the breeding panel");
read("setPaused(true)");
const heldFrames = read("breedingHoldFrames");
runFrames(20);
report.check("the breeding countdown does not run while paused",
    read("breedingHoldFrames") === heldFrames, `(${heldFrames} -> ${read("breedingHoldFrames")})`);
read("setPaused(false)");

report.section("hold releases into the next generation");
const genBefore = read("generation");
// The frame that takes the counter to 0 starts the next generation and runs
// no simulation step, so the cars are untouched at that exact point.
runFrames(read("breedingHoldFrames"));
report.check("the hold clears", read("breedingHoldFrames") === 0);
report.check("a fresh generation is running",
    read("cars.every(c => !c.damaged && c.maxDistance === 0)"));
report.check("generation counter already advanced", read("generation") === genBefore);

report.section("fitness history");
for (let i = 0; i < 4000 && read("fitnessHistory.length") < 4; i++) {
    runFrames(1);
}
const history = read("JSON.parse(JSON.stringify(fitnessHistory))");
report.check("history gains one point per generation", history.length >= 4, `(${history.length})`);
report.check("history points carry generation, best and mean",
    history.every(p => Number.isFinite(p.generation) && Number.isFinite(p.best) && Number.isFinite(p.mean)));
report.check("history generations are consecutive",
    history.every((p, i) => i === 0 || p.generation === history[i - 1].generation + 1));
report.check("history is persisted for reload", store.has("fitnessHistory"));
report.check("best is never below mean", history.every(p => p.best >= p.mean));

report.section("speed and pause");
// Start from a clean running generation rather than poking internals.
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
    read(`trackRows.length > 0 && trackRows.every(row =>
        row.freeLanes.length >= 1 && row.freeLanes.length <= 2 &&
        row.cars.length + row.freeLanes.length === 3)`));
report.check("the recorded open lanes are genuinely empty in that row",
    read(`(() => {
        for (const row of trackRows) {
            for (const lane of row.freeLanes) {
                const gapX = road.getLaneCenter(lane);
                if (row.cars.some(c => Math.abs(c.x - gapX) < 1)) return false;
            }
        }
        return true;
    })()`));
report.check("the first wall leaves the start lane open",
    read(`trackRows[0].freeLanes.includes(1)`));
report.check("no two consecutive walls are identical",
    read(`trackRows.every((row, i) => i === 0 ||
        row.freeLanes.join() !== trackRows[i - 1].freeLanes.join())`));

report.section("stagnation escape");
report.check("no immigrants while the champion is still improving",
    read(`(() => {
        stagnantGenerations = 0;
        const next = buildNextPopulation(cars);
        return next.immigrants === 0 && next.mutationRate === MUTATION_RATE &&
               next.brains.length === POP_SIZE;
    })()`));
report.check("random immigrants and boosted mutation once fitness has been flat",
    read(`(() => {
        stagnantGenerations = STAGNATION_LIMIT;
        const next = buildNextPopulation(cars);
        stagnantGenerations = 0;
        return next.immigrants === Math.round(POP_SIZE * STAGNATION_IMMIGRANT_SHARE) &&
               next.brains.length === POP_SIZE && next.mutationRate > MUTATION_RATE;
    })()`));

report.section("crash markers");
report.check("crashes are recorded once each, with coordinates",
    read(`crashMarks.length === recordedCrashes.size &&
          crashMarks.every(m => Number.isFinite(m.x) && Number.isFinite(m.y))`));

report.finish();
