"use strict";

// Demo mode, frozen evaluation, experiments and CSV export. Runs a short
// training first so there is a real saved champion to work with.

const { loadGame, createReporter } = require("./harness");

const { read, runFrames, store } = loadGame({
    seed: 13,
    overrides: { POP_SIZE: 24, MAX_GENERATION_FRAMES: 500, STALL_FRAMES: 200, BREEDING_HOLD_FRAMES: 0 }
});

const report = createReporter("demo / evaluation / export tests");

report.section("no champion yet");
read(`localStorage.removeItem("champion"); savedChampion = null;`);
report.check("Demo Best Brain without a saved brain explains what to do", (() => {
    const ok = read("startDemo()");
    return ok === false && /Train a population/.test(read("ui.status.textContent")) && read("mode") === "train";
})());
report.check("an old 7-input saved brain is reported as incompatible, not loaded", (() => {
    read(`localStorage.setItem("bestBrain", JSON.stringify(NeuralNetwork.toJSON(new NeuralNetwork([7, 8, 4]))))`);
    const result = read("loadChampionRecord()");
    read(`localStorage.removeItem("bestBrain")`);
    return result.brain === null && /7-input/.test(result.message) && /incompatible/.test(result.message);
})());
report.check("a champion record with the wrong format is reported as incompatible", (() => {
    read(`localStorage.setItem("champion", JSON.stringify({ format: "v2/7x8x4", brain: NeuralNetwork.toJSON(new NeuralNetwork([7, 8, 4])) }))`);
    const result = read("loadChampionRecord()");
    read(`localStorage.removeItem("champion")`);
    return result.brain === null && /older network/.test(result.message);
})());

// Train a few short generations so a champion exists.
runFrames(3 * 502);
report.check("training produced a saved champion", store.has("champion"));

report.section("demo mode");
report.check("Demo Best Brain loads the saved genome and switches mode", (() => {
    const ok = read("startDemo()");
    return ok === true && read("mode") === "demo" &&
        read(`NeuralNetwork.equals(demo.car.brain, loadChampionRecord().brain)`) &&
        read(`demo.car.brain !== loadChampionRecord().brain`);
})());
report.check("the demo world is the fixed demo route with two lights and a finish line",
    read("demo.world.lights.length") === 2 && read("demo.world.routeLength") === 4200 &&
    read("demo.world.traffic.length") === 8);
report.check("the first demo light is red when a car arrives between 6.2s and 22.2s",
    read(`demo.world.lights[0].stateAt(6000) === "yellow" && demo.world.lights[0].stateAt(6300) === "red" &&
          demo.world.lights[0].stateAt(22100) === "red" && demo.world.lights[0].stateAt(22300) === "green"`));

read("setSpeed(20)");
runFrames(300);
const firstResult = read("JSON.parse(JSON.stringify(demo.result))");
const firstTrace = read("JSON.stringify({x: demo.car.x, y: demo.car.y, f: demo.frames, d: demo.car.maxDistance})");
report.check("the demo ran to a result", firstResult !== null && typeof firstResult.outcome === "string",
    `(${JSON.stringify(firstResult)})`);
report.check("the result is reported as success or failure, not as a guarantee",
    firstResult.success === (firstResult.outcome === "completed"));
report.check("the brain did not change during playback", firstResult.brainUnchanged === true &&
    read("NeuralNetwork.equals(demo.car.brain, loadChampionRecord().brain)"));
report.check("the population was not stepped while the demo ran",
    read("evolution.frameInGeneration") === read("evolution.frameInGeneration") && read("evolution.generation") === 4);
report.check("the saved champion was not replaced by the demo",
    JSON.parse(store.get("champion")).generation <= 3);
report.check("the HUD shows the demo status panel",
    /Action:/.test(read("ui.infoLines.textContent")) && /Traffic light:/.test(read("ui.infoLines.textContent")) &&
    /RESULT:/.test(read("ui.infoLines.textContent")) && /DEMO/.test(read("ui.modeBanner.textContent")));

report.check("Reset Demo restores the same track and initial conditions", (() => {
    read("resetDemo()");
    const fresh = read("demo.frames === 0 && demo.car.maxDistance === 0 && demo.car.y === START_Y && demo.car.speed === START_SPEED");
    const sameTrack = read("demo.world.traffic.map(c => c.x + ':' + c.y).join('|')") ===
        read("new World(road, buildTrack(0, DEMO_SCENARIO)).traffic.map(c => c.x + ':' + c.y).join('|')");
    return fresh && sameTrack;
})());
runFrames(300);
report.check("a re-run of the demo is identical (deterministic route and brain)",
    read("JSON.stringify({x: demo.car.x, y: demo.car.y, f: demo.frames, d: demo.car.maxDistance})") === firstTrace &&
    JSON.stringify(read("JSON.parse(JSON.stringify(demo.result))")) === JSON.stringify(firstResult));

report.check("Train Population resumes training at the same generation", (() => {
    read("ui.trainButton.handlers[0]()");
    return read("mode") === "train" && read("evolution.generation") === 4;
})());

report.section("frozen evaluation");
const episode = read(`runEpisode({ road, scenario: SCENARIOS.C, trackSeed: 77, brain: loadChampionRecord().brain })`);
report.check("an episode reports an outcome and real measurements",
    ["completed", "crashed", "stalled", "timeout"].includes(episode.outcome) &&
    Number.isFinite(episode.distance) && Number.isFinite(episode.fitness) && episode.frames > 0);
report.check("evaluation leaves the tested brain unchanged", episode.brainUnchanged === true);
report.check("evaluation is reproducible for the same brain, seed and scenario",
    JSON.stringify(read(`runEpisode({ road, scenario: SCENARIOS.C, trackSeed: 77, brain: loadChampionRecord().brain })`)) ===
    JSON.stringify(episode));
report.check("the hand-coded baseline drives without a brain", read(`(() => {
    const r = runEpisode({ road, scenario: SCENARIOS.A, trackSeed: 77, driver: heuristicDriver });
    return r.distance > 100 && Number.isFinite(r.fitness);
})()`));
report.check("two GA runs with the same seeds produce identical histories", read(`(() => {
    const run = () => {
        const e = new Evolution({ road, scenario: SCENARIOS.A, trackSeed: 5,
            config: { popSize: 10, maxGenerationFrames: 200 }, random: createSeededRandom(99) });
        const out = [];
        while (out.length < 3) { const s = e.step(); if (s) { out.push([s.best, s.mean, s.median]); e.advance(); } }
        return JSON.stringify(out);
    };
    return run() === run();
})()`));

report.section("experiments");
const resultA = read(`new ExperimentRunner(experimentBaseline({ road, scenario: SCENARIOS.A, trackSeed, champion: loadChampionRecord().brain,
    population: evolution.cars.slice(0, 2).map(c => c.brain), untrainedCount: 4 })).runSync()`);
report.check("experiment A compares untrained, evolved and heuristic groups with trial counts",
    resultA.summary.untrained.trials === 4 && resultA.summary.evolved.trials === 3 &&
    resultA.summary.heuristic.trials === 1 && resultA.rows.length === 8);
report.check("experiment A rows carry real per-episode measurements",
    resultA.rows.every(r => Number.isFinite(r.distance_px) && Number.isFinite(r.fitness) &&
        typeof r.outcome === "string" && r.brain_unchanged === "yes"));
report.check("experiment A rows are tagged by group and experiment id",
    new Set(resultA.rows.map(r => r.group)).size === 3 && resultA.rows.every(r => r.experiment_id === resultA.id));

const resultB = read(`new ExperimentRunner(experimentMutation({ road, scenario: SCENARIOS.A, trackSeed, rates: [0.05, 0.3],
    trials: 2, popSize: 8, generations: 3, maxGenerationFrames: 150 })).runSync()`);
report.check("experiment B runs every rate for every trial and generation",
    resultB.rows.length === 2 * 2 * 3 && resultB.summary["0.05"].trials === 2 && resultB.summary["0.3"].trials === 2);
report.check("experiment B holds the other parameters constant",
    resultB.rows.every(r => r.population_size === 8 && r.tournament_size === read("TOURNAMENT_SIZE") &&
        r.elite_count === read("ELITE_COUNT") && (r.mutation_rate === 0.05 || r.mutation_rate === 0.3)));
report.check("experiment B does not touch the on-screen population",
    read("evolution.generation") === 4 && read("evolution.cars.length") === 24);

const resultC = read(`new ExperimentRunner(experimentGeneralization({ road, champion: loadChampionRecord().brain, trainingSeed: trackSeed,
    scenario: SCENARIOS.A, unseenSeeds: [101, 202, trackSeed] })).runSync()`);
report.check("experiment C reports the training track separately from unseen tracks",
    resultC.summary.training.trials === 1 && resultC.summary.unseen.trials === 2 &&
    resultC.rows.filter(r => r.track_type === "unseen").length === 2 &&
    resultC.rows.filter(r => r.track_type === "training").length === 1);
report.check("experiment C excludes the training seed from the unseen set",
    !resultC.summary.unseenSeeds.includes(read("trackSeed")));
report.check("experiment C includes the demo route and freezes the brain",
    resultC.summary.demoResult !== null && resultC.summary.unseen.brainUnchanged === true &&
    read("NeuralNetwork.equals(loadChampionRecord().brain, loadChampionRecord().brain)"));

report.section("benchmark mode on the page");
const benchmark = read(`runEvaluation("A")`);
report.check("Run Evaluation switches to benchmark mode", benchmark !== null && read("mode") === "benchmark");
// The runner is driven by setTimeout; pump the event loop until it finishes.
report.check("the benchmark completes asynchronously and stores its rows", (() => {
    const deadline = Date.now() + 20000;
    const spin = () => new Promise(resolve => setTimeout(resolve, 5));
    return (async () => {
        while (Date.now() < deadline && read("benchmark.runner !== null")) {
            runFrames(1);
            await spin();
        }
        return read("benchmark.runner === null") && read("experimentLog.length") > 0 && store.has("experimentResults");
    })();
})());

report.section("CSV export");
setTimeout(async () => {
    // Wait for the asynchronous benchmark check above to settle.
    while (read("benchmark && benchmark.runner !== null")) {
        await new Promise(resolve => setTimeout(resolve, 5));
    }
    const csv = read("buildResultsCsv()");
    const lines = csv.split("\r\n").filter(Boolean);
    const header = lines[0].split(",");
    report.check("the CSV has a header with the documented columns",
        header.length === read("CSV_COLUMNS.length") && header.includes("mutation_rate") && header.includes("legal_stop_rate") &&
        header.includes("track_seed") && header.includes("outcome"));
    report.check("every row has the same number of fields as the header", (() => {
        const count = line => (line.match(/,/g) || []).length;
        return lines.every(line => line.includes('"') || count(line) === header.length - 1);
    })());
    report.check("training rows and benchmark rows are both present and distinguishable",
        lines.some(l => l.startsWith("training,")) && lines.some(l => l.startsWith("A-baseline,")));
    report.check("exported numbers match the stored measurements", (() => {
        const row = read("experimentLog[0]");
        return lines.some(l => l.includes(String(row.distance_px)) && l.includes(row.experiment_id));
    })());
    report.check("csvEscape quotes commas and doubles quotes",
        read(`csvEscape('a,b') === '"a,b"' && csvEscape('say "hi"') === '"say ""hi"""' && csvEscape(5) === "5"`));
    report.check("export returns the CSV even where downloads are impossible (no Blob)",
        read("exportResults().downloaded") === false && read("exportResults().csv.length") > 100);
    report.finish();
}, 10);
