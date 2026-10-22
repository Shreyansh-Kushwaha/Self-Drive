"use strict";

// The page: wires the genetic algorithm (ga.js), the world (world.js) and the
// benchmarks (evaluate.js) to the canvases and buttons. Three modes:
//   train     — the population evolves on screen
//   demo      — one frozen saved brain drives the fixed demo route
//   benchmark — experiments run headless in slices while a progress panel shows

const ROAD_VIEW_WIDTH = 260;
const NETWORK_VIEW_WIDTH = 420;
const NETWORK_VIEW_HEIGHT = 520;
const CHART_WIDTH = 300;
const CHART_HEIGHT = 190;
const ROAD_CENTER_X = 110;
const ROAD_WIDTH = 180;

const SIM_SPEEDS = [1, 5, 20];

// How long the breeding panel is held between generations. 0 breeds instantly;
// the tests override it to 0 so a long evolutionary run stays quick.
const BREEDING_HOLD_FRAMES = 170;

const HISTORY_LIMIT = 400;
const RESULTS_LIMIT = 5000;
const DEMO_MAX_FRAMES = 90 * 60;

// Persistence. A saved champion is one JSON record carrying the genome and
// the conditions it was bred under; the training state is a second record.
const STORE_CHAMPION = "champion";
const STORE_TRAINING = "trainingState";
const STORE_RESULTS = "experimentResults";
// Keys written by earlier builds (7-input brains). They cannot be loaded.
const LEGACY_BRAIN_KEY = "bestBrain";
const LEGACY_KEYS = ["bestBrain", "bestBrainFormat", "bestBrainFitness", "generation",
    "_gameBestScore", "fitnessHistory", "trackSeed"];

const carCanvas = document.getElementById("carCanvas");
const networkCanvas = document.getElementById("networkCanvas");
const chartCanvas = document.getElementById("chartCanvas");
const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");
const chartCtx = chartCanvas.getContext("2d");

const ui = {
    modeBanner: document.querySelector("#modeBanner"),
    header: document.querySelector("#gameInfoHeader"),
    infoLines: document.querySelector("#infoLines"),
    metricsBody: document.querySelector("#metricsTable tbody"),
    bestScore: document.querySelector("#gameBestScore"),
    status: document.querySelector("#statusMessage"),
    results: document.querySelector("#benchmarkResults"),
    resultsCount: document.querySelector("#resultsCount"),
    configSummary: document.querySelector("#configSummary"),
    pauseButton: document.querySelector("#pauseButton"),
    trainButton: document.querySelector("#trainButton"),
    demoButton: document.querySelector("#demoButton"),
    experimentSelect: document.querySelector("#experimentSelect"),
    stageSelect: document.querySelector("#cfgStage"),
    config: {
        popSize: document.querySelector("#cfgPopSize"),
        mutationRate: document.querySelector("#cfgMutationRate"),
        mutationStrength: document.querySelector("#cfgMutationStrength"),
        tournamentSize: document.querySelector("#cfgTournamentSize"),
        eliteCount: document.querySelector("#cfgEliteCount")
    }
};

const road = new Road(ROAD_CENTER_X, ROAD_WIDTH);
const BRAIN_SHAPE = Car.brainShape(Sensor.RAY_COUNT);

// A saved brain is only meaningful for the sensors and controls it was bred
// with. The shape check alone let a brain trained on 150px rays reload onto
// 280px rays — same input count, every learned threshold miscalibrated. The
// format string names everything the genome's meaning depends on.
const BRAIN_FORMAT = `v3/${BRAIN_SHAPE.join("x")}/rays${Sensor.RAY_LENGTH}/light${LIGHT_LOOKAHEAD}/brake`;

let mode = "train";
let evolution = null;
let curriculum = new Curriculum("auto", "A");
let trackSeed = 0;
let gaSeed = 0;
let fitnessHistory = [];
let allTimeBest = 0;
let savedChampion = null;
let demo = null;
let benchmark = null;
let experimentLog = [];

let simClockMs = 0;
let simSpeed = 1;
let paused = false;
let breedingHoldFrames = 0;
let leaderNote = null;
let liveSummary = null;
let drawnFrames = 0;

// ---------------------------------------------------------------------------
// Storage. localStorage throws in private-browsing modes; the simulation
// should still run, just without remembering anything between sessions.
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

function readJson(key) {
    try {
        return JSON.parse(readStore(key));
    } catch {
        return null;
    }
}

function setStatus(text) {
    ui.status.textContent = text;
}

// ---------------------------------------------------------------------------
// The saved champion.
function loadChampionRecord() {
    const record = readJson(STORE_CHAMPION);
    if (record && record.format === BRAIN_FORMAT && record.brain) {
        const brain = NeuralNetwork.fromJSON(record.brain, BRAIN_SHAPE);
        if (brain) {
            return { record, brain, message: null };
        }
        return { record: null, brain: null, message: "The saved brain is corrupt and cannot be loaded. Train a population to create a new one." };
    }
    if (record && record.format && record.format !== BRAIN_FORMAT) {
        return { record: null, brain: null,
            message: `The saved brain was bred for an older network (${record.format}); this build needs ${BRAIN_FORMAT}. It cannot be loaded — train a new population.` };
    }
    if (readStore(LEGACY_BRAIN_KEY)) {
        return { record: null, brain: null,
            message: "An old 7-input saved brain was found. It is incompatible with the 10-input network and cannot be loaded — train a population to create a new champion." };
    }
    return { record: null, brain: null, message: "No saved brain yet. Train a population first; the champion is saved automatically." };
}

function championRecordFromCar(car, stage, generation) {
    return {
        format: BRAIN_FORMAT,
        brain: NeuralNetwork.toJSON(car.brain),
        fitness: car.fitness,
        stage,
        trackSeed,
        gaSeed,
        generation,
        savedAt: new Date().toISOString(),
        breakdown: car.fitnessBreakdown
    };
}

// A brain from a later stage always replaces one from an earlier stage (the
// later stage is the harder task). Within a stage only a higher fitness wins,
// so a lucky early champion is never overwritten by a worse late one.
function isBetterChampion(candidate, current) {
    if (!current) {
        return true;
    }
    const candidateRank = stageRank(candidate.stage);
    const currentRank = stageRank(current.stage);
    if (candidateRank !== currentRank) {
        return candidateRank > currentRank;
    }
    return candidate.fitness > current.fitness;
}

function saveChampionRecord(record, reason) {
    savedChampion = record;
    writeStore(STORE_CHAMPION, JSON.stringify(record));
    setStatus(`${reason}: fitness ${Math.round(record.fitness)}, stage ${record.stage}, generation ${record.generation}.`);
}

// ---------------------------------------------------------------------------
// Training state.
function historyPoint(summary) {
    const config = evolution.config;
    return {
        generation: summary.generation,
        stage: summary.stage,
        trackSeed: summary.trackSeed,
        frames: summary.frames,
        popSize: summary.popSize,
        best: Math.round(summary.best),
        mean: Math.round(summary.mean),
        median: Math.round(summary.median),
        meanDistance: Math.round(summary.meanDistance),
        meanOvertakes: Number(summary.meanOvertakes.toFixed(3)),
        meanLegalStops: Number(summary.meanLegalStops.toFixed(3)),
        collisions: summary.collisions,
        collisionRate: Number(summary.collisionRate.toFixed(3)),
        redLightViolations: summary.redLightViolations,
        legalStops: summary.legalStops,
        redEncounters: summary.redEncounters,
        legalStopRate: summary.legalStopRate === null ? null : Number(summary.legalStopRate.toFixed(3)),
        lightsCleared: summary.lightsCleared,
        carsMetLight: summary.carsMetLight,
        carsClearedLight: summary.carsClearedLight,
        lightClearRate: summary.lightClearRate === null ? null : Number(summary.lightClearRate.toFixed(3)),
        diversity: Number(summary.diversity.toFixed(4)),
        bestDistance: Math.round(summary.bestDistance),
        bestOvertakes: summary.bestOvertakes,
        bestLegalStops: summary.bestLegalStops,
        bestLightsCleared: summary.bestLightsCleared,
        bestCrashed: summary.bestCrashed,
        mutationRate: config.mutationRate,
        mutationStrength: config.mutationStrength,
        tournamentSize: config.tournamentSize,
        eliteCount: config.eliteCount
    };
}

function persistTraining() {
    writeStore(STORE_TRAINING, JSON.stringify({
        generation: evolution.generation,
        trackSeed,
        gaSeed,
        stage: curriculum.stage,
        curriculumMode: curriculum.mode,
        config: evolution.config,
        allTimeBest,
        history: fitnessHistory.slice(-HISTORY_LIMIT)
    }));
}

function loadTrainingState() {
    const state = readJson(STORE_TRAINING);
    if (!state || !Number.isFinite(state.trackSeed)) {
        return null;
    }
    return state;
}

function clearTraining() {
    clearStore(STORE_TRAINING);
    clearStore(STORE_CHAMPION);
    for (const key of LEGACY_KEYS) {
        clearStore(key);
    }
    savedChampion = null;
    fitnessHistory = [];
    allTimeBest = 0;
}

function newSeed() {
    return Math.floor(Math.random() * 0xFFFFFFFF);
}

// ---------------------------------------------------------------------------
// Config panel.
function readConfigFromUi() {
    const raw = {};
    for (const [key, input] of Object.entries(ui.config)) {
        if (input && input.value !== undefined && input.value !== "") {
            raw[key] = input.value;
        }
    }
    return raw;
}

function writeConfigToUi(config) {
    for (const [key, input] of Object.entries(ui.config)) {
        if (input) {
            input.value = config[key];
        }
    }
    if (ui.stageSelect) {
        ui.stageSelect.value = curriculum.mode;
    }
    ui.configSummary.textContent =
        `pop ${config.popSize} · elite ${config.eliteCount} · tournament ${config.tournamentSize}` +
        ` · mutation ${config.mutationRate} × ${config.mutationStrength}` +
        ` · stage ${curriculum.stage}${curriculum.mode === "auto" ? " (auto)" : " (pinned)"}` +
        ` · track seed ${trackSeed} · GA seed ${gaSeed}`;
}

// Takes effect from the next generation: the brains being bred then use the
// new rates, and the next population has the new size.
function applyConfig(raw = readConfigFromUi(), stageMode = ui.stageSelect ? ui.stageSelect.value : undefined) {
    const { config, warnings } = Evolution.validateConfig(raw, evolution.config);
    evolution.config = config;
    if (stageMode !== undefined) {
        curriculum.setMode(stageMode);
        evolution.setScenario(curriculum.scenario);
    }
    writeConfigToUi(config);
    persistTraining();
    setStatus(warnings.length
        ? "Applied with corrections: " + warnings.join("; ")
        : `Parameters applied; they take effect from generation ${evolution.generation + 1}.`);
    return { config, warnings };
}

// ---------------------------------------------------------------------------
// Training mode.
function createEvolution({ seedBrain = null, generation = 1, config = {} } = {}) {
    return new Evolution({
        road,
        scenario: curriculum.scenario,
        trackSeed,
        config,
        random: createSeededRandom(gaSeed),
        seedBrain,
        generation
    });
}

// `fromSaved`: restart training from the saved champion on its own track and
// stage. Otherwise a completely fresh start on a new track.
function restartTraining(fromSaved) {
    const config = evolution ? evolution.config : {};
    breedingHoldFrames = 0;
    leaderNote = null;
    liveSummary = null;

    if (fromSaved) {
        const { record, brain, message } = loadChampionRecord();
        if (!brain) {
            setStatus(message);
            return false;
        }
        savedChampion = record;
        trackSeed = record.trackSeed;
        gaSeed = newSeed();
        curriculum = new Curriculum(curriculum.mode, record.stage);
        fitnessHistory = [];
        evolution = createEvolution({ seedBrain: brain, generation: 1, config });
        setStatus(`Training restarted from the saved champion (stage ${record.stage}, fitness ${Math.round(record.fitness)}) on track ${trackSeed}.`);
    } else {
        clearTraining();
        trackSeed = newSeed();
        gaSeed = newSeed();
        curriculum = new Curriculum(curriculum.mode, curriculum.mode === "auto" ? "A" : curriculum.mode);
        evolution = createEvolution({ config });
        setStatus(`Fresh start: random population on new track ${trackSeed}, stage ${curriculum.stage}.`);
    }
    setMode("train");
    writeConfigToUi(evolution.config);
    persistTraining();
    return true;
}

function onGenerationEnd(summary) {
    fitnessHistory.push(historyPoint(summary));
    if (fitnessHistory.length > HISTORY_LIMIT) {
        fitnessHistory.shift();
    }

    // Autosave the best brain ever seen under the current stage rules, not
    // merely this generation's. A champion from a later stage replaces an
    // earlier one; within a stage only a better fitness does.
    const candidate = championRecordFromCar(
        { brain: summary.bestBrain, fitness: summary.best, fitnessBreakdown: summary.bestBreakdown },
        summary.stage, summary.generation
    );
    if (isBetterChampion(candidate, savedChampion)) {
        saveChampionRecord(candidate, "New champion saved");
    }

    const promoted = curriculum.observe(summary);
    if (promoted) {
        evolution.setScenario(curriculum.scenario);
        setStatus(`Curriculum: stage ${summary.stage} passed (${CURRICULUM[summary.stage].describe}, ${CURRICULUM[summary.stage].consecutive} generations running). Moving to stage ${promoted}: ${SCENARIOS[promoted].name}.`);
    }

    writeConfigToUi(evolution.config);
    persistTraining();
}

function trainStep() {
    const summary = evolution.step();
    if (summary) {
        onGenerationEnd(summary);
        liveSummary = null;
        if (BREEDING_HOLD_FRAMES > 0) {
            breedingHoldFrames = BREEDING_HOLD_FRAMES;
        } else {
            evolution.advance();
        }
    }
}

// ---------------------------------------------------------------------------
// Demo mode: one frozen brain, one fixed route, nothing evolves.
function buildDemo(brain, record) {
    const world = new World(road, buildTrack(0, DEMO_SCENARIO));
    const car = world.spawnCar(NeuralNetwork.clone(brain));
    return {
        world,
        car,
        record,
        // The genome as it went in; compared against the car's brain at the
        // end to prove nothing touched it.
        brainJson: JSON.stringify(NeuralNetwork.toJSON(brain)),
        frames: 0,
        result: null,
        events: []
    };
}

function startDemo() {
    const { record, brain, message } = loadChampionRecord();
    if (!brain) {
        setStatus(message);
        return false;
    }
    savedChampion = record;
    demo = buildDemo(brain, record);
    setMode("demo");
    setPaused(false);
    setStatus(`Demo: saved champion from stage ${record.stage}, generation ${record.generation}, fitness ${Math.round(record.fitness)}. Evolution is off; the brain is frozen.`);
    return true;
}

function resetDemo() {
    if (!demo) {
        return startDemo();
    }
    demo = buildDemo(NeuralNetwork.fromJSON(JSON.parse(demo.brainJson), BRAIN_SHAPE), demo.record);
    setMode("demo");
    setPaused(false);
    setStatus("Demo reset: same route, same brain, from the start line.");
    return true;
}

function finishDemo() {
    const car = demo.car;
    let outcome = "timeout";
    if (car.damaged) {
        outcome = "crashed";
    } else if (car.finished) {
        outcome = "completed";
    } else if (car.retired) {
        outcome = "stalled";
    }
    return {
        success: car.finished,
        outcome,
        crashCause: car.crashCause,
        distance: car.maxDistance,
        carsPassed: car.carsPassed,
        legalStops: car.legalStops,
        lightsCleared: car.lightsCleared,
        redEncounters: car.redEncounters,
        frames: demo.frames,
        brainUnchanged: JSON.stringify(NeuralNetwork.toJSON(car.brain)) === demo.brainJson
    };
}

function demoStep() {
    if (demo.result) {
        return;
    }
    const car = demo.car;
    const before = { legalStops: car.legalStops, lightsCleared: car.lightsCleared, carsPassed: car.carsPassed, state: car.lightState };
    demo.world.step([car]);
    demo.frames++;

    const seconds = (demo.frames * SIM_FRAME_MS / 1000).toFixed(1);
    if (car.legalStops > before.legalStops) {
        demo.events.push(`${seconds}s  stopped at red light (legal stop ${car.legalStops})`);
    }
    if (car.lightsCleared > before.lightsCleared) {
        demo.events.push(`${seconds}s  cleared light ${car.lightsCleared} on ${car.lightState || "green"}`);
    }
    if (car.carsPassed > before.carsPassed) {
        demo.events.push(`${seconds}s  overtook car ${car.carsPassed}`);
    }
    if (car.lightState === "green" && before.state === "red" && car.speed < STOP_SPEED) {
        demo.events.push(`${seconds}s  light turned green`);
    }

    if (!car.active || demo.frames >= DEMO_MAX_FRAMES) {
        demo.result = finishDemo();
        demo.events.push(`${seconds}s  ${demo.result.outcome.toUpperCase()}` +
            (demo.result.crashCause ? ` (${demo.result.crashCause})` : ""));
    }
}

// ---------------------------------------------------------------------------
// Benchmark mode.
function loadResults() {
    const rows = readJson(STORE_RESULTS);
    return Array.isArray(rows) ? rows : [];
}

function persistResults() {
    if (experimentLog.length > RESULTS_LIMIT) {
        experimentLog = experimentLog.slice(-RESULTS_LIMIT);
    }
    writeStore(STORE_RESULTS, JSON.stringify(experimentLog));
    ui.resultsCount.textContent = `${experimentLog.length} stored result rows`;
}

const fmt = {
    n: (value, digits = 0) => (Number.isFinite(value) ? value.toFixed(digits) : "n/a"),
    pct: value => (value === null || value === undefined || !Number.isFinite(value) ? "n/a" : `${Math.round(value * 100)}%`)
};

function describeEpisodeSummary(label, s) {
    if (!s || s.trials === 0) {
        return [`${label}: no trials`];
    }
    return [
        `${label} (${s.trials} trial${s.trials === 1 ? "" : "s"})`,
        `   mean fitness ${fmt.n(s.meanFitness)}   mean distance ${fmt.n(s.meanDistance)}px   mean overtakes ${fmt.n(s.meanOvertakes, 2)}`,
        `   completed ${s.completed}/${s.trials}   collisions ${s.collisions}/${s.trials} (${fmt.pct(s.collisionRate)})   red-light violations ${s.redLightViolations}`,
        `   legal stops ${s.legalStops} of ${s.redEncounters} red encounters (${fmt.pct(s.legalStopRate)})   lights cleared ${s.lightsCleared}` +
            (s.brainUnchanged ? "" : "   BRAIN CHANGED — bug")
    ];
}

function describeResult(result) {
    const lines = [];
    if (result.key === "A") {
        lines.push(`# Experiment A — evolved vs untrained (track ${trackSeed}, stage ${curriculum.stage} rules, route ${curriculum.scenario.routeLength}px)`);
        lines.push(...describeEpisodeSummary("Untrained random networks", result.summary.untrained));
        lines.push(...describeEpisodeSummary("Evolved networks (saved champion + population elites)", result.summary.evolved));
        lines.push(...describeEpisodeSummary("Hand-coded baseline (rules, NOT the GA)", result.summary.heuristic));
    } else if (result.key === "B") {
        const s = result.settings;
        lines.push(`# Experiment B — mutation rate (pop ${s.popSize}, ${s.generations} generations × ${s.maxGenerationFrames} frames, ${s.trials} trials each, track ${s.trackSeed}, stage ${curriculum.stage})`);
        for (const rate of s.rates) {
            const r = result.summary[rate];
            lines.push(`rate ${rate}: final best ${fmt.n(r.meanFinalBest)} (trials: ${r.finalBests.map(v => Math.round(v)).join(", ")})` +
                `   final mean ${fmt.n(r.meanFinalMean)}   best distance ${fmt.n(r.meanBestDistance)}px   best overtakes ${fmt.n(r.meanBestOvertakes, 1)}`);
        }
        lines.push(`${s.trials} trials per rate is a small sample; differences smaller than the spread between trials are not evidence.`);
    } else if (result.key === "C") {
        const t = result.summary.trainingResult;
        lines.push(`# Experiment C — generalization of the saved champion (frozen)`);
        lines.push(`Training track ${t.trackSeed}: ${t.outcome}${t.crashCause ? ` (${t.crashCause})` : ""}, ${fmt.n(t.distance)}px, ${t.carsPassed} overtakes, ${t.legalStops} legal stops, ${t.lightsCleared} lights`);
        lines.push(...describeEpisodeSummary(`Unseen tracks ${result.summary.unseenSeeds.join(", ")}`, result.summary.unseen));
        for (const r of result.summary.unseenResults) {
            lines.push(`   seed ${r.trackSeed}: ${r.outcome}${r.crashCause ? ` (${r.crashCause})` : ""}, ${fmt.n(r.distance)}px, ${r.carsPassed} overtakes, ${r.legalStops} stops`);
        }
        if (result.summary.demoResult) {
            const d = result.summary.demoResult;
            lines.push(`Demo route: ${d.outcome}${d.crashCause ? ` (${d.crashCause})` : ""}, ${fmt.n(d.distance)}/${d.routeLength}px, ${d.carsPassed} overtakes, ${d.legalStops} legal stops, ${d.lightsCleared} lights`);
        }
        lines.push(`brain unchanged by evaluation: ${result.summary.unseen.brainUnchanged && result.summary.training.brainUnchanged ? "yes" : "NO — bug"}`);
    }
    return lines;
}

function experimentGenerator(kind) {
    const { brain } = loadChampionRecord();
    const ranked = [...evolution.cars].sort((a, b) => b.fitness - a.fitness);
    const scenario = curriculum.scenario;
    if (kind === "A") {
        return experimentBaseline({ road, scenario, trackSeed, champion: brain,
            population: ranked.slice(0, 5).map(car => car.brain), seed: gaSeed + 7 });
    }
    if (kind === "B") {
        return experimentMutation({ road, scenario, trackSeed, baseSeed: gaSeed + 100 });
    }
    if (kind === "C") {
        if (!brain) {
            return null;
        }
        return experimentGeneralization({ road, champion: brain, trainingSeed: trackSeed, scenario });
    }
    return null;
}

// Chains the chosen experiments one after another; each one's rows are kept
// and its summary printed as it finishes.
function runEvaluation(kind = ui.experimentSelect && ui.experimentSelect.value ? ui.experimentSelect.value : "A") {
    const kinds = kind === "all" ? ["A", "B", "C"] : [kind];
    if (kinds.includes("C") && !loadChampionRecord().brain) {
        setStatus("Experiment C needs a saved champion. Train a population first.");
        if (kinds.length === 1) {
            return null;
        }
    }

    benchmark = { kinds, index: 0, lines: [], runner: null, results: [], title: "" };
    setMode("benchmark");

    const startNext = () => {
        while (benchmark.index < benchmark.kinds.length) {
            const next = benchmark.kinds[benchmark.index++];
            const generator = experimentGenerator(next);
            if (!generator) {
                benchmark.lines.push(`# Experiment ${next} skipped: no saved champion`);
                continue;
            }
            benchmark.title = `Experiment ${next} running`;
            benchmark.runner = new ExperimentRunner(generator, {
                onDone: result => {
                    benchmark.results.push(result);
                    experimentLog.push(...result.rows);
                    persistResults();
                    benchmark.lines.push(...describeResult(result), "");
                    ui.results.textContent = benchmark.lines.join("\n");
                    startNext();
                },
                onError: error => {
                    benchmark.lines.push(`# Experiment ${next} failed: ${error.message}`);
                    ui.results.textContent = benchmark.lines.join("\n");
                    startNext();
                }
            }).start();
            return;
        }
        benchmark.runner = null;
        benchmark.title = "Benchmarks finished";
        setStatus(`Benchmarks finished. ${experimentLog.length} rows stored; Export Results writes them as CSV.`);
    };
    startNext();
    return benchmark;
}

function trainingRows() {
    return fitnessHistory.map(point => ({
        experiment: "training",
        experiment_id: `training-${trackSeed}-${gaSeed}`,
        source: "training",
        timestamp: "",
        track_type: "training",
        track_seed: point.trackSeed,
        scenario: point.stage,
        stage: point.stage,
        group: "",
        trial: "",
        generation: point.generation,
        population_size: point.popSize,
        elite_count: point.eliteCount,
        tournament_size: point.tournamentSize,
        mutation_rate: point.mutationRate,
        mutation_strength: point.mutationStrength,
        ga_seed: gaSeed,
        best_fitness: point.best,
        mean_fitness: point.mean,
        median_fitness: point.median,
        mean_distance_px: point.meanDistance,
        mean_overtakes: point.meanOvertakes,
        collisions: point.collisions,
        collision_rate: point.collisionRate,
        legal_stops: point.legalStops,
        red_encounters: point.redEncounters,
        legal_stop_rate: point.legalStopRate === null ? "" : point.legalStopRate,
        lights_cleared: point.lightsCleared,
        red_light_violations: point.redLightViolations,
        diversity: point.diversity,
        frames: point.frames,
        sim_seconds: Number((point.frames * SIM_FRAME_MS / 1000).toFixed(2)),
        outcome: point.bestCrashed ? "champion crashed" : "champion survived",
        notes: `best distance ${point.bestDistance}, best overtakes ${point.bestOvertakes}, best legal stops ${point.bestLegalStops}`
    }));
}

function buildResultsCsv() {
    return toCsv(CSV_COLUMNS, [...trainingRows(), ...experimentLog]);
}

function exportResults() {
    const csv = buildResultsCsv();
    const filename = `ga-results-track${trackSeed}-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    if (typeof Blob === "undefined" || typeof URL === "undefined" || !URL.createObjectURL) {
        return { csv, filename, downloaded: false };
    }
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus(`Exported ${fitnessHistory.length} training rows and ${experimentLog.length} benchmark rows to ${filename}.`);
    return { csv, filename, downloaded: true };
}

// ---------------------------------------------------------------------------
// Interface.
function setMode(value) {
    mode = value;
    ui.modeBanner.textContent = value === "train" ? "TRAINING — population evolving"
        : value === "demo" ? "DEMO — frozen champion, no evolution"
            : "BENCHMARK — evaluating, training paused";
    ui.modeBanner.className = `mode-${value}`;
    ui.trainButton.classList.toggle("activeMode", value === "train");
    ui.demoButton.classList.toggle("activeMode", value === "demo");
}

function setPaused(value) {
    paused = value;
    ui.pauseButton.textContent = paused ? "Resume" : "Pause";
}

function setSpeed(value) {
    simSpeed = value;
    for (const button of document.querySelectorAll(".speedButton")) {
        button.classList.toggle("activeSpeed", Number(button.dataset.speed) === value);
    }
}

function resizeCanvases() {
    carCanvas.width = ROAD_VIEW_WIDTH;
    carCanvas.height = window.innerHeight;
    networkCanvas.width = NETWORK_VIEW_WIDTH;
    networkCanvas.height = Math.min(NETWORK_VIEW_HEIGHT, window.innerHeight);
    chartCanvas.width = CHART_WIDTH;
    chartCanvas.height = CHART_HEIGHT;
}

// Writes a list of {text, cls} lines into a container. Falls back to plain
// text where the DOM is a stub (the Node test harness).
function renderLines(element, lines) {
    if (typeof element.replaceChildren !== "function") {
        element.textContent = lines.map(line => line.text).join("\n");
        return;
    }
    const nodes = lines.map(line => {
        const div = document.createElement("div");
        div.textContent = line.text;
        if (line.cls) {
            div.className = line.cls;
        }
        return div;
    });
    element.replaceChildren(...nodes);
}

function renderMetrics(rows) {
    if (typeof ui.metricsBody.replaceChildren !== "function") {
        ui.metricsBody.textContent = rows.map(row => row.join(" | ")).join("\n");
        return;
    }
    ui.metricsBody.replaceChildren(...rows.map(cells => {
        const tr = document.createElement("tr");
        for (const cell of cells) {
            const td = document.createElement("td");
            td.textContent = cell;
            tr.appendChild(td);
        }
        return tr;
    }));
}

function allTimeMetrics() {
    const best = key => fitnessHistory.reduce((top, p) => (Number.isFinite(p[key]) && p[key] > top ? p[key] : top), -Infinity);
    const least = key => fitnessHistory.reduce((low, p) => (Number.isFinite(p[key]) && p[key] < low ? p[key] : low), Infinity);
    const clean = value => (Number.isFinite(value) ? value : null);
    return {
        best: clean(best("best")),
        mean: clean(best("mean")),
        median: clean(best("median")),
        meanDistance: clean(best("meanDistance")),
        meanOvertakes: clean(best("meanOvertakes")),
        collisionRate: clean(least("collisionRate")),
        legalStopRate: clean(best("legalStopRate")),
        lightClearRate: clean(best("lightClearRate")),
        diversity: clean(least("diversity"))
    };
}

function updateTrainingHud() {
    if (drawnFrames % 15 === 0 || !liveSummary) {
        liveSummary = evolution.summarize();
    }
    const s = liveSummary;
    const car = evolution.bestCar;
    const champion = evolution.champion;
    const elapsed = evolution.frameInGeneration * SIM_FRAME_MS / 1000;
    const b = champion.fitnessBreakdown;

    ui.header.textContent = `Generation ${evolution.generation} · stage ${curriculum.stage}`;
    renderLines(ui.infoLines, [
        { text: SCENARIOS[curriculum.stage].name, cls: "muted" },
        { text: `Generation time: ${elapsed.toFixed(1)} s`, cls: "" },
        { text: `Active cars: ${evolution.activeCount} / ${evolution.cars.length} (alive ${evolution.aliveCount})`, cls: "" },
        { text: `Camera car: ${car.speed.toFixed(2)} px/f · ${car.actionLabel}`, cls: "" },
        { text: `Best distance: ${champion.maxDistance.toFixed(0)} px · passed ${champion.carsPassed} · stops ${champion.legalStops} · lights ${champion.lightsCleared}`, cls: "" },
        { text: `Best fitness: ${champion.fitness.toFixed(0)}`, cls: "strong" },
        { text: `= ${b.distance} dist + ${b.carsPassed}×${OVERTAKE_BONUS} passed + ${b.legalStops}×${LEGAL_STOP_BONUS} stops + ${b.lightsCleared}×${LIGHT_BONUS} lights` +
            (b.crashed ? ` − ${CRASH_PENALTY} crash` : "") + (b.redLightViolation ? ` − ${RED_LIGHT_PENALTY} red light` : ""), cls: "muted" },
        { text: curriculum.mode === "auto" && CURRICULUM[curriculum.stage].next
            ? `Promotion to ${CURRICULUM[curriculum.stage].next} when ${CURRICULUM[curriculum.stage].describe} (${curriculum.streak}/${CURRICULUM[curriculum.stage].consecutive})`
            : (curriculum.mode === "auto" ? "Final stage" : `Stage pinned to ${curriculum.stage}`), cls: "muted" }
    ]);

    const all = allTimeMetrics();
    const rate = (value) => (value === null ? "n/a" : `${Math.round(value * 100)}%`);
    const num = (value, digits = 0) => (value === null || value === undefined ? "–" : Number(value).toFixed(digits));
    document.querySelector("#metricsTitle").textContent =
        `Population metrics — generation ${evolution.generation} (live) vs all-time`;
    renderMetrics([
        ["best fitness", num(s.best), num(all.best)],
        ["mean fitness", num(s.mean), num(all.mean)],
        ["median fitness", num(s.median), num(all.median)],
        ["mean distance (px)", num(s.meanDistance), num(all.meanDistance)],
        ["mean overtakes", num(s.meanOvertakes, 2), num(all.meanOvertakes, 2)],
        ["collision rate", rate(s.collisionRate), all.collisionRate === null ? "–" : `${rate(all.collisionRate)} (lowest)`],
        ["legal-stop rate", s.legalStopRate === null ? `n/a (${s.redEncounters} red)` : rate(s.legalStopRate), rate(all.legalStopRate)],
        ["cars clearing a light", s.lightClearRate === null ? `n/a (${s.carsMetLight} met)` : `${s.carsClearedLight}/${s.carsMetLight}`, rate(all.lightClearRate)],
        ["genome diversity", num(s.diversity, 3), all.diversity === null ? "–" : `${num(all.diversity, 3)} (lowest)`]
    ]);
    ui.bestScore.textContent = String(Math.round(allTimeBest));
}

function updateDemoHud() {
    const car = demo.car;
    const seconds = demo.frames * SIM_FRAME_MS / 1000;
    const readings = car.sensor.readings.map(r => (r ? `${Math.round((1 - r.offset) * 100)}%` : "0%"));
    const light = car.lightState
        ? `${car.lightState.toUpperCase()} in ${car.lightDistance.toFixed(0)}px`
        : "none within 260px";
    const progress = Math.min(1, car.maxDistance / demo.world.routeLength);
    const lines = [
        { text: `Champion: stage ${demo.record.stage}, gen ${demo.record.generation}, fitness ${Math.round(demo.record.fitness)}`, cls: "muted" },
        { text: `Route time: ${seconds.toFixed(1)} s · progress ${Math.round(progress * 100)}% (${car.maxDistance.toFixed(0)}/${demo.world.routeLength}px)`, cls: "" },
        { text: `Speed: ${car.speed.toFixed(2)} px/frame`, cls: "" },
        { text: `Sensors L45 L22 FWD R22 R45: ${readings.join(" ")}`, cls: "muted" },
        { text: `Traffic light: ${light}`, cls: car.lightState === "red" ? "bad" : "" },
        { text: `Action: ${car.actionLabel}`, cls: "strong" },
        { text: `Overtakes ${car.carsPassed} · legal stops ${car.legalStops} · lights cleared ${car.lightsCleared}`, cls: "" },
        { text: `Collision: ${car.damaged ? `YES (${car.crashCause})` : "none"}`, cls: car.damaged ? "bad" : "good" }
    ];
    if (demo.result) {
        const r = demo.result;
        lines.push({
            text: r.success
                ? `RESULT: SUCCESS — route completed in ${(r.frames * SIM_FRAME_MS / 1000).toFixed(1)} s`
                : `RESULT: FAILURE — ${r.outcome}${r.crashCause ? ` (${r.crashCause})` : ""} at ${r.distance.toFixed(0)} px`,
            cls: r.success ? "good" : "bad"
        });
        lines.push({ text: `brain unchanged during playback: ${r.brainUnchanged ? "yes" : "NO — bug"}`, cls: "muted" });
    }
    for (const event of demo.events.slice(-5)) {
        lines.push({ text: event, cls: "muted" });
    }
    ui.header.textContent = "Demo — frozen champion";
    renderLines(ui.infoLines, lines);
    document.querySelector("#metricsTitle").textContent = "Demo route events";
    renderMetrics(demo.events.slice(-9).map(event => [event, "", ""]));
    ui.bestScore.textContent = String(Math.round(allTimeBest));
}

function updateBenchmarkHud() {
    ui.header.textContent = "Benchmark";
    const progress = benchmark.runner ? benchmark.runner.progress : { progress: 1, label: "finished" };
    renderLines(ui.infoLines, [
        { text: benchmark.title, cls: "strong" },
        { text: `${Math.round(progress.progress * 100)}% — ${progress.label}`, cls: "" },
        { text: "Training is paused while the benchmark runs. The population and the saved champion are not changed by it.", cls: "muted" },
        { text: "Results appear under the network panel and in Export Results.", cls: "muted" }
    ]);
}

// A faint cross wherever a car died this generation. Later generations
// visibly die further up the road.
function drawCrashMarks(marks, viewTop, viewBottom) {
    carCtx.strokeStyle = "rgba(0,0,0,0.35)";
    carCtx.lineWidth = 2;
    for (const mark of marks) {
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
function drawGaps(world, viewTop, viewBottom) {
    const laneWidth = road.width / road.laneCount;
    carCtx.fillStyle = "rgba(120,240,150,0.13)";
    for (const row of world.trackRows) {
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

function drawFinishLine(world, viewTop, viewBottom) {
    if (!world.routeLength) {
        return;
    }
    const y = START_Y - world.routeLength;
    if (y < viewTop - 20 || y > viewBottom + 20) {
        return;
    }
    carCtx.fillStyle = "rgba(0,0,0,0.75)";
    for (let i = 0; i < 9; i++) {
        if (i % 2 === 0) {
            carCtx.fillRect(road.left + i * 20, y - 4, 20, 8);
        }
    }
    carCtx.fillStyle = "white";
    carCtx.font = "bold 10px ui-sans-serif, system-ui, sans-serif";
    carCtx.textAlign = "center";
    carCtx.textBaseline = "bottom";
    carCtx.fillText("FINISH", road.x, y - 6);
}

function drawRoadView(world, cars, focus, marks) {
    carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height);

    const viewTop = focus.y - carCanvas.height * 0.7;
    const viewBottom = viewTop + carCanvas.height;

    carCtx.save();
    carCtx.translate(0, -focus.y + carCanvas.height * 0.7);

    road.draw(carCtx, viewTop, viewBottom);
    drawGaps(world, viewTop, viewBottom);
    drawCrashMarks(marks, viewTop, viewBottom);
    drawFinishLine(world, viewTop, viewBottom);

    for (const light of world.lights) {
        if (light.y > viewTop - 200 && light.y < viewBottom + 200) {
            light.draw(carCtx);
        }
    }
    for (const car of world.traffic) {
        if (car.y > viewTop - 60 && car.y < viewBottom + 60) {
            car.draw(carCtx);
        }
    }

    carCtx.globalAlpha = 0.2;
    for (const car of cars) {
        if (car !== focus) {
            car.draw(carCtx);
        }
    }
    carCtx.globalAlpha = 1;
    focus.draw(carCtx, true);

    carCtx.restore();
}

function drawBanner(text, color = "rgba(0,0,0,0.72)") {
    carCtx.fillStyle = color;
    carCtx.fillRect(0, 8, carCanvas.width, 24);
    carCtx.fillStyle = "white";
    carCtx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
    carCtx.textAlign = "center";
    carCtx.textBaseline = "middle";
    carCtx.fillText(text, carCanvas.width / 2, 20);
}

function draw() {
    drawnFrames++;
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

    if (mode === "demo") {
        drawRoadView(demo.world, [demo.car], demo.car, []);
        if (demo.result) {
            drawBanner(demo.result.success ? "ROUTE COMPLETED" : `FAILED: ${demo.result.outcome}`,
                demo.result.success ? "rgba(31,122,58,0.85)" : "rgba(179,38,30,0.85)");
        } else {
            drawBanner("DEMO — frozen champion brain", "rgba(31,122,58,0.72)");
        }
        networkCtx.lineDashOffset = -simClockMs / 50;
        Visualizer.drawNetwork(networkCtx, demo.car.brain, demo.car.decision, "Champion's brain (frozen)");
        Panels.drawFitnessHistory(chartCtx, fitnessHistory);
        updateDemoHud();
        return;
    }

    if (mode === "benchmark") {
        drawRoadView(evolution.world, evolution.cars, evolution.bestCar, evolution.crashMarks);
        drawBanner("BENCHMARK — training paused", "rgba(154,74,0,0.8)");
        const progress = benchmark.runner ? benchmark.runner.progress : { progress: 1, label: "finished" };
        Panels.drawProgress(networkCtx, benchmark.title, progress.progress, progress.label,
            benchmark.lines.slice(-30));
        Panels.drawFitnessHistory(chartCtx, fitnessHistory);
        updateBenchmarkHud();
        return;
    }

    drawRoadView(evolution.world, evolution.cars, evolution.bestCar, evolution.crashMarks);

    if (leaderNote) {
        drawBanner(leaderNote.text);
        // Counted in drawn frames, not simulation steps, so it stays readable
        // at x20 as well.
        leaderNote.frames--;
        if (leaderNote.frames <= 0) {
            leaderNote = null;
        }
    }

    if (breedingHoldFrames > 0 && evolution.breedingSnapshot) {
        Panels.drawBreeding(networkCtx, evolution.breedingSnapshot);
    } else {
        networkCtx.lineDashOffset = -simClockMs / 50;
        Visualizer.drawNetwork(networkCtx, evolution.bestCar.brain, evolution.bestCar.decision);
    }

    Panels.drawFitnessHistory(chartCtx, fitnessHistory);
    updateTrainingHud();
}

function animate() {
    if (!paused) {
        if (mode === "train") {
            // Pause freezes the breeding countdown as well as the simulation,
            // so the breeding panel can be held on screen for as long as it
            // takes to talk through it. Drawing continues either way.
            if (breedingHoldFrames > 0) {
                breedingHoldFrames--;
                if (breedingHoldFrames === 0) {
                    evolution.advance();
                }
            } else {
                const previousLeader = evolution.bestCar;
                for (let i = 0; i < simSpeed; i++) {
                    simClockMs += SIM_FRAME_MS;
                    trainStep();
                    // A generation just ended and the breeding panel is up:
                    // stop fast-forwarding so it stays readable.
                    if (breedingHoldFrames > 0) {
                        break;
                    }
                }
                // The camera jumping to another car with no explanation looked
                // like the car had teleported; say what happened.
                if (previousLeader && !previousLeader.active && evolution.bestCar !== previousLeader &&
                    evolution.activeCount > 0 && breedingHoldFrames === 0) {
                    leaderNote = { text: "leader stopped — following next best car", frames: 110 };
                }
                if (evolution.champion.maxDistance > allTimeBest) {
                    allTimeBest = evolution.champion.maxDistance;
                }
            }
        } else if (mode === "demo") {
            for (let i = 0; i < simSpeed; i++) {
                simClockMs += SIM_FRAME_MS;
                demoStep();
            }
        }
    }

    draw();
    requestAnimationFrame(animate);
}

// ---------------------------------------------------------------------------
// Buttons.
ui.trainButton.addEventListener("click", () => {
    if (mode === "benchmark" && benchmark && benchmark.runner) {
        benchmark.runner.cancel();
        benchmark.runner = null;
        setStatus("Benchmark cancelled.");
    }
    setMode("train");
    setPaused(false);
    setStatus(`Training: generation ${evolution.generation}, stage ${curriculum.stage}.`);
});
ui.demoButton.addEventListener("click", () => startDemo());
document.querySelector("#resetDemoButton").addEventListener("click", () => resetDemo());

document.querySelector("#saveButton").addEventListener("click", () => {
    const car = mode === "demo" ? null : evolution.champion;
    if (!car) {
        setStatus("Save works in training mode: it stores the current generation's best car.");
        return;
    }
    saveChampionRecord(championRecordFromCar(car, curriculum.stage, evolution.generation), "Saved by hand");
});

document.querySelector("#loadButton").addEventListener("click", () => restartTraining(true));
document.querySelector("#discardButton").addEventListener("click", () => restartTraining(false));
document.querySelector("#applyConfigButton").addEventListener("click", () => applyConfig());
document.querySelector("#evaluateButton").addEventListener("click", () => runEvaluation());
document.querySelector("#exportButton").addEventListener("click", () => exportResults());
document.querySelector("#clearResultsButton").addEventListener("click", () => {
    const count = experimentLog.length;
    if (count === 0) {
        setStatus("No stored benchmark results.");
        return;
    }
    if (typeof confirm === "function" && !confirm(`Delete all ${count} stored benchmark rows? Export them first if you need them.`)) {
        return;
    }
    experimentLog = [];
    persistResults();
    ui.results.textContent = "";
    setStatus(`Deleted ${count} stored benchmark rows.`);
});

ui.pauseButton.addEventListener("click", () => setPaused(!paused));

for (const button of document.querySelectorAll(".speedButton")) {
    button.addEventListener("click", () => setSpeed(Number(button.dataset.speed)));
}

window.addEventListener("resize", resizeCanvases);

// ---------------------------------------------------------------------------
// Start-up: resume the saved training run if there is one.
(function boot() {
    experimentLog = loadResults();
    ui.resultsCount.textContent = `${experimentLog.length} stored result rows`;

    const state = loadTrainingState();
    const champion = loadChampionRecord();
    savedChampion = champion.record;

    if (state) {
        trackSeed = state.trackSeed;
        gaSeed = Number.isFinite(state.gaSeed) ? state.gaSeed : newSeed();
        curriculum = new Curriculum(state.curriculumMode || "auto", state.stage || "A");
        fitnessHistory = Array.isArray(state.history) ? state.history : [];
        allTimeBest = Number(state.allTimeBest) || 0;
        evolution = createEvolution({
            seedBrain: champion.brain,
            generation: Number(state.generation) || 1,
            config: state.config || {}
        });
        setStatus(champion.brain
            ? `Resumed: generation ${evolution.generation}, stage ${curriculum.stage}, population seeded from the saved champion (fitness ${Math.round(champion.record.fitness)}).`
            : `Resumed training state at generation ${evolution.generation} with a fresh random population. ${champion.message}`);
    } else {
        trackSeed = newSeed();
        gaSeed = newSeed();
        evolution = createEvolution({ seedBrain: champion.brain });
        setStatus(champion.brain
            ? `Population seeded from the saved champion (stage ${champion.record.stage}).`
            : champion.message);
    }

    resizeCanvases();
    setMode("train");
    setPaused(false);
    setSpeed(SIM_SPEEDS[0]);
    writeConfigToUi(evolution.config);
    persistTraining();
    requestAnimationFrame(animate);
})();
