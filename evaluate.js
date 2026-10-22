"use strict";

// Evaluation and benchmarking. Everything here runs a brain through a World
// without evolving it: the brain is cloned, driven, and checked afterwards to
// be byte-identical to what went in. Experiments are generators so the
// interface can run them in short slices and stay responsive.

const EVALUATION_MAX_FRAMES = 90 * 60;

// Seeds the generalization experiment drives that were never used for
// training. They are fixed so the results are reproducible; they are only
// "unseen" if the training track seed is not one of them, which the
// experiment checks.
const UNSEEN_TRACK_SEEDS = [101, 202, 303, 404, 505];

// Drive one controller once through a scenario. `brain` is cloned first, so
// the caller's brain cannot be changed by anything that happens here.
function runEpisode({ road, scenario, trackSeed, brain = null, driver = null,
    maxFrames = EVALUATION_MAX_FRAMES, stallFrames = STALL_FRAMES, routeLength }) {
    const track = buildTrack(trackSeed, scenario);
    const world = new World(road, track, {
        stallFrames,
        routeLength: routeLength === undefined ? track.routeLength : routeLength
    });

    const frozen = brain ? NeuralNetwork.clone(brain) : null;
    const before = brain ? JSON.stringify(NeuralNetwork.toJSON(brain)) : null;
    const car = world.spawnCar(frozen || new NeuralNetwork(Car.brainShape(Sensor.RAY_COUNT)));
    car.driver = driver;

    let frames = 0;
    while (car.active && frames < maxFrames) {
        world.step([car]);
        frames++;
    }

    let outcome = "timeout";
    if (car.damaged) {
        outcome = "crashed";
    } else if (car.finished) {
        outcome = "completed";
    } else if (car.retired) {
        outcome = "stalled";
    }

    return {
        outcome,
        completed: car.finished,
        crashed: car.damaged,
        crashCause: car.crashCause,
        distance: car.maxDistance,
        carsPassed: car.carsPassed,
        legalStops: car.legalStops,
        lightsCleared: car.lightsCleared,
        lightsMet: car.lightsMet,
        redEncounters: car.redEncounters,
        redLightViolation: car.redLightViolation,
        fitness: car.fitness,
        frames,
        elapsedMs: world.elapsedMs,
        routeLength: world.routeLength,
        trackSeed,
        scenario: scenario.key,
        // True when the input brain is unchanged after the run. It always
        // should be; the benchmark reports it rather than assumes it.
        brainUnchanged: brain
            ? JSON.stringify(NeuralNetwork.toJSON(brain)) === before && NeuralNetwork.equals(brain, car.brain)
            : true
    };
}

// A hand-written controller, for comparison only. It is NOT the product of
// the genetic algorithm and is never used to drive a demo. Its rules: brake
// for a red or yellow light once it is close; otherwise steer towards the
// side whose rays see further when something is ahead; accelerate.
function heuristicDriver(inputs) {
    const [l45, l22, fwd, r22, r45, , red, yellow, , proximity] = inputs;
    const stopping = (red > 0 || yellow > 0) && proximity > 0.4;
    const obstacleAhead = fwd > 0.3 || l22 > 0.6 || r22 > 0.6;
    const leftRoom = (1 - l45) + (1 - l22);
    const rightRoom = (1 - r45) + (1 - r22);
    const steerLeft = obstacleAhead && leftRoom > rightRoom + 0.05;
    const steerRight = obstacleAhead && rightRoom > leftRoom + 0.05;
    const brake = stopping || fwd > 0.75;
    return [brake ? -1 : 1, steerLeft ? 1 : -1, steerRight ? 1 : -1, brake ? 1 : -1];
}

// Averages over a list of episode results; counts and rates are explicit so
// a reader can see how many trials stand behind each number.
function summarizeEpisodes(results) {
    const n = results.length;
    if (n === 0) {
        return { trials: 0 };
    }
    const sum = key => results.reduce((total, r) => total + (r[key] || 0), 0);
    const redEncounters = sum("redEncounters");
    return {
        trials: n,
        meanFitness: sum("fitness") / n,
        meanDistance: sum("distance") / n,
        meanOvertakes: sum("carsPassed") / n,
        completed: results.filter(r => r.completed).length,
        completionRate: results.filter(r => r.completed).length / n,
        collisions: results.filter(r => r.crashed).length,
        collisionRate: results.filter(r => r.crashed).length / n,
        redLightViolations: results.filter(r => r.redLightViolation).length,
        legalStops: sum("legalStops"),
        redEncounters,
        legalStopRate: redEncounters > 0 ? sum("legalStops") / redEncounters : null,
        lightsCleared: sum("lightsCleared"),
        brainUnchanged: results.every(r => r.brainUnchanged)
    };
}

function newExperimentId(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

// One CSV row per episode or per generation. Every experiment fills the same
// columns so the rows of different experiments sit side by side in one file.
const CSV_COLUMNS = [
    "experiment", "experiment_id", "source", "timestamp",
    "track_type", "track_seed", "scenario", "stage", "group", "trial", "generation",
    "population_size", "elite_count", "tournament_size", "mutation_rate", "mutation_strength", "ga_seed",
    "best_fitness", "mean_fitness", "median_fitness", "mean_distance_px", "mean_overtakes",
    "collisions", "collision_rate", "legal_stops", "red_encounters", "legal_stop_rate",
    "lights_cleared", "red_light_violations", "diversity",
    "fitness", "distance_px", "overtakes", "frames", "sim_seconds", "outcome", "brain_unchanged", "notes"
];

function episodeRow(base, result) {
    return {
        ...base,
        track_seed: result.trackSeed,
        scenario: result.scenario,
        fitness: Math.round(result.fitness),
        distance_px: Math.round(result.distance),
        overtakes: result.carsPassed,
        legal_stops: result.legalStops,
        red_encounters: result.redEncounters,
        lights_cleared: result.lightsCleared,
        red_light_violations: result.redLightViolation ? 1 : 0,
        collisions: result.crashed ? 1 : 0,
        frames: result.frames,
        sim_seconds: Number((result.elapsedMs / 1000).toFixed(2)),
        outcome: result.outcome + (result.crashCause ? ` (${result.crashCause})` : ""),
        brain_unchanged: result.brainUnchanged ? "yes" : "NO"
    };
}

function generationRow(base, summary, config, gaSeed) {
    return {
        ...base,
        track_seed: summary.trackSeed,
        scenario: summary.stage,
        stage: summary.stage,
        generation: summary.generation,
        population_size: summary.popSize,
        elite_count: config.eliteCount,
        tournament_size: config.tournamentSize,
        mutation_rate: config.mutationRate,
        mutation_strength: config.mutationStrength,
        ga_seed: gaSeed,
        best_fitness: Math.round(summary.best),
        mean_fitness: Math.round(summary.mean),
        median_fitness: Math.round(summary.median),
        mean_distance_px: Math.round(summary.meanDistance),
        mean_overtakes: Number(summary.meanOvertakes.toFixed(3)),
        collisions: summary.collisions,
        collision_rate: Number(summary.collisionRate.toFixed(3)),
        legal_stops: summary.legalStops,
        red_encounters: summary.redEncounters,
        legal_stop_rate: summary.legalStopRate === null ? "" : Number(summary.legalStopRate.toFixed(3)),
        lights_cleared: summary.lightsCleared,
        red_light_violations: summary.redLightViolations,
        diversity: Number(summary.diversity.toFixed(4)),
        frames: summary.frames,
        sim_seconds: Number((summary.frames * SIM_FRAME_MS / 1000).toFixed(2))
    };
}

// ---------------------------------------------------------------------------
// Experiment A: evolved brains against untrained ones (and a hand-coded
// baseline), on the same track with the same rules.
function* experimentBaseline({ road, scenario, trackSeed, champion, population = [],
    untrainedCount = 20, evolvedCount = 5, seed = 1 }) {
    const id = newExperimentId("A");
    const timestamp = new Date().toISOString();
    const base = { experiment: "A-baseline", experiment_id: id, source: "benchmark", timestamp,
        track_type: "training", stage: scenario.key };
    const rows = [];
    const groups = { untrained: [], evolved: [], heuristic: [] };
    const random = createSeededRandom(seed);
    const shape = Car.brainShape(Sensor.RAY_COUNT);

    const evolved = [];
    if (champion) {
        evolved.push({ brain: champion, label: "saved champion" });
    }
    for (const brain of population.slice(0, evolvedCount)) {
        evolved.push({ brain, label: "population elite" });
    }
    const total = untrainedCount + evolved.length + 1;
    let done = 0;

    for (let i = 0; i < untrainedCount; i++) {
        const result = runEpisode({ road, scenario, trackSeed, brain: new NeuralNetwork(shape, random) });
        groups.untrained.push(result);
        rows.push(episodeRow({ ...base, group: "untrained", trial: i + 1 }, result));
        yield { progress: ++done / total, label: `untrained network ${i + 1}/${untrainedCount}` };
    }
    for (let i = 0; i < evolved.length; i++) {
        const result = runEpisode({ road, scenario, trackSeed, brain: evolved[i].brain });
        groups.evolved.push(result);
        rows.push(episodeRow({ ...base, group: "evolved", trial: i + 1, notes: evolved[i].label }, result));
        yield { progress: ++done / total, label: `evolved network ${i + 1}/${evolved.length}` };
    }
    const heuristic = runEpisode({ road, scenario, trackSeed, driver: heuristicDriver });
    groups.heuristic.push(heuristic);
    rows.push(episodeRow({ ...base, group: "heuristic-baseline", trial: 1,
        notes: "hand-coded rules, not produced by the GA" }, heuristic));
    yield { progress: 1, label: "hand-coded baseline" };

    return {
        key: "A", id, rows,
        summary: {
            untrained: summarizeEpisodes(groups.untrained),
            evolved: summarizeEpisodes(groups.evolved),
            heuristic: summarizeEpisodes(groups.heuristic)
        }
    };
}

// ---------------------------------------------------------------------------
// Experiment B: the same GA with different mutation rates, everything else
// held constant, several independent seeded trials per rate.
function* experimentMutation({ road, scenario, trackSeed, rates = [0.05, 0.1, 0.2, 0.3],
    trials = 2, popSize = 30, generations = 10, maxGenerationFrames = 1200, baseSeed = 1000,
    config = {} }) {
    const id = newExperimentId("B");
    const timestamp = new Date().toISOString();
    const base = { experiment: "B-mutation-rate", experiment_id: id, source: "benchmark", timestamp,
        track_type: "training" };
    const rows = [];
    const perRate = {};
    const totalGenerations = rates.length * trials * generations;
    let doneGenerations = 0;

    for (const rate of rates) {
        perRate[rate] = { finalBest: [], finalMean: [], bestDistance: [], bestOvertakes: [], trials: 0 };
        for (let trial = 1; trial <= trials; trial++) {
            const gaSeed = baseSeed + trial;
            const evolution = new Evolution({
                road, scenario, trackSeed,
                config: { ...config, popSize, mutationRate: rate, maxGenerationFrames },
                random: createSeededRandom(gaSeed)
            });
            let summary = null;
            while (evolution.generation <= generations) {
                // Slices of ~150 frames keep each yield short enough for the
                // interface to repaint between them.
                for (let i = 0; i < 150 && summary === null; i++) {
                    summary = evolution.step();
                }
                if (summary === null) {
                    yield { progress: doneGenerations / totalGenerations,
                        label: `rate ${rate}, trial ${trial}, generation ${evolution.generation}` };
                    continue;
                }
                rows.push(generationRow({ ...base, group: `rate ${rate}`, trial }, summary,
                    evolution.config, gaSeed));
                doneGenerations++;
                yield { progress: doneGenerations / totalGenerations,
                    label: `rate ${rate}, trial ${trial}, generation ${summary.generation} done` };
                if (evolution.generation <= generations) {
                    evolution.advance();
                }
                const last = summary;
                summary = null;
                if (evolution.generation > generations) {
                    const stats = perRate[rate];
                    stats.trials++;
                    stats.finalBest.push(last.best);
                    stats.finalMean.push(last.mean);
                    stats.bestDistance.push(last.bestDistance);
                    stats.bestOvertakes.push(last.bestOvertakes);
                }
            }
        }
    }

    const summary = {};
    for (const rate of rates) {
        const stats = perRate[rate];
        summary[rate] = {
            trials: stats.trials,
            meanFinalBest: mean(stats.finalBest),
            meanFinalMean: mean(stats.finalMean),
            meanBestDistance: mean(stats.bestDistance),
            meanBestOvertakes: mean(stats.bestOvertakes),
            finalBests: stats.finalBest
        };
    }
    return { key: "B", id, rows, summary,
        settings: { rates, trials, popSize, generations, maxGenerationFrames, trackSeed, baseSeed } };
}

// ---------------------------------------------------------------------------
// Experiment C: the saved champion on tracks it never trained on. The brain is
// frozen throughout; the training-track result is reported separately.
function* experimentGeneralization({ road, champion, trainingSeed, scenario,
    unseenSeeds = UNSEEN_TRACK_SEEDS, includeDemo = true }) {
    const id = newExperimentId("C");
    const timestamp = new Date().toISOString();
    const base = { experiment: "C-generalization", experiment_id: id, source: "benchmark", timestamp,
        stage: scenario.key, group: "saved champion" };
    const rows = [];
    const seeds = unseenSeeds.filter(seed => seed !== trainingSeed);
    const total = 1 + seeds.length + (includeDemo ? 1 : 0);
    let done = 0;

    const training = runEpisode({ road, scenario, trackSeed: trainingSeed, brain: champion });
    rows.push(episodeRow({ ...base, track_type: "training", trial: 1 }, training));
    yield { progress: ++done / total, label: "training track" };

    const unseen = [];
    for (let i = 0; i < seeds.length; i++) {
        const result = runEpisode({ road, scenario, trackSeed: seeds[i], brain: champion });
        unseen.push(result);
        rows.push(episodeRow({ ...base, track_type: "unseen", trial: i + 1 }, result));
        yield { progress: ++done / total, label: `unseen track ${i + 1}/${seeds.length}` };
    }

    let demo = null;
    if (includeDemo) {
        demo = runEpisode({ road, scenario: DEMO_SCENARIO, trackSeed: 0, brain: champion });
        rows.push(episodeRow({ ...base, track_type: "demo", stage: "DEMO", trial: 1 }, demo));
        yield { progress: 1, label: "demo route" };
    }

    return {
        key: "C", id, rows,
        summary: {
            training: summarizeEpisodes([training]),
            trainingResult: training,
            unseen: summarizeEpisodes(unseen),
            unseenResults: unseen,
            demo: demo ? summarizeEpisodes([demo]) : null,
            demoResult: demo,
            unseenSeeds: seeds
        }
    };
}

// Runs a generator in time slices so the page stays responsive. `onProgress`
// gets each yielded value; `onDone` the generator's return value. runSync
// drains it in one go (used by the tests).
class ExperimentRunner {
    constructor(generator, { sliceMs = 40, onProgress = null, onDone = null, onError = null } = {}) {
        this.generator = generator;
        this.sliceMs = sliceMs;
        this.onProgress = onProgress;
        this.onDone = onDone;
        this.onError = onError;
        this.result = null;
        this.progress = { progress: 0, label: "starting" };
        this.done = false;
        this.cancelled = false;
    }

    static now() {
        return typeof performance !== "undefined" ? performance.now() : Date.now();
    }

    start() {
        const slice = () => {
            if (this.cancelled || this.done) {
                return;
            }
            const until = ExperimentRunner.now() + this.sliceMs;
            try {
                while (ExperimentRunner.now() < until) {
                    const next = this.generator.next();
                    if (next.done) {
                        this.finish(next.value);
                        return;
                    }
                    this.progress = next.value;
                    if (this.onProgress) {
                        this.onProgress(next.value);
                    }
                }
            } catch (error) {
                this.done = true;
                if (this.onError) {
                    this.onError(error);
                } else {
                    throw error;
                }
                return;
            }
            setTimeout(slice, 0);
        };
        setTimeout(slice, 0);
        return this;
    }

    runSync() {
        let next = this.generator.next();
        while (!next.done) {
            this.progress = next.value;
            if (this.onProgress) {
                this.onProgress(next.value);
            }
            next = this.generator.next();
        }
        this.finish(next.value);
        return this.result;
    }

    finish(value) {
        this.result = value;
        this.done = true;
        this.progress = { progress: 1, label: "done" };
        if (this.onDone) {
            this.onDone(value);
        }
    }

    cancel() {
        this.cancelled = true;
    }
}
