"use strict";

// The genetic algorithm, as a self-contained object. main.js drives one of
// these on screen; the benchmarks run several headless ones with their own
// seeds. Nothing in here touches the DOM.

// Genetic algorithm parameters (defaults; the interface can change them).
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

// A generation ends when every car is wrecked, retired or finished, or when
// it has simply run long enough.
const MAX_GENERATION_FRAMES = 60 * 60;

// When resuming from a saved brain, this share of the population is random
// "immigrants" rather than mutants of the saved brain. A population descended
// from a single ancestor has almost no diversity, so if that ancestor sits in
// a local optimum the search can take many generations to escape it.
const IMMIGRANT_SHARE = 0.3;

const CONFIG_LIMITS = Object.freeze({
    popSize: { min: 4, max: 400, integer: true },
    eliteCount: { min: 0, max: 20, integer: true },
    tournamentSize: { min: 1, max: 20, integer: true },
    mutationRate: { min: 0, max: 1 },
    mutationStrength: { min: 0.01, max: 1 },
    stagnationLimit: { min: 1, max: 100, integer: true },
    stagnationMutationBoost: { min: 1, max: 5 },
    stagnationImmigrantShare: { min: 0, max: 0.9 },
    maxGenerationFrames: { min: 60, max: 36000, integer: true },
    stallFrames: { min: 30, max: 36000, integer: true }
});

class Evolution {
    static get DEFAULT_CONFIG() {
        return {
            popSize: POP_SIZE,
            eliteCount: ELITE_COUNT,
            tournamentSize: TOURNAMENT_SIZE,
            mutationRate: MUTATION_RATE,
            mutationStrength: MUTATION_STRENGTH,
            stagnationLimit: STAGNATION_LIMIT,
            stagnationMutationBoost: STAGNATION_MUTATION_BOOST,
            stagnationImmigrantShare: STAGNATION_IMMIGRANT_SHARE,
            maxGenerationFrames: MAX_GENERATION_FRAMES,
            stallFrames: STALL_FRAMES
        };
    }

    // Every parameter is forced into its valid range; anything that is not a
    // number keeps its current value (`base`, the defaults unless given).
    // Returns the clean config and a note for each value that had to be
    // changed, so the interface can say so.
    static validateConfig(raw = {}, base = Evolution.DEFAULT_CONFIG) {
        const config = { ...Evolution.DEFAULT_CONFIG, ...base };
        const warnings = [];
        for (const [key, limit] of Object.entries(CONFIG_LIMITS)) {
            if (raw[key] === undefined || raw[key] === null || raw[key] === "") {
                continue;
            }
            let value = Number(raw[key]);
            if (!Number.isFinite(value)) {
                warnings.push(`${key}: "${raw[key]}" is not a number, using ${config[key]}`);
                continue;
            }
            if (limit.integer) {
                value = Math.round(value);
            }
            const clamped = Math.min(limit.max, Math.max(limit.min, value));
            if (clamped !== value) {
                warnings.push(`${key}: ${value} is outside ${limit.min}–${limit.max}, using ${clamped}`);
            }
            config[key] = clamped;
        }
        // Elites and the tournament cannot exceed the population.
        if (config.eliteCount >= config.popSize) {
            warnings.push(`eliteCount: ${config.eliteCount} ≥ population, using ${config.popSize - 1}`);
            config.eliteCount = config.popSize - 1;
        }
        if (config.tournamentSize > config.popSize) {
            warnings.push(`tournamentSize: ${config.tournamentSize} > population, using ${config.popSize}`);
            config.tournamentSize = config.popSize;
        }
        return { config, warnings };
    }

    // `random` is the source of every stochastic decision — initial genes,
    // selection, crossover and mutation — so a run is reproducible from its
    // seed. `seedBrain` starts the population from a saved champion;
    // `brains` starts it from an explicit list (used when a stage changes).
    constructor({ road, scenario = SCENARIOS.A, trackSeed = 1, config = {}, random = Math.random,
        brains = null, seedBrain = null, generation = 1 }) {
        this.road = road;
        this.scenario = scenario;
        this.trackSeed = trackSeed;
        this.config = Evolution.validateConfig(config).config;
        this.random = random;
        this.brainShape = Car.brainShape(Sensor.RAY_COUNT);

        this.generation = generation;
        this.cars = [];
        this.world = null;
        this.bestCar = null;
        this.champion = null;
        this.aliveCount = 0;
        this.activeCount = 0;
        this.frameInGeneration = 0;

        this.history = [];
        this.lastSummary = null;
        this.stagnantGenerations = 0;
        this.lastChampionFitness = null;
        this.breedingSnapshot = null;
        this.pendingBrains = null;
        this.crashMarks = [];
        this.recordedCrashes = new Set();
        this.bestEver = null;

        this.startGeneration(brains || this.seedPopulation(seedBrain));
    }

    get stage() {
        return this.scenario.key;
    }

    // Generation 1: either a fresh random population, or the saved champion
    // kept intact plus mutated copies of it plus some random immigrants.
    seedPopulation(parent = null) {
        const { popSize, mutationRate, mutationStrength } = this.config;
        const brains = [];

        if (parent) {
            const descendants = Math.round(popSize * (1 - IMMIGRANT_SHARE));
            brains.push(NeuralNetwork.clone(parent));
            while (brains.length < descendants) {
                const child = NeuralNetwork.clone(parent);
                NeuralNetwork.mutate(child, mutationRate, mutationStrength, null, this.random);
                brains.push(child);
            }
        }
        while (brains.length < popSize) {
            brains.push(new NeuralNetwork(this.brainShape, this.random));
        }
        return brains;
    }

    // Change the track recipe. Takes effect when the next generation starts;
    // the stagnation counter resets because fitness on the new track is not
    // comparable with fitness on the old one.
    setScenario(scenario) {
        if (scenario.key === this.scenario.key) {
            return;
        }
        this.scenario = scenario;
        this.stagnantGenerations = 0;
        this.lastChampionFitness = null;
    }

    startGeneration(brains) {
        // The world is rebuilt from the seed every generation, so the track is
        // byte-identical each time and a scenario change simply takes effect.
        this.world = new World(this.road, buildTrack(this.trackSeed, this.scenario), {
            stallFrames: this.config.stallFrames,
            routeLength: null
        });
        this.cars = brains.map(brain => this.world.spawnCar(brain));
        this.bestCar = this.cars[0];
        this.champion = this.cars[0];
        this.aliveCount = this.cars.length;
        this.activeCount = this.cars.length;
        this.crashMarks = [];
        this.recordedCrashes = new Set();
        this.frameInGeneration = 0;
        this.pendingBrains = null;
    }

    // One simulation step. Returns the generation summary when this step ended
    // the generation (the next population is then waiting in pendingBrains
    // until advance() is called), otherwise null.
    step() {
        this.world.step(this.cars);
        this.frameInGeneration++;

        // The camera follows the leading active survivor; it used to follow
        // whichever car was furthest up the road even if that car was a wreck.
        this.aliveCount = 0;
        this.activeCount = 0;
        let leader = null;
        let champion = this.cars[0];
        for (const car of this.cars) {
            if (car.damaged) {
                if (!this.recordedCrashes.has(car)) {
                    this.recordedCrashes.add(car);
                    this.crashMarks.push({ x: car.x, y: car.y });
                }
            } else {
                this.aliveCount++;
            }
            if (car.active) {
                this.activeCount++;
                if (leader === null || car.fitness > leader.fitness) {
                    leader = car;
                }
            }
            if (car.fitness > champion.fitness) {
                champion = car;
            }
        }
        this.champion = champion;
        this.bestCar = leader !== null ? leader : champion;

        if (this.activeCount === 0 || this.frameInGeneration >= this.config.maxGenerationFrames) {
            return this.endGeneration();
        }
        return null;
    }

    // Every number the interface, the chart and the CSV export report for a
    // generation. Rates whose denominator is zero are null, not 0: "no car
    // met a red light" is different from "every car ran the red light".
    summarize() {
        const cars = this.cars;
        const fitnesses = cars.map(car => car.fitness);
        const best = cars.reduce((a, b) => (b.fitness > a.fitness ? b : a), cars[0]);
        const crashed = cars.filter(car => car.damaged).length;
        const legalStops = cars.reduce((sum, car) => sum + car.legalStops, 0);
        const redEncounters = cars.reduce((sum, car) => sum + car.redEncounters, 0);
        const lightsCleared = cars.reduce((sum, car) => sum + car.lightsCleared, 0);
        const carsMetLight = cars.filter(car => car.lightsMet > 0).length;
        const carsClearedLight = cars.filter(car => car.lightsCleared > 0).length;

        return {
            generation: this.generation,
            stage: this.stage,
            trackSeed: this.trackSeed,
            frames: this.frameInGeneration,
            popSize: cars.length,
            best: best.fitness,
            mean: mean(fitnesses),
            median: median(fitnesses),
            meanDistance: mean(cars.map(car => car.maxDistance)),
            meanOvertakes: mean(cars.map(car => car.carsPassed)),
            meanLegalStops: cars.length ? legalStops / cars.length : 0,
            collisions: crashed,
            collisionRate: cars.length ? crashed / cars.length : 0,
            redLightViolations: cars.filter(car => car.redLightViolation).length,
            legalStops,
            redEncounters,
            legalStopRate: redEncounters > 0 ? legalStops / redEncounters : null,
            lightsCleared,
            carsMetLight,
            carsClearedLight,
            lightClearRate: carsMetLight > 0 ? carsClearedLight / carsMetLight : null,
            diversity: NeuralNetwork.diversity(cars.map(car => car.brain)),
            survivors: cars.length - crashed,
            bestDistance: best.maxDistance,
            bestOvertakes: best.carsPassed,
            bestLegalStops: best.legalStops,
            bestLightsCleared: best.lightsCleared,
            bestCrashed: best.damaged,
            bestBrain: best.brain,
            bestBreakdown: best.fitnessBreakdown
        };
    }

    endGeneration() {
        const summary = this.summarize();
        this.lastSummary = summary;
        this.history.push(summary);

        if (!this.bestEver || summary.best > this.bestEver.fitness) {
            this.bestEver = {
                fitness: summary.best,
                distance: summary.bestDistance,
                generation: this.generation,
                stage: this.stage
            };
        }

        if (this.lastChampionFitness !== null && Math.abs(summary.best - this.lastChampionFitness) < 1) {
            this.stagnantGenerations++;
        } else {
            this.stagnantGenerations = 0;
        }
        this.lastChampionFitness = summary.best;

        const next = this.buildNextPopulation(this.cars);
        this.breedingSnapshot = this.buildBreedingSnapshot(next, summary);
        this.pendingBrains = next.brains;
        this.generation++;
        return summary;
    }

    // Start the next generation from the population bred at the end of the
    // last one. Kept separate from endGeneration so the interface can hold
    // the breeding panel on screen in between.
    advance() {
        if (this.pendingBrains) {
            this.startGeneration(this.pendingBrains);
        }
    }

    // Tournament selection: sample a few individuals and take the best of them.
    // Fitness can go negative once a crash penalty applies, which rules out
    // roulette-wheel selection without rescaling.
    tournamentSelect(population) {
        const size = this.config.tournamentSize;
        let winner = population[Math.floor(this.random() * population.length)];
        for (let i = 1; i < size; i++) {
            const challenger = population[Math.floor(this.random() * population.length)];
            if (challenger.fitness > winner.fitness) {
                winner = challenger;
            }
        }
        return winner;
    }

    buildNextPopulation(population) {
        const { popSize, eliteCount, stagnationLimit, stagnationMutationBoost,
            stagnationImmigrantShare } = this.config;
        const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
        const brains = [];
        const pairs = [];
        let geneTrace = null;

        const stagnant = this.stagnantGenerations >= stagnationLimit;
        const mutationRate = stagnant
            ? Math.min(1, this.config.mutationRate * stagnationMutationBoost)
            : this.config.mutationRate;
        const mutationStrength = stagnant
            ? Math.min(1, this.config.mutationStrength * stagnationMutationBoost)
            : this.config.mutationStrength;
        const immigrants = stagnant ? Math.round(popSize * stagnationImmigrantShare) : 0;

        // Elitism: the best brains cross over untouched, so a generation can
        // never come out strictly worse than the one that produced it.
        for (let i = 0; i < eliteCount && i < ranked.length && brains.length < popSize; i++) {
            brains.push(NeuralNetwork.clone(ranked[i].brain));
        }

        while (brains.length < popSize - immigrants) {
            const parentA = this.tournamentSelect(ranked);
            const parentB = this.tournamentSelect(ranked);

            // The first child's gene origins and the first few pairings are
            // recorded so the breeding panel shows what actually happened rather
            // than an illustration of it.
            const origins = geneTrace === null ? [] : null;
            const child = NeuralNetwork.crossover(parentA.brain, parentB.brain, origins, this.random);
            const mutated = origins ? [] : null;
            NeuralNetwork.mutate(child, mutationRate, mutationStrength, mutated, this.random);

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

        while (brains.length < popSize) {
            brains.push(new NeuralNetwork(this.brainShape, this.random));
        }

        return { brains, pairs, geneTrace, ranked, immigrants, mutationRate };
    }

    buildBreedingSnapshot(next, summary) {
        const { ranked, pairs, geneTrace, immigrants, mutationRate } = next;
        return {
            generation: this.generation,
            stage: this.stage,
            eliteCount: Math.min(this.config.eliteCount, this.config.popSize),
            childCount: next.brains.length - Math.min(this.config.eliteCount, this.config.popSize) - immigrants,
            tournamentSize: this.config.tournamentSize,
            mutationPercent: Math.round(mutationRate * 100),
            geneCount: NeuralNetwork.geneCount(this.brainShape),
            immigrants,
            stagnantGenerations: this.stagnantGenerations,
            meanFitness: Math.round(summary.mean),
            medianFitness: Math.round(summary.median),
            diversity: summary.diversity,
            pairs,
            geneTrace,
            rows: ranked.slice(0, 8).map((car, index) => ({
                rank: index + 1,
                fitness: Math.round(car.fitness),
                distance: Math.round(car.maxDistance),
                carsPassed: car.carsPassed,
                legalStops: car.legalStops,
                lightsCleared: car.lightsCleared,
                crashed: car.damaged
            }))
        };
    }
}

// Staged training. The population starts on stage A and is promoted when its
// measured behaviour, not elapsed time, clears the bar — two generations
// running, so a single lucky generation does not promote it. Set the mode to
// a stage letter to pin training to that stage.
const CURRICULUM = Object.freeze({
    A: {
        next: "B",
        consecutive: 2,
        describe: "champion overtakes ≥ 4 cars and the population averages ≥ 0.5 overtakes",
        test: summary => summary.bestOvertakes >= 4 && summary.meanOvertakes >= 0.5
    },
    B: {
        next: "C",
        consecutive: 2,
        describe: "champion makes ≥ 2 legal stops and clears ≥ 3 lights, population averages ≥ 0.15 legal stops",
        test: summary => summary.bestLegalStops >= 2 && summary.bestLightsCleared >= 3 &&
            summary.meanLegalStops >= 0.15
    },
    C: {
        next: null,
        consecutive: Infinity,
        describe: "final stage",
        test: () => false
    }
});

class Curriculum {
    constructor(mode = "auto", stage = "A") {
        this.mode = Curriculum.MODES.includes(mode) ? mode : "auto";
        this.stage = this.mode !== "auto" ? this.mode : (CURRICULUM[stage] ? stage : "A");
        this.streak = 0;
    }

    static get MODES() {
        return ["auto", ...STAGE_ORDER];
    }

    get scenario() {
        return SCENARIOS[this.stage];
    }

    get rule() {
        return CURRICULUM[this.stage];
    }

    setMode(mode) {
        this.mode = Curriculum.MODES.includes(mode) ? mode : "auto";
        if (this.mode !== "auto") {
            this.stage = this.mode;
        }
        this.streak = 0;
    }

    // Feed a generation summary in. Returns the new stage key if the
    // population was just promoted, otherwise null.
    observe(summary) {
        if (this.mode !== "auto") {
            return null;
        }
        const rule = this.rule;
        if (!rule.next) {
            return null;
        }
        this.streak = rule.test(summary) ? this.streak + 1 : 0;
        if (this.streak >= rule.consecutive) {
            this.stage = rule.next;
            this.streak = 0;
            return this.stage;
        }
        return null;
    }
}

function stageRank(stage) {
    const index = STAGE_ORDER.indexOf(stage);
    return index < 0 ? -1 : index;
}
