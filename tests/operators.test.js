"use strict";

// Unit tests for the neural network, the genetic algorithm operators, brain
// persistence, the car's controls and the traffic rules. Nothing here runs
// the page; the headless files are loaded without main.js.

const { loadGame, createReporter, HEADLESS_FILES } = require("./harness");

const { read } = loadGame({ files: HEADLESS_FILES });

const report = createReporter(`operator tests (brain shape ${read("JSON.stringify(Car.brainShape(Sensor.RAY_COUNT))")})`);

// A minimal world: borders, optional traffic/lights, no stop lines unless red.
read(`
    var road = new Road(110, 180);
    function worldWith({ traffic = [], lights = [] } = {}) {
        return {
            borders: road.borders, traffic, lights,
            stopLines: lights.filter(l => l.isRed).map(l => l.stopLine.polygon)
        };
    }
    function aiCar(lane = 1, y = 0) {
        const car = new Car(road.getLaneCenter(lane), y, 30, 50, "AI");
        return car;
    }
    // A controller that returns fixed commands, bypassing the network.
    function fixedDriver(forward, left, right, brake) {
        return () => [forward ? 1 : -1, left ? 1 : -1, right ? 1 : -1, brake ? 1 : -1];
    }
`);

report.section("network dimensions");

report.check("the network takes 10 inputs, has 8 hidden neurons and 4 outputs",
    read("JSON.stringify(Car.brainShape(Sensor.RAY_COUNT))") === "[10,8,4]");

report.check("the genome has 124 genes (80 + 8 + 32 + 4)",
    read("NeuralNetwork.geneCount([10, 8, 4])") === 124 &&
    read("NeuralNetwork.genes(new NeuralNetwork([10, 8, 4])).length") === 124);

report.check("the car builds exactly 10 inputs, one per label", read(`(() => {
    const car = aiCar();
    car.update(worldWith());
    return car.inputs.length === 10 && Car.INPUT_LABELS.length === 10;
})()`));

report.check("feed-forward produces 4 finite outputs in [-1, 1] for any finite input", read(`(() => {
    const net = new NeuralNetwork([10, 8, 4]);
    for (const value of [0, 1, -1, 0.5, 1e6]) {
        const out = NeuralNetwork.feedForward(new Array(10).fill(value), net);
        if (out.length !== 4 || !out.every(o => Number.isFinite(o) && o >= -1 && o <= 1)) return false;
    }
    return true;
})()`));

report.check("non-finite inputs are treated as 0, never propagated as NaN", read(`(() => {
    const net = new NeuralNetwork([10, 8, 4]);
    const out = NeuralNetwork.feedForward([NaN, undefined, Infinity, 0, 0, 0, 0, 0, 0, 0], net);
    return out.every(Number.isFinite);
})()`));

report.check("a fresh network has no NaN or undefined genes", read(`(() => {
    const genes = NeuralNetwork.genes(new NeuralNetwork([10, 8, 4]));
    return genes.every(g => typeof g === "number" && Number.isFinite(g) && g >= -1 && g <= 1);
})()`));

report.section("persistence round-trip");

report.check("a saved brain reloads and produces finite outputs", read(`(() => {
    const shape = Car.brainShape(Sensor.RAY_COUNT);
    const original = new NeuralNetwork(shape);
    const revived = NeuralNetwork.fromJSON(JSON.parse(JSON.stringify(NeuralNetwork.toJSON(original))), shape);
    if (!revived || !(revived instanceof NeuralNetwork)) return false;
    const outputs = NeuralNetwork.feedForward(new Array(shape[0]).fill(0.5), revived);
    return outputs.length === shape[shape.length - 1] && outputs.every(Number.isFinite);
})()`));

report.check("a reloaded brain reproduces the original's outputs exactly", read(`(() => {
    const shape = Car.brainShape(Sensor.RAY_COUNT);
    const original = new NeuralNetwork(shape);
    const inputs = new Array(shape[0]).fill(0).map((_, i) => i / shape[0]);
    const before = [...NeuralNetwork.feedForward(inputs, original)];
    const revived = NeuralNetwork.fromJSON(JSON.parse(JSON.stringify(NeuralNetwork.toJSON(original))), shape);
    const after = NeuralNetwork.feedForward(inputs, revived);
    return NeuralNetwork.equals(original, revived) && before.every((value, i) => Math.abs(value - after[i]) < 1e-12);
})()`));

report.check("an old 7-input brain is rejected, not silently loaded", read(`
    NeuralNetwork.fromJSON(NeuralNetwork.toJSON(new NeuralNetwork([7, 8, 4])), Car.brainShape(Sensor.RAY_COUNT)) === null`));

report.check("truncated or malformed saved data is rejected", read(`
    NeuralNetwork.fromJSON({ levels: [] }, [10, 8, 4]) === null &&
    NeuralNetwork.fromJSON(null, [10, 8, 4]) === null &&
    NeuralNetwork.fromJSON({ neuronCounts: [10, 8, 4], levels: [{ biases: [1], weights: [[1]] }] }, [10, 8, 4]) === null`));

report.check("a genome containing NaN, a string or null is rejected", read(`(() => {
    const good = NeuralNetwork.toJSON(new NeuralNetwork([10, 8, 4]));
    const poisoned = (value) => {
        const copy = JSON.parse(JSON.stringify(good));
        copy.levels[0].weights[3][2] = value;
        return NeuralNetwork.fromJSON(copy, [10, 8, 4]);
    };
    return poisoned(null) === null && poisoned("0.5") === null && poisoned(undefined) === null &&
        NeuralNetwork.fromJSON(good, [10, 8, 4]) !== null;
})()`));

report.section("mutation operator");

report.check("rate 0 changes nothing", read(`(() => {
    const network = new NeuralNetwork([10, 8, 4]);
    const before = JSON.stringify(network);
    NeuralNetwork.mutate(network, 0, 0.5);
    return JSON.stringify(network) === before;
})()`));

report.check("repeated mutation keeps every gene inside [-1, 1]", read(`(() => {
    const network = new NeuralNetwork([10, 8, 4]);
    const before = JSON.stringify(network);
    for (let i = 0; i < 40; i++) NeuralNetwork.mutate(network, 1, 0.5);
    const genes = NeuralNetwork.genes(network);
    return JSON.stringify(network) !== before && genes.every(g => g >= -1 && g <= 1);
})()`));

report.check("rate is a per-gene probability, not a blanket nudge", read(`(() => {
    const network = new NeuralNetwork([10, 8, 4]);
    const before = NeuralNetwork.genes(network);
    NeuralNetwork.mutate(network, 0.1, 0.5);
    const after = NeuralNetwork.genes(network);
    const changed = before.filter((value, i) => value !== after[i]).length;
    return changed > 0 && changed < before.length * 0.45;
})()`));

report.check("mutation with the same seeded random source is reproducible", read(`(() => {
    const a = new NeuralNetwork([10, 8, 4], createSeededRandom(9));
    const b = new NeuralNetwork([10, 8, 4], createSeededRandom(9));
    NeuralNetwork.mutate(a, 0.3, 0.5, null, createSeededRandom(5));
    NeuralNetwork.mutate(b, 0.3, 0.5, null, createSeededRandom(5));
    return NeuralNetwork.equals(a, b);
})()`));

report.section("crossover operator");

report.check("every child gene comes from one of the two parents", read(`(() => {
    const a = new NeuralNetwork([10, 8, 4]);
    const b = new NeuralNetwork([10, 8, 4]);
    const child = NeuralNetwork.crossover(a, b);
    const ga = NeuralNetwork.genes(a), gb = NeuralNetwork.genes(b), gc = NeuralNetwork.genes(child);
    return gc.every((g, i) => g === ga[i] || g === gb[i]);
})()`));

report.check("the child mixes both parents instead of copying one", read(`(() => {
    const a = new NeuralNetwork([10, 8, 4]);
    const b = new NeuralNetwork([10, 8, 4]);
    const child = NeuralNetwork.crossover(a, b);
    const ga = NeuralNetwork.genes(a), gb = NeuralNetwork.genes(b), gc = NeuralNetwork.genes(child);
    return gc.filter((g, i) => g === ga[i]).length > 10 &&
           gc.filter((g, i) => g === gb[i]).length > 10;
})()`));

report.check("the origin trace has one entry per gene and names both parents", read(`(() => {
    const origins = [];
    NeuralNetwork.crossover(new NeuralNetwork([10, 8, 4]), new NeuralNetwork([10, 8, 4]), origins);
    return origins.length === 124 && new Set(origins).size === 2;
})()`));

report.check("crossover leaves both parents untouched", read(`(() => {
    const a = new NeuralNetwork([10, 8, 4]);
    const b = new NeuralNetwork([10, 8, 4]);
    const snapshotA = JSON.stringify(a), snapshotB = JSON.stringify(b);
    NeuralNetwork.mutate(NeuralNetwork.crossover(a, b), 1, 1);
    return JSON.stringify(a) === snapshotA && JSON.stringify(b) === snapshotB;
})()`));

report.section("elitism, selection and population size");

report.check("clone is a deep copy, so an elite survives its children mutating", read(`(() => {
    const original = new NeuralNetwork([10, 8, 4]);
    const snapshot = JSON.stringify(original);
    const copy = NeuralNetwork.clone(original);
    NeuralNetwork.mutate(copy, 1, 1);
    return JSON.stringify(original) === snapshot && JSON.stringify(copy) !== snapshot;
})()`));

read(`
    var evo = new Evolution({ road, scenario: SCENARIOS.A, trackSeed: 3,
        config: { popSize: 20, eliteCount: 2, tournamentSize: 3 }, random: createSeededRandom(4) });
    // Give the cars distinct fitness without driving them.
    evo.cars.forEach((car, i) => { car.maxDistance = i * 37; });
`);

report.check("the elites are copied into the next generation unchanged", read(`(() => {
    const ranked = [...evo.cars].sort((a, b) => b.fitness - a.fitness);
    const next = evo.buildNextPopulation(evo.cars);
    return NeuralNetwork.equals(next.brains[0], ranked[0].brain) &&
           NeuralNetwork.equals(next.brains[1], ranked[1].brain) &&
           next.brains[0] !== ranked[0].brain;
})()`));

report.check("the next population has exactly popSize brains", read(`(() => {
    const next = evo.buildNextPopulation(evo.cars);
    return next.brains.length === evo.config.popSize;
})()`));

report.check("tournament selection returns the fittest of the cars it sampled", read(`(() => {
    // Drive the sampling with a scripted random source so the three sampled
    // indices are known (4, 9 and 17 of 20), then the winner must be the
    // fittest of exactly those three.
    const picks = [4 / 20, 9 / 20, 17 / 20];
    let i = 0;
    const saved = evo.random;
    evo.random = () => picks[i++ % picks.length];
    const winner = evo.tournamentSelect(evo.cars);
    evo.random = saved;
    const sampled = [evo.cars[4], evo.cars[9], evo.cars[17]];
    const best = sampled.reduce((a, b) => (b.fitness > a.fitness ? b : a));
    return winner === best;
})()`));

report.check("selection favours fitter parents (mean parent fitness above population mean)", read(`(() => {
    let total = 0;
    for (let i = 0; i < 400; i++) total += evo.tournamentSelect(evo.cars).fitness;
    const populationMean = evo.cars.reduce((s, c) => s + c.fitness, 0) / evo.cars.length;
    return total / 400 > populationMean * 1.2;
})()`));

report.check("stagnation adds immigrants and boosts mutation; otherwise none", read(`(() => {
    evo.stagnantGenerations = 0;
    const calm = evo.buildNextPopulation(evo.cars);
    evo.stagnantGenerations = evo.config.stagnationLimit;
    const stuck = evo.buildNextPopulation(evo.cars);
    evo.stagnantGenerations = 0;
    return calm.immigrants === 0 && calm.mutationRate === evo.config.mutationRate &&
           stuck.immigrants === Math.round(evo.config.popSize * evo.config.stagnationImmigrantShare) &&
           stuck.mutationRate > evo.config.mutationRate &&
           stuck.brains.length === evo.config.popSize;
})()`));

report.section("parameter validation");

report.check("defaults are returned when nothing is given", read(`(() => {
    const { config, warnings } = Evolution.validateConfig({});
    return config.popSize === POP_SIZE && config.mutationRate === MUTATION_RATE && warnings.length === 0;
})()`));

report.check("out-of-range and non-numeric values are clamped or replaced, with a warning each", read(`(() => {
    const { config, warnings } = Evolution.validateConfig({ popSize: 100000, mutationRate: -3, tournamentSize: "abc", eliteCount: 2.6 });
    return config.popSize === 400 && config.mutationRate === 0 && config.tournamentSize === TOURNAMENT_SIZE &&
           config.eliteCount === 3 && warnings.length === 3;
})()`));

report.check("elite count and tournament size can never exceed the population", read(`(() => {
    const { config } = Evolution.validateConfig({ popSize: 5, eliteCount: 9, tournamentSize: 9 });
    return config.eliteCount === 4 && config.tournamentSize === 5;
})()`));

report.check("an Evolution built from an invalid config still has a valid population", read(`(() => {
    const e = new Evolution({ road, scenario: SCENARIOS.A, trackSeed: 1, config: { popSize: -5, mutationRate: "x" } });
    return e.cars.length === 4 && e.config.mutationRate === MUTATION_RATE;
})()`));

report.section("braking and control arbitration");

report.check("braking reduces speed to zero and never below (no reverse)", read(`(() => {
    const car = aiCar();
    car.speed = 3;
    car.driver = fixedDriver(false, false, false, true);
    const world = worldWith();
    const yStart = car.y;
    const speeds = [];
    for (let i = 0; i < 40; i++) { car.update(world); speeds.push(car.speed); }
    return speeds.every(s => s >= 0) && car.speed === 0 && car.y <= yStart &&
           speeds[0] < 3 && speeds.slice(0, 9).every((s, i) => i === 0 || s <= speeds[i - 1]);
})()`));

// A command issued in one frame is acted on in the next (sense, decide, then
// move), so these drive two frames before reading the decision.
report.check("brake overrides accelerate when both are commanded", read(`(() => {
    const car = aiCar();
    car.speed = 3;
    car.driver = fixedDriver(true, false, false, true);
    car.update(worldWith());
    car.update(worldWith());
    return car.decision.brake && !car.decision.accelerate && car.speed < 2.9;
})()`));

report.check("left and right together cancel instead of fighting", read(`(() => {
    const car = aiCar();
    car.speed = 2;
    car.driver = fixedDriver(true, true, true, false);
    for (let i = 0; i < 30; i++) car.update(worldWith());
    return car.angle === 0 && !car.decision.left && !car.decision.right && car.maxDistance > 30;
})()`));

report.check("a lone steering command does turn the car", read(`(() => {
    const car = aiCar();
    car.speed = 2;
    car.driver = fixedDriver(true, true, false, false);
    for (let i = 0; i < 10; i++) car.update(worldWith());
    return car.angle > 0.2 && car.decision.left;
})()`));

report.check("a stationary car cannot pivot", read(`(() => {
    const car = aiCar();
    car.speed = 0;
    car.driver = fixedDriver(false, true, false, false);
    for (let i = 0; i < 10; i++) car.update(worldWith());
    return car.angle === 0 && car.speed === 0;
})()`));

report.check("the reported action matches the arbitrated decision", read(`(() => {
    const car = aiCar();
    car.speed = 2;
    car.driver = fixedDriver(true, false, true, false);
    car.update(worldWith());
    car.update(worldWith());
    return car.actionLabel === "ACCELERATE + STEER RIGHT";
})()`));

report.section("obstacle detection");

report.check("the forward ray reads an obstacle ahead and nothing on an empty road", read(`(() => {
    const dummy = new Car(road.getLaneCenter(1), -150, 30, 50, "DUMMY", 0);
    const clear = aiCar();
    clear.update(worldWith());
    const blocked = aiCar();
    blocked.update(worldWith({ traffic: [dummy] }));
    const fwd = Car.INPUT_LABELS.indexOf("FWD");
    return clear.inputs[fwd] === 0 && blocked.inputs[fwd] > 0.3;
})()`));

report.check("the side rays see the road borders", read(`(() => {
    const car = aiCar(0);
    car.update(worldWith());
    return car.inputs[0] > 0.5 && car.inputs[4] < car.inputs[0];
})()`));

report.section("traffic lights and the network's light inputs");

read(`
    function lightAt(y, phase = 0, durations) {
        return new TrafficLight(road.right + 28, y, 44, 44, road, phase, durations);
    }
    function withState(light, state) {
        const d = light.durations;
        light.update(state === "green" ? 0 : state === "yellow" ? d.green + 1 : d.green + d.yellow + 1);
        return light;
    }
    var RED = Car.INPUT_LABELS.indexOf("RED"), YEL = Car.INPUT_LABELS.indexOf("YEL"),
        GRN = Car.INPUT_LABELS.indexOf("GRN"), DIST = Car.INPUT_LABELS.indexOf("DIST");
`);

report.check("light colour is one-hot and distance is separate", read(`(() => {
    const results = {};
    for (const state of ["red", "yellow", "green"]) {
        const light = withState(lightAt(-200), state);
        const car = aiCar(1, -70);
        car.update(worldWith({ lights: [light] }));
        results[state] = car.inputs.slice(RED, DIST + 1);
    }
    const oneHot = (v, idx) => v.every((x, i) => (i === 3 ? x > 0.4 && x < 0.6 : x === (i === idx ? 1 : 0)));
    return oneHot(results.red, 0) && oneHot(results.yellow, 1) && oneHot(results.green, 2);
})()`));

report.check("with no light in range all four light inputs are 0", read(`(() => {
    const light = withState(lightAt(-2000), "red");
    const car = aiCar(1, 0);
    car.update(worldWith({ lights: [light] }));
    return car.inputs.slice(RED, DIST + 1).every(v => v === 0) && car.lightState === null;
})()`));

report.check("a car crossing a red stop line is damaged and marked as a red-light violation", read(`(() => {
    const light = withState(lightAt(-500), "red");
    const car = aiCar(1, -500);
    car.update(worldWith({ lights: [light] }));
    return car.damaged === true && car.redLightViolation === true && car.crashCause === "redLight" &&
           car.fitness < -CRASH_PENALTY;
})()`));

report.check("crossing the same line on green is not penalised and counts as a cleared light", read(`(() => {
    const light = withState(lightAt(-500), "green");
    const car = aiCar(1, -480);
    car.speed = 3;
    car.driver = fixedDriver(true, false, false, false);
    const world = worldWith({ lights: [light] });
    for (let i = 0; i < 30; i++) car.update(world);
    return !car.damaged && !car.redLightViolation && car.lightsCleared === 1 &&
           car.fitness >= LIGHT_BONUS;
})()`));

report.check("the stop line spans exactly the road", read(`(() => {
    const light = lightAt(-500);
    const xs = light.stopLine.polygon.map(p => p.x);
    return Math.min(...xs) === road.left && Math.max(...xs) === road.right;
})()`));

report.check("each light cycles green -> yellow -> red within one period", read(`(() => {
    const light = lightAt(-500);
    const seen = new Set();
    for (let ms = 0; ms < LIGHT_CYCLE_MS; ms += 100) {
        light.update(ms);
        seen.add(light.state);
    }
    return seen.size === 3;
})()`));

report.check("custom light durations are honoured (demo route lights)", read(`(() => {
    const light = lightAt(-500, 0, { green: 1000, yellow: 500, red: 3000 });
    light.update(900); const a = light.state;
    light.update(1200); const b = light.state;
    light.update(4000); const c = light.state;
    light.update(4600); const d = light.state;
    return a === "green" && b === "yellow" && c === "red" && d === "green";
})()`));

report.section("legal stops");

report.check("a legal stop is counted once when the car halts in the zone on red, then cleared on green", read(`(() => {
    const light = withState(lightAt(-300), "red");
    const car = aiCar(1, -200);
    car.speed = 2;
    car.driver = fixedDriver(false, false, false, true);
    const world = worldWith({ lights: [light] });
    for (let i = 0; i < 60; i++) car.update(world);
    const stoppedOnce = car.legalStops === 1 && car.speed === 0 && !car.damaged && car.redEncounters === 1;
    // Sitting there longer does not add more stops.
    for (let i = 0; i < 60; i++) car.update(world);
    const stillOne = car.legalStops === 1;
    // Light goes green: the car resumes and clears the light legally.
    withState(light, "green");
    const green = worldWith({ lights: [light] });
    car.driver = fixedDriver(true, false, false, false);
    for (let i = 0; i < 120; i++) car.update(green);
    return stoppedOnce && stillOne && car.lightsCleared === 1 && !car.damaged &&
           car.fitness >= LEGAL_STOP_BONUS + LIGHT_BONUS;
})()`));

report.check("stopping far short of the line is not a legal stop", read(`(() => {
    const light = withState(lightAt(-500), "red");
    const car = aiCar(1, -100);
    car.speed = 2;
    car.driver = fixedDriver(false, false, false, true);
    const world = worldWith({ lights: [light] });
    for (let i = 0; i < 60; i++) car.update(world);
    return car.speed === 0 && car.legalStops === 0;
})()`));

report.check("stopping in the zone while the light is green earns nothing", read(`(() => {
    const light = withState(lightAt(-300), "green");
    const car = aiCar(1, -200);
    car.speed = 2;
    car.driver = fixedDriver(false, false, false, true);
    for (let i = 0; i < 60; i++) car.update(worldWith({ lights: [light] }));
    return car.speed === 0 && car.legalStops === 0;
})()`));

report.section("fitness shaping");

report.check("crashing costs fitness against the same distance survived", read(`(() => {
    const safe = aiCar();
    const wrecked = aiCar();
    safe.maxDistance = 1000;
    wrecked.maxDistance = 1000;
    wrecked.damaged = true;
    return safe.fitness - wrecked.fitness === CRASH_PENALTY;
})()`));

report.check("overtakes, legal stops and cleared lights are each rewarded once per event", read(`(() => {
    const car = aiCar();
    car.maxDistance = 500;
    car.carsPassed = 2;
    car.legalStops = 1;
    car.lightsCleared = 3;
    return car.fitness === 500 + 2 * OVERTAKE_BONUS + LEGAL_STOP_BONUS + 3 * LIGHT_BONUS;
})()`));

report.check("an overtake is counted once per car passed", read(`(() => {
    const dummy = new Car(road.getLaneCenter(0), -100, 30, 50, "DUMMY", 0);
    const car = aiCar(2, 0);
    car.speed = 3;
    car.driver = fixedDriver(true, false, false, false);
    const world = worldWith({ traffic: [dummy] });
    for (let i = 0; i < 120; i++) car.update(world);
    return car.carsPassed === 1 && car.y < dummy.y - 100;
})()`));

report.check("distance is the best reached, so drifting back cannot rewind it", read(`(() => {
    const car = new Car(road.getLaneCenter(1), 0, 30, 50, "DUMMY", 2);
    const world = worldWith();
    for (let i = 0; i < 120; i++) car.update(world);
    const peak = car.maxDistance;
    car.y += 300;
    car.update(world);
    return peak > 100 && car.maxDistance === peak;
})()`));

report.check("a stationary car earns nothing and is retired by the world as stalled", read(`(() => {
    const world = new World(road, buildTrack(1, SCENARIOS.A), { stallFrames: 50, routeLength: null });
    const car = world.spawnCar(new NeuralNetwork([10, 8, 4]));
    car.speed = 0;
    car.driver = fixedDriver(false, false, false, true);
    for (let i = 0; i < 80 && car.active; i++) world.step([car]);
    return car.retired && !car.damaged && car.fitness === 0;
})()`));

report.section("curriculum");

report.check("promotion needs the bar cleared in consecutive generations", read(`(() => {
    const c = new Curriculum("auto", "A");
    const good = { bestOvertakes: 5, meanOvertakes: 0.8 };
    const bad = { bestOvertakes: 1, meanOvertakes: 0.1 };
    const first = c.observe(good);
    const reset = c.observe(bad);
    c.observe(good);
    const promoted = c.observe(good);
    return first === null && reset === null && promoted === "B" && c.stage === "B";
})()`));

report.check("a pinned stage never promotes", read(`(() => {
    const c = new Curriculum("A", "C");
    for (let i = 0; i < 5; i++) c.observe({ bestOvertakes: 9, meanOvertakes: 5 });
    return c.stage === "A";
})()`));

report.check("stage C is final", read(`(() => {
    const c = new Curriculum("auto", "C");
    return c.observe({}) === null && c.stage === "C";
})()`));

report.finish();
