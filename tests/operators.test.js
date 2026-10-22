"use strict";

// Unit tests for the genetic algorithm operators, brain persistence, fitness
// shaping and traffic light collision.

const { loadGame, createReporter } = require("./harness");

const { read } = loadGame({
    files: ["utils.js", "network.js", "road.js", "trafficLightSensor.js",
        "trafficLight.js", "controls.js", "sensor.js", "car.js"]
});

const report = createReporter(`operator tests (brain shape ${read("JSON.stringify(Car.brainShape(Sensor.RAY_COUNT))")})`);

report.section("persistence round-trip");

report.check("a saved brain reloads and produces finite outputs", read(`(() => {
    const shape = Car.brainShape(Sensor.RAY_COUNT);
    const original = new NeuralNetwork(shape);
    const revived = NeuralNetwork.fromJSON(JSON.parse(JSON.stringify(original)), shape);
    if (!revived || !(revived instanceof NeuralNetwork)) return false;
    const outputs = NeuralNetwork.feedForward(new Array(shape[0]).fill(0.5), revived);
    return outputs.length === shape[shape.length - 1] && outputs.every(Number.isFinite);
})()`));

report.check("a reloaded brain reproduces the original's outputs exactly", read(`(() => {
    const shape = Car.brainShape(Sensor.RAY_COUNT);
    const original = new NeuralNetwork(shape);
    const inputs = new Array(shape[0]).fill(0).map((_, i) => i / shape[0]);
    const before = [...NeuralNetwork.feedForward(inputs, original)];
    const revived = NeuralNetwork.fromJSON(JSON.parse(JSON.stringify(original)), shape);
    const after = NeuralNetwork.feedForward(inputs, revived);
    return before.every((value, i) => Math.abs(value - after[i]) < 1e-12);
})()`));

report.check("a brain saved for a different sensor count is rejected", read(`
    NeuralNetwork.fromJSON(
        JSON.parse(JSON.stringify(new NeuralNetwork([3, 6, 4]))),
        Car.brainShape(Sensor.RAY_COUNT)
    ) === null`));

report.check("truncated or malformed saved data is rejected", read(`
    NeuralNetwork.fromJSON({ levels: [] }, [7, 8, 4]) === null &&
    NeuralNetwork.fromJSON(null, [7, 8, 4]) === null &&
    NeuralNetwork.fromJSON({ neuronCounts: [7, 8, 4], levels: [{ biases: [1], weights: [[1]] }] }, [7, 8, 4]) === null`));

report.section("mutation operator");

report.check("rate 0 changes nothing", read(`(() => {
    const network = new NeuralNetwork([7, 8, 4]);
    const before = JSON.stringify(network);
    NeuralNetwork.mutate(network, 0, 0.5);
    return JSON.stringify(network) === before;
})()`));

report.check("repeated mutation keeps every gene inside [-1, 1]", read(`(() => {
    const network = new NeuralNetwork([7, 8, 4]);
    const before = JSON.stringify(network);
    for (let i = 0; i < 40; i++) NeuralNetwork.mutate(network, 1, 0.5);
    const genes = network.levels.flatMap(l => [...l.biases, ...l.weights.flat()]);
    return JSON.stringify(network) !== before && genes.every(g => g >= -1 && g <= 1);
})()`));

report.check("rate is a per-gene probability, not a blanket nudge", read(`(() => {
    const network = new NeuralNetwork([7, 8, 4]);
    const before = network.levels.flatMap(l => [...l.biases, ...l.weights.flat()]);
    NeuralNetwork.mutate(network, 0.1, 0.5);
    const after = network.levels.flatMap(l => [...l.biases, ...l.weights.flat()]);
    const changed = before.filter((value, i) => value !== after[i]).length;
    return changed > 0 && changed < before.length * 0.45;
})()`));

report.section("crossover operator");

report.check("every child gene comes from one of the two parents", read(`(() => {
    const a = new NeuralNetwork([7, 8, 4]);
    const b = new NeuralNetwork([7, 8, 4]);
    const child = NeuralNetwork.crossover(a, b);
    for (let l = 0; l < child.levels.length; l++) {
        for (let i = 0; i < child.levels[l].biases.length; i++) {
            const gene = child.levels[l].biases[i];
            if (gene !== a.levels[l].biases[i] && gene !== b.levels[l].biases[i]) return false;
        }
        for (let i = 0; i < child.levels[l].weights.length; i++) {
            for (let j = 0; j < child.levels[l].weights[i].length; j++) {
                const gene = child.levels[l].weights[i][j];
                if (gene !== a.levels[l].weights[i][j] && gene !== b.levels[l].weights[i][j]) return false;
            }
        }
    }
    return true;
})()`));

report.check("the child mixes both parents instead of copying one", read(`(() => {
    const a = new NeuralNetwork([7, 8, 4]);
    const b = new NeuralNetwork([7, 8, 4]);
    const child = NeuralNetwork.crossover(a, b);
    const genes = (n) => n.levels.flatMap(l => [...l.biases, ...l.weights.flat()]);
    const ga = genes(a), gb = genes(b), gc = genes(child);
    return gc.filter((g, i) => g === ga[i]).length > 10 &&
           gc.filter((g, i) => g === gb[i]).length > 10;
})()`));

report.check("crossover leaves both parents untouched", read(`(() => {
    const a = new NeuralNetwork([7, 8, 4]);
    const b = new NeuralNetwork([7, 8, 4]);
    const snapshotA = JSON.stringify(a), snapshotB = JSON.stringify(b);
    NeuralNetwork.mutate(NeuralNetwork.crossover(a, b), 1, 1);
    return JSON.stringify(a) === snapshotA && JSON.stringify(b) === snapshotB;
})()`));

report.section("elitism");

report.check("clone is a deep copy, so an elite survives its children mutating", read(`(() => {
    const original = new NeuralNetwork([7, 8, 4]);
    const snapshot = JSON.stringify(original);
    const copy = NeuralNetwork.clone(original);
    NeuralNetwork.mutate(copy, 1, 1);
    return JSON.stringify(original) === snapshot && JSON.stringify(copy) !== snapshot;
})()`));

report.section("traffic light");

report.check("a car crossing a red stop line is damaged", read(`(() => {
    const road = new Road(110, 180);
    const light = new TrafficLight(road.right + 28, -500, 44, 44, road);
    const car = new Car(road.getLaneCenter(1), -500, 30, 50, "AI");
    car.update({ borders: [], traffic: [], lights: [light], stopLines: [light.stopLine.polygon] });
    return car.damaged === true;
})()`));

report.check("the same car is untouched when the line is not red", read(`(() => {
    const road = new Road(110, 180);
    const light = new TrafficLight(road.right + 28, -500, 44, 44, road);
    const car = new Car(road.getLaneCenter(1), -500, 30, 50, "AI");
    car.update({ borders: [], traffic: [], lights: [light], stopLines: [] });
    return car.damaged === false;
})()`));

report.check("the kill zone spans exactly the road, not 500px of empty space", read(`(() => {
    const road = new Road(110, 180);
    const light = new TrafficLight(road.right + 28, -500, 44, 44, road);
    const xs = light.stopLine.polygon.map(p => p.x);
    return Math.min(...xs) === road.left && Math.max(...xs) === road.right;
})()`));

report.check("each light cycles green -> yellow -> red within one period", read(`(() => {
    const road = new Road(110, 180);
    const light = new TrafficLight(road.right + 28, -500, 44, 44, road, 0);
    const seen = new Set();
    for (let ms = 0; ms < LIGHT_CYCLE_MS; ms += 100) {
        light.update(ms);
        seen.add(light.state);
    }
    return seen.size === 3;
})()`));

report.check("the network's light input is raised by red and zero on green", read(`(() => {
    const road = new Road(110, 180);
    const light = new TrafficLight(road.right + 28, -200, 44, 44, road, 0);
    const world = { borders: [], traffic: [], lights: [light], stopLines: [] };
    const signalAt = (atMs) => {
        light.update(atMs);
        const car = new Car(road.getLaneCenter(1), -100, 30, 50, "AI");
        car.update(world);
        return { state: light.state, signal: car.lightSignal };
    };

    const onRed = signalAt(LIGHT_GREEN_MS + LIGHT_YELLOW_MS + 10);
    const onGreen = signalAt(0);
    return onRed.state === "red" && onRed.signal > 0 &&
           onGreen.state === "green" && onGreen.signal === 0;
})()`));

report.section("fitness shaping");

report.check("crashing costs fitness against the same distance survived", read(`(() => {
    const road = new Road(110, 180);
    const safe = new Car(road.getLaneCenter(1), 0, 30, 50, "AI");
    const wrecked = new Car(road.getLaneCenter(1), 0, 30, 50, "AI");
    safe.maxDistance = 1000;
    wrecked.maxDistance = 1000;
    wrecked.damaged = true;
    return safe.fitness - wrecked.fitness === CRASH_PENALTY;
})()`));

report.check("clearing lights is rewarded", read(`(() => {
    const road = new Road(110, 180);
    const car = new Car(road.getLaneCenter(1), 0, 30, 50, "AI");
    car.maxDistance = 500;
    car.lightsCleared = 2;
    return car.fitness === 500 + 2 * LIGHT_BONUS;
})()`));

report.check("distance is the best reached, so drifting back cannot rewind it", read(`(() => {
    const road = new Road(110, 180);
    // A dummy drives forward unconditionally, so this measures the tracking
    // rather than whatever a random brain happens to do.
    const car = new Car(road.getLaneCenter(1), 0, 30, 50, "DUMMY", 2);
    const world = { borders: [], traffic: [], lights: [], stopLines: [] };
    for (let i = 0; i < 120; i++) car.update(world);
    const peak = car.maxDistance;
    car.y += 300;
    car.update(world);
    return peak > 100 && car.maxDistance === peak;
})()`));

report.finish();
