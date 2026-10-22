"use strict";

class NeuralNetwork {
    constructor(neuronCounts) {
        this.neuronCounts = [...neuronCounts];
        this.levels = [];
        for (let i = 0; i < neuronCounts.length - 1; i++) {
            this.levels.push(new Level(
                neuronCounts[i], neuronCounts[i + 1]
            ));
        }
    }

    static feedForward(givenInputs, network) {
        let outputs = Level.feedForward(givenInputs, network.levels[0]);
        for (let i = 1; i < network.levels.length; i++) {
            outputs = Level.feedForward(outputs, network.levels[i]);
        }
        return outputs;
    }

    // GA mutation operator: `rate` is the probability that an individual gene
    // changes at all, `strength` how far it may move. Nudging every gene at
    // once (the old behaviour) cannot preserve a good gene while exploring.
    // `mutatedOut`, when given, collects the index of every gene that actually
    // changed, so the breeding panel can show the real mutations rather than a
    // mock-up. Genes are indexed in one fixed order: per level, biases first,
    // then weights row by row — the same order crossover records.
    static mutate(network, rate = 0.1, strength = 0.3, mutatedOut = null) {
        let index = 0;
        for (const level of network.levels) {
            for (let i = 0; i < level.biases.length; i++) {
                if (Math.random() < rate) {
                    level.biases[i] = clamp(level.biases[i] + randomGene() * strength);
                    if (mutatedOut) {
                        mutatedOut.push(index);
                    }
                }
                index++;
            }
            for (let i = 0; i < level.weights.length; i++) {
                for (let j = 0; j < level.weights[i].length; j++) {
                    if (Math.random() < rate) {
                        level.weights[i][j] = clamp(level.weights[i][j] + randomGene() * strength);
                        if (mutatedOut) {
                            mutatedOut.push(index);
                        }
                    }
                    index++;
                }
            }
        }
    }

    // GA recombination operator: uniform crossover, each gene drawn from one
    // parent or the other with equal probability.
    // `originsOut`, when given, records 0 or 1 per gene for which parent it
    // came from, in the same fixed gene order mutate() uses.
    static crossover(parentA, parentB, originsOut = null) {
        const child = new NeuralNetwork(parentA.neuronCounts);
        for (let l = 0; l < child.levels.length; l++) {
            const a = parentA.levels[l];
            const b = parentB.levels[l];
            const c = child.levels[l];
            for (let i = 0; i < c.biases.length; i++) {
                const fromA = Math.random() < 0.5;
                c.biases[i] = fromA ? a.biases[i] : b.biases[i];
                if (originsOut) {
                    originsOut.push(fromA ? 0 : 1);
                }
            }
            for (let i = 0; i < c.weights.length; i++) {
                for (let j = 0; j < c.weights[i].length; j++) {
                    const fromA = Math.random() < 0.5;
                    c.weights[i][j] = fromA ? a.weights[i][j] : b.weights[i][j];
                    if (originsOut) {
                        originsOut.push(fromA ? 0 : 1);
                    }
                }
            }
        }
        return child;
    }

    static clone(network) {
        const copy = new NeuralNetwork(network.neuronCounts);
        for (let l = 0; l < copy.levels.length; l++) {
            copy.levels[l].biases = [...network.levels[l].biases];
            copy.levels[l].weights = network.levels[l].weights.map(row => [...row]);
        }
        return copy;
    }

    // A brain saved by an earlier build can have a shape the car's current
    // sensors no longer produce. Rebuilding through here rejects it outright
    // instead of silently feeding NaN forward and producing a dead car.
    static fromJSON(data, expectedCounts) {
        if (!data || !Array.isArray(data.neuronCounts) || !Array.isArray(data.levels)) {
            return null;
        }
        if (data.neuronCounts.length !== expectedCounts.length ||
            data.neuronCounts.some((count, i) => count !== expectedCounts[i])) {
            return null;
        }
        if (data.levels.length !== expectedCounts.length - 1) {
            return null;
        }

        const network = new NeuralNetwork(expectedCounts);
        for (let l = 0; l < network.levels.length; l++) {
            const level = network.levels[l];
            const saved = data.levels[l];
            if (!saved || !Array.isArray(saved.biases) || !Array.isArray(saved.weights)) {
                return null;
            }
            if (saved.biases.length !== level.biases.length ||
                saved.weights.length !== level.weights.length) {
                return null;
            }
            for (let i = 0; i < level.weights.length; i++) {
                if (!Array.isArray(saved.weights[i]) ||
                    saved.weights[i].length !== level.weights[i].length) {
                    return null;
                }
            }
            level.biases = saved.biases.map(value => clamp(Number(value) || 0));
            level.weights = level.weights.map(
                (row, i) => row.map((_, j) => clamp(Number(saved.weights[i][j]) || 0))
            );
        }
        return network;
    }
}

class Level {
    constructor(inputCount, outputCount) {
        this.inputs = new Array(inputCount).fill(0);
        this.outputs = new Array(outputCount).fill(0);
        this.biases = new Array(outputCount).fill(0);

        this.weights = [];
        for (let i = 0; i < inputCount; i++) {
            this.weights[i] = new Array(outputCount).fill(0);
        }

        Level.#randomize(this);
    }

    static #randomize(level) {
        for (let i = 0; i < level.inputs.length; i++) {
            for (let j = 0; j < level.outputs.length; j++) {
                level.weights[i][j] = randomGene();
            }
        }

        for (let i = 0; i < level.biases.length; i++) {
            level.biases[i] = randomGene();
        }
    }

    // tanh of (weighted sum + bias). The original compared the sum against
    // the bias and emitted 0/1, which threw away all magnitude between
    // layers and inverted the usual meaning of a bias.
    static feedForward(givenInputs, level) {
        for (let i = 0; i < level.inputs.length; i++) {
            const value = givenInputs[i];
            level.inputs[i] = Number.isFinite(value) ? value : 0;
        }

        for (let i = 0; i < level.outputs.length; i++) {
            let sum = 0;
            for (let j = 0; j < level.inputs.length; j++) {
                sum += level.inputs[j] * level.weights[j][i];
            }
            level.outputs[i] = Math.tanh(sum + level.biases[i]);
        }

        return level.outputs;
    }
}
