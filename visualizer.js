"use strict";

class Visualizer {
    // One label per network input, in order: the five sensor rays from left to
    // right, the car's own speed, the nearest light's colour as three one-hot
    // inputs, and how close that light is.
    // Read lazily: this file loads before car.js.
    static get INPUT_LABELS() {
        return Car.INPUT_LABELS;
    }

    static get OUTPUT_LABELS() {
        return Car.OUTPUT_LABELS;
    }

    // `decision`, when given, is the car's arbitrated action this frame, so
    // the output nodes can be marked with what the car actually did.
    static drawNetwork(ctx, network, decision = null, title = "Best car's brain") {
        if (!network || !network.levels.length) {
            return;
        }

        const margin = 50;
        const left = margin;
        const top = margin + 10;
        const width = ctx.canvas.width - margin * 2;
        const height = ctx.canvas.height - margin * 2 - 20;

        const levelHeight = height / network.levels.length;

        for (let i = network.levels.length - 1; i >= 0; i--) {
            const levelTop = top + lerp(
                height - levelHeight,
                0,
                network.levels.length == 1
                    ? 0.5
                    : i / (network.levels.length - 1)
            );

            ctx.setLineDash([7, 3]);
            Visualizer.drawLevel(
                ctx, network.levels[i],
                left, levelTop,
                width, levelHeight,
                i == network.levels.length - 1 ? Visualizer.OUTPUT_LABELS : [],
                i == 0 ? Visualizer.INPUT_LABELS : [],
                i == network.levels.length - 1 ? decision : null
            );
        }

        Visualizer.#drawKey(ctx, title);
    }

    // Without this the panel is a pretty picture nobody can read: the input
    // nodes need names and the line colours need explaining.
    static #drawKey(ctx, title) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.setLineDash([]);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "white";
        ctx.fillText(title, 10, 8);

        ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.fillText("outputs: accelerate, steer left, steer right, brake (green ring = active)", 10, 24);
        ctx.textAlign = "right";
        ctx.fillText("inputs: 5 rays, speed, red / yellow / green light, light distance", width - 10, height - 42);

        ctx.textAlign = "left";
        ctx.fillStyle = "#ffd21f";
        ctx.fillText("● positive weight", 10, height - 28);
        ctx.fillStyle = "#4da3ff";
        ctx.fillText("● negative weight", 10, height - 15);
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fillText("brightness = strength", 130, height - 15);
    }

    static drawLevel(ctx, level, left, top, width, height, outputLabels, inputLabels, decision) {
        const right = left + width;
        const bottom = top + height;

        const { inputs, outputs, weights, biases } = level;

        for (let i = 0; i < inputs.length; i++) {
            for (let j = 0; j < outputs.length; j++) {
                ctx.beginPath();
                ctx.moveTo(
                    Visualizer.#getNodeX(inputs, i, left, right),
                    bottom
                );
                ctx.lineTo(
                    Visualizer.#getNodeX(outputs, j, left, right),
                    top
                );
                ctx.lineWidth = 2;
                ctx.strokeStyle = getRGBA(weights[i][j]);
                ctx.stroke();
            }
        }

        const nodeRadius = inputs.length > 8 ? 15 : 18;
        for (let i = 0; i < inputs.length; i++) {
            const x = Visualizer.#getNodeX(inputs, i, left, right);
            ctx.beginPath();
            ctx.arc(x, bottom, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = "black";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, bottom, nodeRadius * 0.6, 0, Math.PI * 2);
            ctx.fillStyle = getRGBA(inputs[i]);
            ctx.fill();

            if (inputLabels[i]) {
                ctx.beginPath();
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "white";
                ctx.font = "bold 9px ui-sans-serif, system-ui, sans-serif";
                ctx.fillText(inputLabels[i], x, bottom + nodeRadius * 1.5);
            }
        }

        const activeFlags = decision
            ? [decision.accelerate, decision.left, decision.right, decision.brake]
            : null;

        for (let i = 0; i < outputs.length; i++) {
            const x = Visualizer.#getNodeX(outputs, i, left, right);
            ctx.beginPath();
            ctx.arc(x, top, 18, 0, Math.PI * 2);
            ctx.fillStyle = "black";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, top, 18 * 0.6, 0, Math.PI * 2);
            ctx.fillStyle = getRGBA(outputs[i]);
            ctx.fill();

            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.arc(x, top, 18 * 0.8, 0, Math.PI * 2);
            ctx.strokeStyle = getRGBA(biases[i]);
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);

            // The arbitrated decision: a raw output above zero is a request,
            // the green ring means the request was acted on.
            if (activeFlags && activeFlags[i]) {
                ctx.beginPath();
                ctx.lineWidth = 3;
                ctx.arc(x, top, 18 + 4, 0, Math.PI * 2);
                ctx.strokeStyle = "#8bf59b";
                ctx.stroke();
            }

            if (outputLabels[i]) {
                ctx.beginPath();
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "white";
                ctx.font = "bold 10px ui-sans-serif, system-ui, sans-serif";
                ctx.fillText(outputLabels[i], x, top - 18 - 12);
            }
        }
    }

    static #getNodeX(nodes, index, left, right) {
        return lerp(
            left,
            right,
            nodes.length == 1
                ? 0.5
                : index / (nodes.length - 1)
        );
    }
}
