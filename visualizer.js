"use strict";

class Visualizer {
    // One label per network input, in order: the five sensor rays from left to
    // right, then the car's own speed, then the red-light signal.
    static INPUT_LABELS = ["L45", "L22", "FWD", "R22", "R45", "SPD", "RED"];

    static drawNetwork(ctx, network) {
        if (!network || !network.levels.length) {
            return;
        }

        const margin = 50;
        const left = margin;
        const top = margin;
        const width = ctx.canvas.width - margin * 2;
        const height = ctx.canvas.height - margin * 2;

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
                i == network.levels.length - 1 ? ["↑", "←", "→", "↓"] : [],
                i == 0 ? Visualizer.INPUT_LABELS : []
            );
        }

        Visualizer.#drawKey(ctx);
    }

    // Without this the panel is a pretty picture nobody can read: the input
    // nodes need names and the line colours need explaining.
    static #drawKey(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.setLineDash([]);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "white";
        ctx.fillText("Best car's brain", 10, 8);

        ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.fillText("outputs (steering)", 10, 24);
        ctx.textAlign = "right";
        ctx.fillText("inputs (what it senses)", width - 10, height - 42);

        ctx.textAlign = "left";
        ctx.fillStyle = "#ffd21f";
        ctx.fillText("● positive weight", 10, height - 28);
        ctx.fillStyle = "#4da3ff";
        ctx.fillText("● negative weight", 10, height - 15);
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fillText("brightness = strength", 130, height - 15);
    }

    static drawLevel(ctx, level, left, top, width, height, outputLabels, inputLabels) {
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

        const nodeRadius = 18;
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
                ctx.font = "bold 10px ui-sans-serif, system-ui, sans-serif";
                ctx.fillText(inputLabels[i], x, bottom + nodeRadius * 1.5);
            }
        }

        for (let i = 0; i < outputs.length; i++) {
            const x = Visualizer.#getNodeX(outputs, i, left, right);
            ctx.beginPath();
            ctx.arc(x, top, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = "black";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, top, nodeRadius * 0.6, 0, Math.PI * 2);
            ctx.fillStyle = getRGBA(outputs[i]);
            ctx.fill();

            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.arc(x, top, nodeRadius * 0.8, 0, Math.PI * 2);
            ctx.strokeStyle = getRGBA(biases[i]);
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);

            if (outputLabels[i]) {
                ctx.beginPath();
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "black";
                ctx.strokeStyle = "white";
                ctx.font = (nodeRadius * 1.5) + "px Arial";
                ctx.fillText(outputLabels[i], x, top + nodeRadius * 0.1);
                ctx.lineWidth = 0.5;
                ctx.strokeText(outputLabels[i], x, top + nodeRadius * 0.1);
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
