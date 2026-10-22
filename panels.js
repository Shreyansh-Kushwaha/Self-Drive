"use strict";

// The two panels that show the ALGORITHM rather than the car. Without these
// the genetic algorithm is invisible: fitness, ranking, selection, elitism and
// crossover all happen in the gap between one generation and the next, so a
// viewer only ever sees the result and has to take the method on trust.
class Panels {
    static #font(size, weight = "") {
        return `${weight} ${size}px ui-sans-serif, system-ui, sans-serif`.trim();
    }

    // Best and mean fitness, one point per generation. This is the single
    // picture that shows learning is actually happening.
    static drawFitnessHistory(ctx, history) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.setLineDash([]);
        ctx.fillStyle = "#101828";
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = "white";
        ctx.font = Panels.#font(12, "bold");
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("Fitness by generation", 8, 6);

        const pad = { left: 40, right: 10, top: 24, bottom: 26 };
        const plotWidth = width - pad.left - pad.right;
        const plotHeight = height - pad.top - pad.bottom;

        if (history.length === 0) {
            ctx.fillStyle = "rgba(255,255,255,0.55)";
            ctx.font = Panels.#font(11);
            ctx.fillText("generation 1 running...", pad.left, pad.top + plotHeight / 2);
            return;
        }

        let high = 1;
        let low = 0;
        for (const point of history) {
            high = Math.max(high, point.best, point.mean);
            low = Math.min(low, point.mean, point.best);
        }
        const span = high - low || 1;

        const xAt = (index) => (history.length === 1
            ? pad.left + plotWidth / 2
            : pad.left + (plotWidth * index) / (history.length - 1));
        const yAt = (value) => pad.top + plotHeight - (plotHeight * (value - low)) / span;

        ctx.strokeStyle = "rgba(255,255,255,0.14)";
        ctx.lineWidth = 1;
        ctx.font = Panels.#font(9);
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i <= 2; i++) {
            const value = low + (span * i) / 2;
            const y = Math.round(yAt(value)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(width - pad.right, y);
            ctx.stroke();
            ctx.fillStyle = "rgba(255,255,255,0.6)";
            ctx.fillText(String(Math.round(value)), pad.left - 5, y);
        }

        const stroke = (key, color, lineWidth) => {
            ctx.beginPath();
            history.forEach((point, index) => {
                const x = xAt(index);
                const y = yAt(point[key]);
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.stroke();
        };
        stroke("mean", "#7aa7ff", 1.5);
        stroke("best", "#ffd21f", 2);

        const last = history[history.length - 1];
        ctx.beginPath();
        ctx.arc(xAt(history.length - 1), yAt(last.best), 3, 0, Math.PI * 2);
        ctx.fillStyle = "#ffd21f";
        ctx.fill();

        ctx.textBaseline = "bottom";
        ctx.textAlign = "left";
        ctx.font = Panels.#font(9);
        ctx.fillStyle = "rgba(255,255,255,0.6)";
        ctx.fillText(`gen ${history[0].generation}`, pad.left, height - 14);
        ctx.textAlign = "right";
        ctx.fillText(`gen ${last.generation}`, width - pad.right, height - 14);

        ctx.textAlign = "left";
        ctx.fillStyle = "#ffd21f";
        ctx.fillText("● best", pad.left, height - 2);
        ctx.fillStyle = "#7aa7ff";
        ctx.fillText("● population mean", pad.left + 52, height - 2);
    }

    // The whole genome of one real child, one cell per gene, coloured by which
    // parent it came from. Crossover is the hardest operator to picture, and
    // this makes "each gene comes from one parent or the other" literal.
    static #drawGeneStrip(ctx, snapshot, left, top) {
        const trace = snapshot.geneTrace;
        if (!trace || !trace.origins.length) {
            return top;
        }

        ctx.fillStyle = "#ffd21f";
        ctx.font = Panels.#font(11, "bold");
        ctx.fillText("4. ONE CHILD'S 100 GENES", left, top);
        let y = top + 16;

        const perRow = 20;
        const cell = 13;
        const gap = 2;
        const mutated = new Set(trace.mutated || []);

        trace.origins.forEach((origin, index) => {
            const column = index % perRow;
            const row = Math.floor(index / perRow);
            const x = left + 2 + column * (cell + gap);
            const cellY = y + row * (cell + gap);

            ctx.fillStyle = origin === 0 ? "#ffd21f" : "#7aa7ff";
            ctx.fillRect(x, cellY, cell, cell);

            if (mutated.has(index)) {
                ctx.strokeStyle = "#ff4d4d";
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 1, cellY + 1, cell - 2, cell - 2);
            }
        });

        y += Math.ceil(trace.origins.length / perRow) * (cell + gap) + 8;

        ctx.font = Panels.#font(9);
        ctx.fillStyle = "#ffd21f";
        ctx.fillText("■ parent A", left + 2, y);
        ctx.fillStyle = "#7aa7ff";
        ctx.fillText("■ parent B", left + 64, y);
        ctx.fillStyle = "#ff4d4d";
        ctx.fillText(`□ mutated (${mutated.size})`, left + 126, y);

        return y + 22;
    }

    // Shown for a couple of seconds between generations: the ranking, which
    // brains survive untouched, and which pairs were actually chosen to breed.
    static drawBreeding(ctx, snapshot) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(8,12,24,0.94)";
        ctx.fillRect(0, 0, width, height);

        let y = 24;
        const left = 18;

        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "white";
        ctx.font = Panels.#font(15, "bold");
        ctx.fillText(`Generation ${snapshot.generation} finished`, left, y);
        y += 20;
        ctx.fillStyle = "rgba(255,255,255,0.65)";
        ctx.font = Panels.#font(11);
        ctx.fillText(`breeding generation ${snapshot.generation + 1}`, left, y);
        y += 26;

        const heading = (text) => {
            ctx.fillStyle = "#ffd21f";
            ctx.font = Panels.#font(11, "bold");
            ctx.fillText(text, left, y);
            y += 16;
        };

        heading("1. SCORE AND RANK BY FITNESS");
        ctx.font = Panels.#font(10);
        for (const row of snapshot.rows) {
            const elite = row.rank <= snapshot.eliteCount;
            ctx.fillStyle = elite ? "#8bf59b" : "rgba(255,255,255,0.8)";
            ctx.fillText(`#${row.rank}`, left + 2, y);
            ctx.fillText(String(row.fitness), left + 26, y);
            ctx.fillStyle = "rgba(255,255,255,0.45)";
            ctx.fillText(
                `= ${row.distance} dist  + ${row.carsPassed} passed  + ${row.lightsCleared} lights` +
                (row.crashed ? "  - crash" : ""),
                left + 78, y
            );
            if (elite) {
                ctx.fillStyle = "#8bf59b";
                ctx.fillText("ELITE", width - 48, y);
            }
            y += 14;
        }
        y += 10;

        heading("2. ELITISM");
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.font = Panels.#font(10);
        ctx.fillText(
            `top ${snapshot.eliteCount} copied into the next generation unchanged`,
            left + 2, y
        );
        y += 24;

        heading("3. TOURNAMENT SELECTION + CROSSOVER + MUTATION");
        ctx.font = Panels.#font(10);
        for (const pair of snapshot.pairs) {
            ctx.fillStyle = "rgba(255,255,255,0.8)";
            ctx.fillText(
                `#${pair.aRank} (${pair.aFitness})  x  #${pair.bRank} (${pair.bFitness})`,
                left + 2, y
            );
            ctx.fillStyle = "rgba(255,255,255,0.45)";
            ctx.fillText("-> child", left + 150, y);
            y += 14;
        }
        y += 6;
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.font = Panels.#font(10);
        ctx.fillText(
            `...${snapshot.childCount} children, each parent pair the winner of a ` +
            `${snapshot.tournamentSize}-car tournament`,
            left + 2, y
        );
        y += 22;

        if (snapshot.immigrants > 0) {
            ctx.fillStyle = "#ff9f43";
            ctx.font = Panels.#font(10, "bold");
            ctx.fillText(
                `STAGNATION ESCAPE: best fitness flat for ${snapshot.stagnantGenerations} generations`,
                left + 2, y
            );
            y += 13;
            ctx.font = Panels.#font(10);
            ctx.fillText(
                `-> ${snapshot.immigrants} random immigrants, mutation raised to ${snapshot.mutationPercent}%`,
                left + 2, y
            );
            y += 20;
        }

        y = Panels.#drawGeneStrip(ctx, snapshot, left, y);

        ctx.fillStyle = "rgba(255,255,255,0.75)";
        ctx.font = Panels.#font(11, "bold");
        ctx.fillText(
            `population mean fitness ${snapshot.meanFitness}  |  best ${snapshot.rows[0].fitness}`,
            left, y
        );
    }
}
