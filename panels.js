"use strict";

// The panels that show the ALGORITHM rather than the car. Without these the
// genetic algorithm is invisible: fitness, ranking, selection, elitism and
// crossover all happen in the gap between one generation and the next, so a
// viewer only ever sees the result and has to take the method on trust.
class Panels {
    static #font(size, weight = "") {
        return `${weight} ${size}px ui-sans-serif, system-ui, sans-serif`.trim();
    }

    // Best, mean and median fitness, one point per generation. The best line
    // alone proves little — elitism makes it non-decreasing by construction —
    // so the mean and median are drawn with it: those only rise when the
    // whole population is getting better.
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
            high = Math.max(high, point.best, point.mean, point.median || 0);
            low = Math.min(low, point.mean, point.best, point.median || 0);
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

        // A stage change is a change of track, so fitness before and after it
        // is not comparable; mark where it happened.
        for (let i = 1; i < history.length; i++) {
            if (history[i].stage && history[i].stage !== history[i - 1].stage) {
                const x = Math.round(xAt(i)) + 0.5;
                ctx.strokeStyle = "rgba(255,159,67,0.7)";
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x, pad.top);
                ctx.lineTo(x, pad.top + plotHeight);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#ff9f43";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                ctx.fillText(`stage ${history[i].stage}`, x + 3, pad.top);
            }
        }

        const stroke = (key, color, lineWidth) => {
            ctx.beginPath();
            let started = false;
            history.forEach((point, index) => {
                if (!Number.isFinite(point[key])) {
                    return;
                }
                const x = xAt(index);
                const y = yAt(point[key]);
                if (!started) {
                    ctx.moveTo(x, y);
                    started = true;
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.stroke();
        };
        stroke("median", "#8bf59b", 1.2);
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
        ctx.fillText("● mean", pad.left + 42, height - 2);
        ctx.fillStyle = "#8bf59b";
        ctx.fillText("● median", pad.left + 86, height - 2);
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
        ctx.fillText(`4. ONE CHILD'S ${trace.origins.length} GENES`, left, top);
        let y = top + 16;

        const perRow = 31;
        const cell = 10;
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
        ctx.fillText(`breeding generation ${snapshot.generation + 1}` +
            (snapshot.stage ? `  ·  stage ${snapshot.stage}` : ""), left, y);
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
                `= ${row.distance} dist + ${row.carsPassed} passed + ${row.legalStops} stops + ${row.lightsCleared} lights` +
                (row.crashed ? " - crash" : ""),
                left + 70, y
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
            `...${snapshot.childCount} children, each parent the winner of a ` +
            `${snapshot.tournamentSize}-car tournament, ${snapshot.mutationPercent}% of genes mutated`,
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
            `mean ${snapshot.meanFitness}  |  median ${snapshot.medianFitness}  |  best ${snapshot.rows[0].fitness}` +
            `  |  diversity ${snapshot.diversity.toFixed(2)}`,
            left, y
        );
    }

    // Full-panel progress display while a benchmark runs.
    static drawProgress(ctx, title, progress, label, lines = []) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(8,12,24,0.96)";
        ctx.fillRect(0, 0, width, height);

        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "white";
        ctx.font = Panels.#font(15, "bold");
        ctx.fillText(title, 18, 24);

        const barY = 60;
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        ctx.fillRect(18, barY, width - 36, 14);
        ctx.fillStyle = "#8bf59b";
        ctx.fillRect(18, barY, (width - 36) * Math.max(0, Math.min(1, progress)), 14);

        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = Panels.#font(11);
        ctx.fillText(`${Math.round(progress * 100)}%  ${label}`, 18, barY + 22);

        let y = barY + 50;
        ctx.font = Panels.#font(10);
        for (const line of lines) {
            ctx.fillStyle = line.startsWith("#") ? "#ffd21f" : "rgba(255,255,255,0.8)";
            ctx.fillText(line.replace(/^#\s?/, ""), 18, y);
            y += 14;
        }
    }
}
