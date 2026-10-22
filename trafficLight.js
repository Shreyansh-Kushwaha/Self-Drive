"use strict";

const LIGHT_GREEN_MS = 9000;
const LIGHT_YELLOW_MS = 2200;
const LIGHT_RED_MS = 7000;
const LIGHT_CYCLE_MS = LIGHT_GREEN_MS + LIGHT_YELLOW_MS + LIGHT_RED_MS;

const DEFAULT_LIGHT_DURATIONS = Object.freeze({
    green: LIGHT_GREEN_MS,
    yellow: LIGHT_YELLOW_MS,
    red: LIGHT_RED_MS
});

class TrafficLight {
    static #nextId = 1;
    static #sprite = null;

    // Each light carries its own phase, so they no longer all switch together
    // off one global clock, and the yellow phase gives a car long enough to
    // brake instead of being killed the instant the light flips to red.
    // `durations` lets a scenario (the demo route) hold a light red for
    // longer than the training lights do.
    constructor(x, y, width, height, road,
        phaseOffsetMs = Math.random() * LIGHT_CYCLE_MS,
        durations = DEFAULT_LIGHT_DURATIONS) {
        this.id = TrafficLight.#nextId++;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.phaseOffsetMs = phaseOffsetMs;
        this.durations = { ...DEFAULT_LIGHT_DURATIONS, ...durations };
        this.cycleMs = this.durations.green + this.durations.yellow + this.durations.red;
        this.state = "green";
        this.stopLine = new TrafficLightSensor(this, road);
    }

    static get sprite() {
        if (!TrafficLight.#sprite) {
            TrafficLight.#sprite = new Image();
            TrafficLight.#sprite.src = "trafficLight.jpeg";
        }
        return TrafficLight.#sprite;
    }

    get isRed() {
        return this.state === "red";
    }

    get isGreen() {
        return this.state === "green";
    }

    get isStopping() {
        return this.state !== "green";
    }

    // What the light will show `elapsedMs` into a run, without changing it.
    stateAt(elapsedMs) {
        const phase = (elapsedMs + this.phaseOffsetMs) % this.cycleMs;
        if (phase < this.durations.green) {
            return "green";
        }
        if (phase < this.durations.green + this.durations.yellow) {
            return "yellow";
        }
        return "red";
    }

    update(elapsedMs) {
        this.state = this.stateAt(elapsedMs);
    }

    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.drawImage(
            TrafficLight.sprite,
            -this.width / 2,
            -this.height / 2,
            this.width,
            this.height
        );
        ctx.restore();

        // The jpeg is a single static image, so the live state is shown by a
        // lamp drawn on top of it rather than by swapping the picture.
        ctx.beginPath();
        ctx.arc(this.x, this.y - this.height / 2 - 9, 6, 0, Math.PI * 2);
        ctx.fillStyle = this.state === "green"
            ? "#1fdd55"
            : this.state === "yellow" ? "#ffd21f" : "#ff2d2d";
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "black";
        ctx.stroke();

        this.stopLine.draw(ctx);
    }
}
