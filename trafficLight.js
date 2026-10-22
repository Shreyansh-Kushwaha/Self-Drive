"use strict";

const LIGHT_GREEN_MS = 9000;
const LIGHT_YELLOW_MS = 2200;
const LIGHT_RED_MS = 7000;
const LIGHT_CYCLE_MS = LIGHT_GREEN_MS + LIGHT_YELLOW_MS + LIGHT_RED_MS;

class TrafficLight {
    static #nextId = 1;
    static #sprite = null;

    // Each light carries its own phase, so they no longer all switch together
    // off one global clock, and the yellow phase gives a car long enough to
    // brake instead of being killed the instant the light flips to red.
    constructor(x, y, width, height, road, phaseOffsetMs = Math.random() * LIGHT_CYCLE_MS) {
        this.id = TrafficLight.#nextId++;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.phaseOffsetMs = phaseOffsetMs;
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

    get isStopping() {
        return this.state !== "green";
    }

    update(elapsedMs) {
        const phase = (elapsedMs + this.phaseOffsetMs) % LIGHT_CYCLE_MS;
        if (phase < LIGHT_GREEN_MS) {
            this.state = "green";
        } else if (phase < LIGHT_GREEN_MS + LIGHT_YELLOW_MS) {
            this.state = "yellow";
        } else {
            this.state = "red";
        }
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
