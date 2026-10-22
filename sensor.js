"use strict";

class Sensor {
    static RAY_COUNT = 5;

    // Reaction window = rayLength / closing speed. At 150px against traffic
    // closing at ~2px/frame the car saw a wall ~65 frames out, but a lane
    // change at full speed takes ~76 frames, so it was physically unable to
    // react in time no matter what its brain decided.
    static RAY_LENGTH = 280;

    constructor(car) {
        this.car = car;
        this.rayCount = Sensor.RAY_COUNT;
        this.rayLength = Sensor.RAY_LENGTH;
        this.raySpread = Math.PI / 2;
        this.rays = [];
        this.readings = [];
    }

    update(world) {
        this.#castRays();

        // Anything further away than a ray can reach cannot be hit, so it is
        // dropped before the per-ray loop rather than tested 5 more times.
        const reach = this.rayLength + 60;
        const obstacles = [];
        for (const other of world.traffic) {
            if (other === this.car || !other.polygon) {
                continue;
            }
            if (Math.abs(other.y - this.car.y) <= reach && Math.abs(other.x - this.car.x) <= reach) {
                obstacles.push(other.polygon);
            }
        }
        for (const line of world.stopLines) {
            if (Math.abs(line[0].y - this.car.y) <= reach) {
                obstacles.push(line);
            }
        }

        this.readings = [];
        for (let i = 0; i < this.rays.length; i++) {
            this.readings.push(this.#getReading(this.rays[i], world.borders, obstacles));
        }
    }

    #getReading(ray, roadBorders, obstacles) {
        let nearest = null;

        for (let i = 0; i < roadBorders.length; i++) {
            const touch = getIntersection(
                ray[0], ray[1], roadBorders[i][0], roadBorders[i][1]
            );
            if (touch && (nearest === null || touch.offset < nearest.offset)) {
                nearest = touch;
            }
        }

        for (const poly of obstacles) {
            for (let j = 0; j < poly.length; j++) {
                const touch = getIntersection(
                    ray[0], ray[1], poly[j], poly[(j + 1) % poly.length]
                );
                if (touch && (nearest === null || touch.offset < nearest.offset)) {
                    nearest = touch;
                }
            }
        }

        return nearest;
    }

    #castRays() {
        this.rays = [];
        for (let i = 0; i < this.rayCount; i++) {
            const rayAngle = lerp(
                this.raySpread / 2,
                -this.raySpread / 2,
                this.rayCount == 1 ? 0.5 : i / (this.rayCount - 1)
            ) + this.car.angle;

            const start = { x: this.car.x, y: this.car.y };
            const end = {
                x: this.car.x - Math.sin(rayAngle) * this.rayLength,
                y: this.car.y - Math.cos(rayAngle) * this.rayLength
            };
            this.rays.push([start, end]);
        }
    }

    draw(ctx) {
        for (let i = 0; i < this.rays.length; i++) {
            let end = this.rays[i][1];
            if (this.readings[i]) {
                end = this.readings[i];
            }

            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "yellow";
            ctx.moveTo(this.rays[i][0].x, this.rays[i][0].y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();

            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "black";
            ctx.moveTo(this.rays[i][1].x, this.rays[i][1].y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
        }
    }
}
