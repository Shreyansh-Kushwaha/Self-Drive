"use strict";

const STOP_LINE_THICKNESS = 4;

// The stop line belonging to one traffic light. It is created only by the
// TrafficLight that owns it, and its polygon is exactly the strip that gets
// drawn, so a car is never destroyed outside the bar it can actually see.
// (The old version built a 500px-wide bar from a 188px-wide drawn line and
// expressed "green" as a zero-area polygon.)
class TrafficLightSensor {
    constructor(trafficLight, road) {
        this.trafficLight = trafficLight;
        this.left = road.left;
        this.right = road.right;
        this.y = trafficLight.y;

        const half = STOP_LINE_THICKNESS / 2;
        this.polygon = [
            { x: this.left, y: this.y - half },
            { x: this.right, y: this.y - half },
            { x: this.right, y: this.y + half },
            { x: this.left, y: this.y + half }
        ];
    }

    draw(ctx) {
        if (!this.trafficLight.isStopping) {
            return;
        }

        ctx.save();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.lineWidth = STOP_LINE_THICKNESS;
        ctx.strokeStyle = this.trafficLight.isRed ? "red" : "rgba(255,210,31,0.9)";
        ctx.moveTo(this.left, this.y);
        ctx.lineTo(this.right, this.y);
        ctx.stroke();
        ctx.restore();
    }
}
