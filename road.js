"use strict";

class Road {
    constructor(x, width, laneCount = 3) {
        this.x = x;
        this.width = width;
        this.laneCount = laneCount;

        this.left = x - width / 2;
        this.right = x + width / 2;

        const infinity = 1000000;
        this.top = -infinity;
        this.bottom = infinity;

        const topLeft = { x: this.left, y: this.top };
        const topRight = { x: this.right, y: this.top };
        const bottomLeft = { x: this.left, y: this.bottom };
        const bottomRight = { x: this.right, y: this.bottom };
        this.borders = [
            [topLeft, bottomLeft],
            [topRight, bottomRight]
        ];
    }

    getLaneCenter(laneIndex) {
        const laneWidth = this.width / this.laneCount;
        const index = Math.max(0, Math.min(laneIndex, this.laneCount - 1));
        return this.left + laneWidth / 2 + index * laneWidth;
    }

    // Only the visible slice is stroked. Drawing the full 2,000,000px span
    // made the rasteriser walk ~50k dash segments per lane line every frame.
    // `from` snaps to the dash period so the dashes stay still as we scroll.
    draw(ctx, topY = this.top, bottomY = this.bottom) {
        const dashPeriod = 40;
        const from = Math.max(this.top, Math.floor(topY / dashPeriod) * dashPeriod - dashPeriod);
        const to = Math.min(this.bottom, bottomY + dashPeriod);

        ctx.lineWidth = 5;
        ctx.strokeStyle = "white";

        for (let i = 1; i <= this.laneCount - 1; i++) {
            const x = lerp(this.left, this.right, i / this.laneCount);

            ctx.setLineDash([20, 20]);
            ctx.beginPath();
            ctx.moveTo(x, from);
            ctx.lineTo(x, to);
            ctx.stroke();
        }

        ctx.setLineDash([]);
        for (const border of this.borders) {
            ctx.beginPath();
            ctx.moveTo(border[0].x, from);
            ctx.lineTo(border[1].x, to);
            ctx.stroke();
        }
    }
}
