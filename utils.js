"use strict";

function lerp(A, B, t) {
    return A + (B - A) * t;
}

function clamp(value, min = -1, max = 1) {
    return Math.min(max, Math.max(min, value));
}

function randomGene() {
    return Math.random() * 2 - 1;
}

// mulberry32: a small seeded generator with the same [0, 1) contract as
// Math.random. The track is built from one of these so every generation
// drives the identical course and a brain's fitness is a property of the
// brain, not of the dice it happened to roll that run.
function createSeededRandom(seed) {
    let state = seed >>> 0;
    return function random() {
        state = (state + 0x6D2B79F5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function shuffled(array, random = Math.random) {
    const copy = [...array];
    for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
}

function getIntersection(A, B, C, D) {
    const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
    const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
    const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);

    if (bottom != 0) {
        const t = tTop / bottom;
        const u = uTop / bottom;
        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            return {
                x: lerp(A.x, B.x, t),
                y: lerp(A.y, B.y, t),
                offset: t
            };
        }
    }

    return null;
}

function polysIntersect(poly1, poly2) {
    for (let i = 0; i < poly1.length; i++) {
        for (let j = 0; j < poly2.length; j++) {
            const touch = getIntersection(
                poly1[i],
                poly1[(i + 1) % poly1.length],
                poly2[j],
                poly2[(j + 1) % poly2.length]
            );
            if (touch) {
                return true;
            }
        }
    }
    return false;
}

function getRGBA(value) {
    const safe = Number.isFinite(value) ? clamp(value) : 0;
    const alpha = Math.abs(safe);
    const R = safe < 0 ? 0 : 255;
    const G = R;
    const B = safe > 0 ? 0 : 255;
    return "rgba(" + R + "," + G + "," + B + "," + alpha + ")";
}

function getRandomColor(random = Math.random) {
    const hue = 290 + random() * 260;
    return "hsl(" + hue + ", 100%, 60%)";
}
