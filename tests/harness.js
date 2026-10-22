"use strict";

// Shared test harness. The game files are plain browser scripts, so they are
// loaded into a vm context with just enough DOM and canvas stubbed out to let
// them run under Node. Math.random is replaced with a seeded generator so an
// evolutionary run is reproducible instead of flaky.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PROJECT_DIR = path.join(__dirname, "..");

// Same order as index.html. HEADLESS_FILES is everything except the page.
const HEADLESS_FILES = [
    "utils.js", "network.js", "controls.js", "road.js", "trafficLightSensor.js",
    "trafficLight.js", "sensor.js", "car.js", "visualizer.js", "panels.js",
    "world.js", "ga.js", "evaluate.js"
];
const GAME_FILES = [...HEADLESS_FILES, "main.js"];

function makeContext2d(canvas) {
    return new Proxy({ canvas }, {
        get: (target, prop) => (prop in target ? target[prop] : () => {}),
        set: (target, prop, value) => { target[prop] = value; return true; }
    });
}

function makeCanvas(width = 300, height = 300) {
    const canvas = { width, height, addEventListener() {} };
    let context = null;
    canvas.getContext = () => (context || (context = makeContext2d(canvas)));
    return canvas;
}

function mulberry32(seed) {
    let a = seed >>> 0;
    return function random() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// `files` lets a test load only part of the project (operator tests do not
// need main.js). `overrides` rewrites top-level constants so a test can run
// shorter generations than the real game uses.
function loadGame({ files = GAME_FILES, seed = 1, overrides = {} } = {}) {
    const elements = {
        carCanvas: makeCanvas(),
        networkCanvas: makeCanvas(),
        chartCanvas: makeCanvas()
    };
    const element = (name) => {
        if (!elements[name]) {
            elements[name] = {
                textContent: "",
                handlers: [],
                dataset: {},
                classList: { toggle() {}, add() {}, remove() {} },
                addEventListener(event, fn) { this.handlers.push(fn); }
            };
        }
        return elements[name];
    };

    const store = new Map();
    let pendingFrame = null;

    const sandbox = {
        console,
        setTimeout,
        clearTimeout,
        performance,
        Image: class {
            constructor() {
                this.src = "";
                this.complete = true;
                this.naturalWidth = 32;
                this.naturalHeight = 32;
            }
            addEventListener() {}
        },
        localStorage: {
            getItem: (key) => (store.has(key) ? store.get(key) : null),
            setItem: (key, value) => store.set(key, String(value)),
            removeItem: (key) => store.delete(key)
        },
        document: {
            getElementById: (id) => elements[id] || element(id),
            querySelector: (selector) => element(selector.replace("#", "")),
            // The speed buttons only exist in the browser; the tests drive the
            // simulation through runFrames instead of clicking them.
            querySelectorAll: () => [],
            createElement: () => makeCanvas(30, 50),
            addEventListener() {}
        },
        window: { innerHeight: 800, addEventListener() {} },
        requestAnimationFrame: (callback) => { pendingFrame = callback; }
    };

    const context = vm.createContext(sandbox);
    vm.runInContext(`Math.random = ${mulberry32.toString()}(${seed});`, context);

    for (const file of files) {
        let source = fs.readFileSync(path.join(PROJECT_DIR, file), "utf8");
        for (const [name, value] of Object.entries(overrides)) {
            const pattern = new RegExp(`const ${name} = [^;]+;`);
            if (pattern.test(source)) {
                source = source.replace(pattern, `const ${name} = ${value};`);
            }
        }
        vm.runInContext(source, context, { filename: file });
    }

    const read = (expression) => vm.runInContext(expression, context);

    // Drives the real requestAnimationFrame loop with synthetic timestamps.
    function runFrames(count, onFrame) {
        let elapsed = read("typeof __testElapsed === 'number' ? __testElapsed : 0");
        for (let i = 0; i < count; i++) {
            const callback = pendingFrame;
            pendingFrame = null;
            if (!callback) {
                throw new Error(`animation loop stopped after ${i} frames`);
            }
            elapsed += 1000 / 60;
            callback(elapsed);
            if (onFrame) {
                onFrame(i);
            }
        }
    }

    return { read, runFrames, store, elements };
}

function createReporter(title) {
    let passed = 0;
    let failed = 0;
    console.log(title);
    return {
        section(name) {
            console.log(`\n${name}`);
        },
        check(name, condition, detail = "") {
            if (condition) {
                passed++;
                console.log(`  PASS  ${name}`);
            } else {
                failed++;
                console.log(`  FAIL  ${name} ${detail}`);
            }
        },
        finish() {
            console.log(`\n${passed} passed, ${failed} failed`);
            if (failed > 0) {
                process.exitCode = 1;
            }
        }
    };
}

module.exports = { loadGame, createReporter, GAME_FILES, HEADLESS_FILES, PROJECT_DIR };
