"use strict";

const CAR_SPRITE_SRC = "images.png";

// How far ahead a traffic light is reported to the network, and the range of
// the proximity input (1 at the line, 0 at LIGHT_LOOKAHEAD and beyond).
const LIGHT_LOOKAHEAD = 260;

// How close a dummy gets to the car ahead in its lane before lifting off.
// Coasting from 1.2 with friction 0.05 takes ~14px to stop, so it settles
// ~80px behind: clear of a 50px car, never driving through it.
const DUMMY_FOLLOW_DISTANCE = 95;

// A legal stop is counted when the car's front bumper is at rest within this
// many pixels before a red light's stop line. Stopping 300px short is safe
// but is not "stopping at the light"; it is simply not rewarded.
const STOP_ZONE = 120;
const STOP_SPEED = 0.1;

// Braking scrubs this much speed per frame. From top speed (3) that is a
// 10-frame, ~15px stop, so a car that decides to stop inside the 260px
// lookahead always can.
const BRAKE_FORCE = 0.3;

// Fitness shaping: distance is the main reward, crashing costs a fixed
// amount so a car that crawls safely can still beat one that sprints into a
// wall, and clearing a light pays for the patience it takes to wait for green.
//
// Overtaking is rewarded explicitly because distance alone is not enough to
// find it. On distance alone the population reliably evolved to drive up to
// the first wall of traffic and then tailgate it forever at the wall's own
// speed: completely safe, and worth more than any attempt to pass that ends
// in a crash. Paying for each car passed puts a gradient into that valley.
// The crash penalty is deliberately small. Crashing already costs a car every
// metre it would have driven afterwards, so a large extra penalty just
// double-counts it — and at 300 it priced every steering experiment out of
// the search, which is what kept the population parked behind the first wall.
//
// A legal stop is paid once per light, the moment the car comes to rest in the
// stop zone while the light is red. It is paid once, so sitting there earns
// nothing further; the LIGHT_BONUS for then clearing the line on green, plus
// all the distance after it, is what makes resuming worth more than waiting.
const CRASH_PENALTY = 100;
const RED_LIGHT_PENALTY = 50;
const LIGHT_BONUS = 150;
const OVERTAKE_BONUS = 120;
const LEGAL_STOP_BONUS = 100;

class Car {
    static #sprite = null;
    static #masks = new Map();

    // Network inputs, in order. Five rays, own speed, the colour of the
    // nearest light ahead as three one-hot inputs, and how close it is. With
    // no light inside LIGHT_LOOKAHEAD all four light inputs are 0.
    static INPUT_LABELS = ["L45", "L22", "FWD", "R22", "R45", "SPD", "RED", "YEL", "GRN", "DIST"];
    static OUTPUT_LABELS = ["ACC", "LEFT", "RIGHT", "BRAKE"];

    #passedLights = new Set();
    #passedCars = new Set();
    #metLights = new Set();
    #redLights = new Set();
    #stoppedLights = new Set();

    static #getSprite() {
        if (!Car.#sprite) {
            Car.#sprite = new Image();
            Car.#sprite.src = CAR_SPRITE_SRC;
        }
        return Car.#sprite;
    }

    // One tinted sprite per colour, shared by every car that uses it. The
    // original gave all 100 cars their own Image plus their own offscreen
    // canvas, rebuilt from scratch on every reload.
    static #getMask(color, width, height) {
        const key = color + ":" + width + "x" + height;
        const cached = Car.#masks.get(key);
        if (cached) {
            return cached;
        }

        const mask = document.createElement("canvas");
        mask.width = width;
        mask.height = height;
        const maskCtx = mask.getContext("2d");
        const sprite = Car.#getSprite();

        const paint = () => {
            maskCtx.fillStyle = color;
            maskCtx.beginPath();
            maskCtx.rect(0, 0, width, height);
            maskCtx.fill();
            maskCtx.globalCompositeOperation = "destination-atop";
            maskCtx.drawImage(sprite, 0, 0, width, height);
        };

        if (sprite.complete && sprite.naturalWidth > 0) {
            paint();
        } else {
            sprite.addEventListener("load", paint, { once: true });
        }

        Car.#masks.set(key, mask);
        return mask;
    }

    static brainShape(rayCount) {
        return [rayCount + 5, 8, 4];
    }

    constructor(x, y, width, height, controlType, maxSpeed = 3, color = "blue") {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;

        this.startY = y;
        this.speed = 0;
        this.acceleration = 0.2;
        this.maxSpeed = maxSpeed;
        this.friction = 0.05;
        this.angle = 0;
        this.damaged = false;
        this.crashCause = null;
        // A stalled car is retired, not wrecked: it keeps its score and stops
        // consuming simulation time. A car that reaches the end of a route
        // with a finish line is marked finished.
        this.retired = false;
        this.finished = false;

        this.useBrain = controlType == "AI";
        this.maxDistance = 0;
        this.framesAlive = 0;
        // Progress bookkeeping for the stall rule (see World.step).
        this.checkpointDistance = 0;
        this.checkpointFrame = 0;
        this.waitedSinceCheckpoint = false;
        this.lightsCleared = 0;
        this.carsPassed = 0;
        this.legalStops = 0;
        this.lightsMet = 0;
        this.redEncounters = 0;
        this.redLightViolation = false;

        // What the car knows about the nearest light ahead, refreshed each
        // frame. lightSignal is the old combined "red or yellow, how close"
        // value; the dummies and the stall timer still use it.
        this.lightState = null;
        this.lightDistance = Infinity;
        this.lightSignal = 0;

        // The decision after arbitration, so the interface can show what the
        // car actually did rather than the four raw network outputs.
        this.decision = { accelerate: false, brake: false, left: false, right: false };
        this.inputs = [];

        // An optional hand-written controller: inputs -> four outputs. Used
        // only by the benchmark's "hand-coded baseline"; a car with a driver
        // is not driven by its brain at all.
        this.driver = null;

        if (controlType != "DUMMY") {
            this.sensor = new Sensor(this);
            this.brain = new NeuralNetwork(Car.brainShape(this.sensor.rayCount));
        }
        this.controls = new Controls(controlType);
        this.polygon = this.#createPolygon();
    }

    get active() {
        return !this.damaged && !this.retired && !this.finished;
    }

    get fitness() {
        let score = this.maxDistance +
            this.lightsCleared * LIGHT_BONUS +
            this.carsPassed * OVERTAKE_BONUS +
            this.legalStops * LEGAL_STOP_BONUS;
        if (this.damaged) {
            score -= CRASH_PENALTY;
        }
        if (this.redLightViolation) {
            score -= RED_LIGHT_PENALTY;
        }
        return score;
    }

    // The terms of the fitness sum, for the interface and the breeding panel.
    get fitnessBreakdown() {
        return {
            distance: Math.round(this.maxDistance),
            carsPassed: this.carsPassed,
            legalStops: this.legalStops,
            lightsCleared: this.lightsCleared,
            crashed: this.damaged,
            redLightViolation: this.redLightViolation,
            fitness: Math.round(this.fitness)
        };
    }

    // Short label for what the car is doing this frame.
    get actionLabel() {
        const d = this.decision;
        const parts = [];
        parts.push(d.brake ? "BRAKE" : d.accelerate ? "ACCELERATE" : "COAST");
        if (d.left) {
            parts.push("STEER LEFT");
        }
        if (d.right) {
            parts.push("STEER RIGHT");
        }
        return parts.join(" + ");
    }

    update(world) {
        if (!this.active) {
            return;
        }

        this.#senseLight(world.lights);
        if (!this.useBrain && !this.driver) {
            this.#driveDummy(world);
        }

        this.#move();
        this.polygon = this.#createPolygon();
        this.framesAlive++;
        this.maxDistance = Math.max(this.maxDistance, this.startY - this.y);
        if (this.lightSignal > 0 && this.speed < STOP_SPEED) {
            // Waiting at a red light is not being stuck; the progress check
            // in World.step skips a window in which the car waited.
            this.waitedSinceCheckpoint = true;
        }

        this.#trackLights(world.lights);
        this.#trackLegalStop(world.lights);
        this.#trackOvertakes(world.traffic);
        this.damaged = this.#assessDamage(world);

        // A wrecked car used to keep raycasting and feeding forward forever;
        // with 100 cars that was most of the frame budget spent on corpses.
        if (this.damaged || !this.sensor) {
            return;
        }

        this.sensor.update(world);
        this.inputs = this.#buildInputs();

        const outputs = this.driver
            ? this.driver(this.inputs, this)
            : NeuralNetwork.feedForward(this.inputs, this.brain);
        if (this.useBrain || this.driver) {
            this.controls.forward = outputs[0] > 0;
            this.controls.left = outputs[1] > 0;
            this.controls.right = outputs[2] > 0;
            this.controls.brake = outputs[3] > 0;
        }
    }

    #buildInputs() {
        const inputs = this.sensor.readings.map(
            reading => (reading == null ? 0 : 1 - reading.offset)
        );
        inputs.push(this.speed / this.maxSpeed);
        inputs.push(this.lightState === "red" ? 1 : 0);
        inputs.push(this.lightState === "yellow" ? 1 : 0);
        inputs.push(this.lightState === "green" ? 1 : 0);
        inputs.push(this.lightState === null ? 0 : 1 - this.lightDistance / LIGHT_LOOKAHEAD);
        return inputs;
    }

    // Dummy traffic obeys the lights too, so the AI is not the only road user
    // held to the rules. Dummies brake rather than being destroyed, which
    // keeps a stalled wreck from permanently blocking a lane.
    #driveDummy(world) {
        // Lift off and coast for a red light or for a car ahead in the lane;
        // never brake hard. A dummy stopping dead in front of a following car
        // would kill that car for something it did not do and put noise
        // straight into its fitness.
        this.controls.forward = this.lightSignal < 0.55 && !this.#blockedAhead(world.traffic);
        this.controls.brake = false;
    }

    // Dummies take no damage from each other, so without this a faster wall
    // simply drove through a slower one and the merged pile blocked all three
    // lanes — an obstacle no brain could get past.
    #blockedAhead(traffic) {
        // On open road only the car directly ahead in this lane matters. When
        // braking for a light, any stopped car ahead counts whatever its lane:
        // otherwise a wall pulled up level with the wall already queued at the
        // light, and after green the two drove on side by side at near-equal
        // speeds as one block across every lane.
        const queueing = this.lightSignal > 0;
        for (const other of traffic) {
            if (other === this) {
                continue;
            }
            const gap = this.y - other.y;
            if (gap <= 0 || gap >= DUMMY_FOLLOW_DISTANCE) {
                continue;
            }
            if (Math.abs(other.x - this.x) < this.width) {
                return true;
            }
            if (queueing && Math.abs(other.speed) < 0.3) {
                return true;
            }
        }
        return false;
    }

    // The nearest light ahead within LIGHT_LOOKAHEAD: its colour and distance
    // (centre of car to stop line). Lights the car has already crossed are
    // behind it and ignored. Encounters are recorded once per light so the
    // legal-stop rate has a denominator.
    #senseLight(lights) {
        let nearest = null;
        let nearestDistance = Infinity;
        for (const light of lights) {
            const distance = this.y - light.y;
            if (distance > 0 && distance < LIGHT_LOOKAHEAD && distance < nearestDistance) {
                nearest = light;
                nearestDistance = distance;
            }
        }

        if (nearest === null) {
            this.lightState = null;
            this.lightDistance = Infinity;
            this.lightSignal = 0;
            return;
        }

        this.lightState = nearest.state;
        this.lightDistance = nearestDistance;
        this.lightSignal = nearest.isStopping ? 1 - nearestDistance / LIGHT_LOOKAHEAD : 0;

        if (!this.#metLights.has(nearest.id)) {
            this.#metLights.add(nearest.id);
            this.lightsMet++;
        }
        if (nearest.isRed && !this.#redLights.has(nearest.id)) {
            this.#redLights.add(nearest.id);
            this.redEncounters++;
        }
    }

    #trackLights(lights) {
        for (const light of lights) {
            if (this.y < light.y && !this.#passedLights.has(light.id)) {
                this.#passedLights.add(light.id);
                this.lightsCleared++;
            }
        }
    }

    // Counted once per light: the car is at rest with its front bumper inside
    // the stop zone and the light is red. Rewarded only the first time, so a
    // car cannot farm the bonus by sitting there.
    #trackLegalStop(lights) {
        if (this.speed >= STOP_SPEED) {
            return;
        }
        const front = this.y - this.height / 2;
        for (const light of lights) {
            if (!light.isRed || this.#stoppedLights.has(light.id)) {
                continue;
            }
            const gap = front - light.y;
            if (gap > 0 && gap <= STOP_ZONE) {
                this.#stoppedLights.add(light.id);
                this.legalStops++;
            }
        }
    }

    #trackOvertakes(traffic) {
        for (const other of traffic) {
            if (this.y < other.y && !this.#passedCars.has(other)) {
                this.#passedCars.add(other);
                this.carsPassed++;
            }
        }
    }

    #assessDamage(world) {
        for (let i = 0; i < world.borders.length; i++) {
            if (polysIntersect(this.polygon, world.borders[i])) {
                this.crashCause = "wall";
                return true;
            }
        }
        // Only the driven cars can be wrecked by traffic or a red light. The
        // dummies need the traffic list to follow the car ahead, but a wrecked
        // dummy would sit on the road for the rest of the generation.
        if (!this.useBrain && !this.driver) {
            return false;
        }
        for (const other of world.traffic) {
            // Two 50px cars cannot touch with centres 80px apart, and the whole
            // track is live at once, so this skips almost every polygon test.
            if (other === this || Math.abs(other.y - this.y) > 80) {
                continue;
            }
            if (polysIntersect(this.polygon, other.polygon)) {
                this.crashCause = "traffic";
                return true;
            }
        }
        for (const line of world.stopLines) {
            if (polysIntersect(this.polygon, line)) {
                this.crashCause = "redLight";
                this.redLightViolation = true;
                return true;
            }
        }
        return false;
    }

    #createPolygon() {
        const points = [];
        const rad = Math.hypot(this.width, this.height) / 2;
        const alpha = Math.atan2(this.width, this.height);
        points.push({
            x: this.x - Math.sin(this.angle - alpha) * rad,
            y: this.y - Math.cos(this.angle - alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(this.angle + alpha) * rad,
            y: this.y - Math.cos(this.angle + alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad,
            y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
            y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad
        });
        return points;
    }

    // Action arbitration. The four commands can all be raised at once by the
    // network, so they are resolved into one consistent action first:
    //   - brake overrides forward, so "accelerate and brake" brakes;
    //   - left and right together cancel, so the car holds its heading;
    //   - the car never reverses: braking stops at 0 and friction does the rest.
    #move() {
        const brake = this.controls.brake;
        const accelerate = this.controls.forward && !brake;
        const left = this.controls.left && !this.controls.right;
        const right = this.controls.right && !this.controls.left;
        this.decision = { accelerate, brake, left, right };

        if (brake) {
            this.speed = Math.max(0, this.speed - BRAKE_FORCE);
        } else if (accelerate) {
            this.speed += this.acceleration;
        }

        if (this.speed > this.maxSpeed) {
            this.speed = this.maxSpeed;
        }
        if (this.speed > 0) {
            this.speed -= this.friction;
        }
        if (this.speed < this.friction) {
            this.speed = 0;
        }

        // Steering only acts while moving; a parked car cannot pivot.
        if (this.speed > 0) {
            if (left) {
                this.angle += 0.03;
            }
            if (right) {
                this.angle -= 0.03;
            }
        }

        this.x -= Math.sin(this.angle) * this.speed;
        this.y -= Math.cos(this.angle) * this.speed;
    }

    draw(ctx, drawSensor = false) {
        if (this.sensor && drawSensor) {
            this.sensor.draw(ctx);
        }

        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(-this.angle);
        if (!this.damaged) {
            ctx.drawImage(
                Car.#getMask(this.color, this.width, this.height),
                -this.width / 2,
                -this.height / 2,
                this.width,
                this.height
            );
            ctx.globalCompositeOperation = "multiply";
        }
        ctx.drawImage(
            Car.#getSprite(),
            -this.width / 2,
            -this.height / 2,
            this.width,
            this.height
        );
        ctx.restore();
    }
}
