"use strict";

const CAR_SPRITE_SRC = "images.png";
const LIGHT_LOOKAHEAD = 260;

// How close a dummy gets to the car ahead in its lane before lifting off.
// Coasting from 1.2 with friction 0.05 takes ~14px to stop, so it settles
// ~80px behind: clear of a 50px car, never driving through it.
const DUMMY_FOLLOW_DISTANCE = 95;

// Fitness shaping: distance is the main reward, crashing costs a fixed
// amount so a car that crawls safely can still beat one that sprints into a
// wall, and clearing a light pays for the patience it takes to wait for green.
//
// Overtaking is rewarded explicitly because distance alone is not enough to
// find it. On distance alone the population reliably evolved to drive up to
// the first wall of traffic and then tailgate it forever at the wall's own
// speed: completely safe, and worth more than any attempt to pass that ends
// in a crash. Paying for each car passed puts a gradient into that valley.
// The penalty is deliberately small. Crashing already costs a car every
// metre it would have driven afterwards, so a large extra penalty just
// double-counts it — and at 300 it priced every steering experiment out of
// the search, which is what kept the population parked behind the first wall.
const CRASH_PENALTY = 100;
const LIGHT_BONUS = 150;
const OVERTAKE_BONUS = 120;

class Car {
    static #sprite = null;
    static #masks = new Map();

    #passedLights = new Set();
    #passedCars = new Set();

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
        return [rayCount + 2, 8, 4];
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

        this.useBrain = controlType == "AI";
        this.maxDistance = 0;
        this.framesAlive = 0;
        this.lightsCleared = 0;
        this.carsPassed = 0;
        this.lightSignal = 0;

        if (controlType != "DUMMY") {
            this.sensor = new Sensor(this);
            this.brain = new NeuralNetwork(Car.brainShape(this.sensor.rayCount));
        }
        this.controls = new Controls(controlType);
        this.polygon = this.#createPolygon();
    }

    get fitness() {
        let score = this.maxDistance +
            this.lightsCleared * LIGHT_BONUS +
            this.carsPassed * OVERTAKE_BONUS;
        if (this.damaged) {
            score -= CRASH_PENALTY;
        }
        return score;
    }

    update(world) {
        if (this.damaged) {
            return;
        }

        this.lightSignal = this.#senseLight(world.lights);
        if (!this.useBrain) {
            this.#driveDummy(world);
        }

        this.#move();
        this.polygon = this.#createPolygon();
        this.framesAlive++;
        this.maxDistance = Math.max(this.maxDistance, this.startY - this.y);
        this.#trackLights(world.lights);
        this.#trackOvertakes(world.traffic);
        this.damaged = this.#assessDamage(world);

        // A wrecked car used to keep raycasting and feeding forward forever;
        // with 100 cars that was most of the frame budget spent on corpses.
        if (this.damaged || !this.sensor) {
            return;
        }

        this.sensor.update(world);
        const inputs = this.sensor.readings.map(
            reading => (reading == null ? 0 : 1 - reading.offset)
        );
        inputs.push(this.speed / this.maxSpeed);
        inputs.push(this.lightSignal);

        const outputs = NeuralNetwork.feedForward(inputs, this.brain);
        if (this.useBrain) {
            this.controls.forward = outputs[0] > 0;
            this.controls.left = outputs[1] > 0;
            this.controls.right = outputs[2] > 0;
            this.controls.reverse = outputs[3] > 0;
        }
    }

    // Dummy traffic obeys the lights too, so the AI is not the only road user
    // held to the rules. Dummies brake rather than being destroyed, which
    // keeps a stalled wreck from permanently blocking a lane.
    #driveDummy(world) {
        // Lift off and coast for a red light or for a car ahead in the lane;
        // never reverse. A dummy backing away from a light could ram the car
        // waiting behind it, which killed that car for something it did not
        // do and put noise straight into its fitness.
        this.controls.forward = this.lightSignal < 0.55 && !this.#blockedAhead(world.traffic);
        this.controls.reverse = false;
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

    // 0 when the road ahead is clear or the light is green, rising to 1 as a
    // red or yellow light gets closer. Without this the network had no way of
    // knowing a light existed, let alone what colour it was.
    #senseLight(lights) {
        let signal = 0;
        for (const light of lights) {
            if (!light.isStopping) {
                continue;
            }
            const distance = this.y - light.y;
            if (distance > 0 && distance < LIGHT_LOOKAHEAD) {
                signal = Math.max(signal, 1 - distance / LIGHT_LOOKAHEAD);
            }
        }
        return signal;
    }

    #trackLights(lights) {
        for (const light of lights) {
            if (this.y < light.y && !this.#passedLights.has(light.id)) {
                this.#passedLights.add(light.id);
                this.lightsCleared++;
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
                return true;
            }
        }
        // Only the driven cars can be wrecked by traffic or a red light. The
        // dummies need the traffic list to follow the car ahead, but a wrecked
        // dummy would sit on the road for the rest of the generation.
        if (!this.useBrain) {
            return false;
        }
        for (const other of world.traffic) {
            // Two 50px cars cannot touch with centres 80px apart, and the whole
            // track is live at once, so this skips almost every polygon test.
            if (other === this || Math.abs(other.y - this.y) > 80) {
                continue;
            }
            if (polysIntersect(this.polygon, other.polygon)) {
                return true;
            }
        }
        for (const line of world.stopLines) {
            if (polysIntersect(this.polygon, line)) {
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

    #move() {
        if (this.controls.forward) {
            this.speed += this.acceleration;
        }
        if (this.controls.reverse) {
            this.speed -= this.acceleration;
        }

        if (this.speed > this.maxSpeed) {
            this.speed = this.maxSpeed;
        }
        if (this.speed < -this.maxSpeed / 2) {
            this.speed = -this.maxSpeed / 2;
        }

        if (this.speed > 0) {
            this.speed -= this.friction;
        }
        if (this.speed < 0) {
            this.speed += this.friction;
        }
        if (Math.abs(this.speed) < this.friction) {
            this.speed = 0;
        }

        if (this.speed != 0) {
            const flip = this.speed > 0 ? 1 : -1;
            if (this.controls.left) {
                this.angle += 0.03 * flip;
            }
            if (this.controls.right) {
                this.angle -= 0.03 * flip;
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
