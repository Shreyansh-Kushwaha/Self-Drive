"use strict";

// The four raw commands a driver can issue. `brake` replaces the old
// `reverse`: the cars never drive backwards any more, a brake command only
// scrubs speed off towards zero (see Car.#move for the arbitration rules).
class Controls {
    constructor(type) {
        this.forward = false;
        this.left = false;
        this.right = false;
        this.brake = false;

        switch (type) {
            case "KEYS":
                this.#addKeyboardListeners();
                break;
            case "DUMMY":
                this.forward = true;
                break;
        }
    }

    #addKeyboardListeners() {
        document.addEventListener("keydown", (event) => {
            switch (event.key) {
                case "ArrowLeft":
                    this.left = true;
                    break;
                case "ArrowRight":
                    this.right = true;
                    break;
                case "ArrowUp":
                    this.forward = true;
                    break;
                case "ArrowDown":
                    this.brake = true;
                    break;
            }
        });
        document.addEventListener("keyup", (event) => {
            switch (event.key) {
                case "ArrowLeft":
                    this.left = false;
                    break;
                case "ArrowRight":
                    this.right = false;
                    break;
                case "ArrowUp":
                    this.forward = false;
                    break;
                case "ArrowDown":
                    this.brake = false;
                    break;
            }
        });
    }
}
