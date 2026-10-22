# SelfDrivingCar

A self-driving car simulation in plain JavaScript (Canvas 2D, no libraries).
Each car is controlled by a small neural network whose weights are found by a
**genetic algorithm**. Soft Computing college project.

Full technical documentation, including the measured results and the
presentation notes, is in [`DOCUMENTATION.md`](DOCUMENTATION.md).

---

## 0. Running Locally

No installation and no build step. All you need is a static HTTP server, because
`localStorage` and image loading do not work reliably over `file://`:

```bash
python3 -m http.server 8000     # or: npm start
# then open http://localhost:8000
```

Tests (Node.js required, no dependencies):

```bash
npm test
```

The four suites under `tests/` stub out the DOM and the canvas so the real game
files can run inside Node. `operators.test.js` covers the network, the genetic
operators, parameter validation, braking, obstacle detection and the traffic
rules; `panels.test.js` covers the page (breeding panel, metrics, controls,
configuration, stage selection); `demo.test.js` covers the frozen demo mode,
evaluation, the three experiments and CSV export; `evolution.test.js` drives
the real animation loop over shortened generations and asserts that the
population genuinely improves. The harness replaces `Math.random` with a seeded
generator, so a run is reproducible rather than flaky.

## 0.1 Using the simulation

### Train a champion

1. Open the page. Training starts on its own: **stage A** (walls of traffic,
   no lights). Press **×20**.
2. The curriculum promotes the population automatically when its *measured*
   behaviour clears the bar (A → B once the champion overtakes ≥ 4 cars and the
   population averages ≥ 0.5 overtakes for two generations running; B → C once
   the champion makes ≥ 2 legal stops and clears ≥ 3 lights). The status line
   and the chart announce each promotion. You can pin a stage in **GA
   parameters → stage** instead.
3. The best brain is saved automatically (**Saved brain** panel shows it).
   A champion from a later stage always replaces one from an earlier stage;
   within a stage only a higher fitness does. **Save Best Brain** stores the
   current generation's best by hand; **Discard & Restart** wipes everything
   and rolls a new track.
4. Let stage C run for at least 20–30 generations. Watch the **Population
   metrics** table: the mean and median fitness, the collision rate and the
   legal-stop rate are the evidence of learning, not the best line alone
   (elitism makes the best line non-decreasing by construction).

### Demo the champion

Press **Demo Best Brain**. The saved champion is cloned and driven, frozen, on
the fixed demo route: three walls that force two lane changes, a light that is
red when the car arrives (6.2 s–22.2 s) and turns green, three more walls, a
second light, and a finish line at 4200 px. The status panel shows its speed,
sensor readings, the light state and distance, the action it chose
(accelerate / brake / steer), overtakes, legal stops, collisions and route
progress, and a **SUCCESS / FAILURE** verdict at the end. **Reset Demo** restarts
the identical route from the start line. Evolution is off throughout and the
panel confirms the genome is unchanged after playback.

If the saved brain fails the route, the demo says so. Go back to **Train
Population** and keep training; nothing is scripted and nothing is overridden.

### Measure it

**Benchmarks → Run Evaluation** runs, in time slices so the page stays usable:

- **A — GA vs untrained**: 20 random networks, the saved champion plus the five
  current elites, and a clearly labelled hand-coded rule baseline, all on the
  training track with the same rules.
- **B — mutation rates**: 0.05 / 0.10 / 0.20 / 0.30 with everything else held
  constant, two seeded trials each (population 30, 10 generations of 20 s), so
  the result is a comparison and not one lucky run.
- **C — unseen tracks**: the frozen champion on the training track, on five
  track seeds it never saw, and on the demo route, reported separately.

Results are printed under the network panel and stored. **Export Results
(CSV)** downloads every stored benchmark row plus the per-generation training
history (seeds, parameters, best / mean / median fitness, mean distance,
overtakes, collisions, legal stops, lights, diversity, outcome). Stored results
are only deleted by **Clear stored results**, which asks first.

### Other controls

| Control | Effect |
| --- | --- |
| **Pause / Resume** | Freezes the simulation and the breeding panel |
| **×1 / ×5 / ×20** | Simulation steps per drawn frame; lights still cycle at the correct simulated rate |
| **Load Saved Brain** | Restarts training from the saved champion, on its track, in its stage |
| **GA parameters → Apply** | Population, mutation rate and strength, tournament size, elite count, stage. Values are clamped to valid ranges and the active configuration is shown; changes take effect from the next generation |

## 0.2 Genetic Algorithm Implementation

Every component of a textbook genetic algorithm is implemented (`ga.js`,
operators in `network.js`, fitness in `car.js`):

| Component | Implementation |
| --- | --- |
| **Population** | 100 cars (`POP_SIZE`, configurable); each genome is the 124 weights and biases of one neural network |
| **Fitness** | `distance + 120 × overtakes + 100 × legal stops + 150 × lights cleared − 100 if crashed − 50 if the crash was running a red light` — see `Car.fitness` |
| **Selection** | Tournament selection, k = 3 (`Evolution.tournamentSelect`). Tournament was chosen over a roulette wheel because fitness can be negative once the crash penalty applies |
| **Crossover** | Uniform crossover: every gene is taken from one of the two parents with equal probability (`NeuralNetwork.crossover`) |
| **Mutation** | Per-gene probabilistic mutation: each gene changes with probability `rate = 0.2`, by up to `strength = 0.5` (`NeuralNetwork.mutate`) |
| **Elitism** | The best 2 brains (`ELITE_COUNT`) are carried into the next generation unmutated |
| **Stagnation escape** | If the champion's fitness is flat for 6 generations, mutation is boosted ×1.5 and 20% of the children are replaced by random immigrants until it moves again |
| **Curriculum** | Stage A (obstacles) → B (lights) → C (both), promoted on measured behaviour |
| **Generation loop** | No page reload. Each car runs until it crashes, finishes, or is retired for not making progress (less than 1.3 px/frame over 7 s, except while waiting at a red light); the generation ends when every car has stopped or after 60 s |

The course is one **fixed track built from a seed**, and every generation drives
the identical track, so a brain's fitness is a pure function of its genome.
Every random draw of the GA itself goes through a seeded generator too, so a
training run is reproducible from `(track seed, GA seed, parameters)`.

### Neural network: 10 inputs, 8 hidden, 4 outputs — 124 genes

| # | Input | Encoding |
| --- | --- | --- |
| 1–5 | `L45 L22 FWD R22 R45` | `1 − offset` of each sensor ray (280 px) |
| 6 | `SPD` | `speed / maxSpeed`, 0–1 |
| 7–9 | `RED YEL GRN` | one-hot colour of the nearest light within 260 px; all 0 if none |
| 10 | `DIST` | `1 − distance / 260` to that light; 0 if none |

Outputs: `accelerate`, `steer left`, `steer right`, `brake`. **Brake overrides
accelerate, left and right together cancel, and the car never reverses**: a
brake command scrubs 0.3 px/frame off the speed down to zero. The interface
shows the arbitrated action the car actually took.

The activation function is `tanh(weighted sum + bias)`.

### An accuracy note on "deep learning"

The network is trained by **evolution**, not by gradient descent. There is no
backpropagation and no loss function anywhere in this project — the weights are
searched by the genetic algorithm above. The phrase "deep learning" in the
sections below should be read as "an artificial neural network is used".

---
## 1. Problem Definition

In this project we built a game that uses an artificial neural network. The
subject of the game is to create a car that drives itself in traffic. Alongside
the network we also used a genetic algorithm, which lets our cars achieve better
results each time. In more detail: our car (or cars) drive themselves through
traffic containing other cars and traffic lights. The goal is for our car to
avoid hitting the cars around it, to stay on the road, and to pay attention to
red lights.

## 2. Research (Preliminary Work)

We carried out our research on the topics of deep learning and genetic
algorithms. First we researched how a car could drive itself using artificial
neural networks. At this point we learned how neural networks work, how their
formula is applied, and how a machine can make decisions without needing human
input. Afterwards we researched how we could optimise our AI-driven cars with a
genetic algorithm. We then applied what we had learned and researched in our
project. (We explain what we learned in detail in the sections below.)

## 3. Environment, Methods and Libraries Used

This project covers artificial neural networks and genetic algorithms. We did
our development in "VS Code" with JavaScript, HTML and CSS. Drawing is done with
the browser's built-in Canvas 2D API; **no library was used** at all, p5.js
included. We used "Object Oriented Programming" throughout development.

## 4. Proposed (Developed/Used) Method

### What is Deep Learning?

<img width="410" alt="Screenshot 2023-01-27 at 10 21 15" src="https://user-images.githubusercontent.com/57062662/215030507-51cd3577-b1b6-4552-8194-69ccbf14f163.png">

Deep learning is a subfield of machine learning (and therefore of artificial
intelligence) that focuses on a family of models called artificial neural
networks (ANN). The "deep" part of deep learning is a technical term and refers
to the number of layers or sections in the "network" part of "neural networks".
Deep learning currently plays a critical role in developing highly automated
systems such as driverless cars and natural language recognition and
understanding.

### Now let us look at how the deep learning layers we used in the project work

- <b><i>Inputs:</i></b>
Inputs come from an external trigger or from the outputs of other artificial
neurons. They are evaluated by the network and used as "food" for the neurons.
Thanks to the training we give the neurons, we can interpret the inputs.

- <b><i>Weights:</i></b>
They are multiplied by their corresponding input values. By making sense of the
inputs entering the neuron, a weight tries to make the output more successful.
The purpose of neural network training algorithms is to find the best possible
weight values needed to solve the problem.

- <b><i>Bias:</i></b>
It is assigned a value between -1 and 1.

- <b><i>Output:</i></b>
In our library-free algorithm, for each input we multiply the input by its weight
value and check whether the result is greater than the bias value. If it is
greater our output is 1, otherwise it is 0. Looking at our own layers:

> **Updated:** the current code uses `tanh(weighted sum + bias)` instead of this
> 0/1 threshold, and adds the bias rather than comparing against it. See
> section 0.1.

### NEURAL NETWORK LAYERS

<img width="369" alt="Screenshot 2023-01-27 at 10 22 08" src="https://user-images.githubusercontent.com/57062662/215030632-40d9cfd9-0937-4110-a4ef-a621ecc20244.png">

MULTIPLYING THE "MOVE FORWARD" INPUT VALUE BY THE WEIGHT VALUES:

<img width="366" alt="Screenshot 2023-01-27 at 10 22 27" src="https://user-images.githubusercontent.com/57062662/215030681-86696b89-40ce-4559-ae74-448692636b7b.png">

COMPARING THE INPUT * WEIGHT VALUE AGAINST THE BIAS VALUE:

<img width="364" alt="Screenshot 2023-01-27 at 10 22 47" src="https://user-images.githubusercontent.com/57062662/215030712-02dabe47-48bc-4583-9aeb-7ba65d56afcf.png">

### How Do We Produce the Input and Output Values?

We used 2 layers in our neural network system. So the inputs arrive at the first
level first. These inputs are processed and produce certain outputs. The outputs
produced at Level 0 then become the inputs of Level 1. By processing those inputs
as well, we determine which direction our car will go.

At Level 0 we need inputs. To obtain those inputs we added sensors to our car as
shown in the figure.

<img width="231" alt="Screenshot 2023-01-27 at 10 23 13" src="https://user-images.githubusercontent.com/57062662/215030781-f03e0369-dcb0-4818-8991-79ab0a13f484.png">

We can see there are black segments at the right and left ends of the sensors.
This is how the sensors detect the places where the car would crash and die
(cars in traffic, red lights and the edges of the road). Let us accept the value
of a sensor with no black segment as 1, and suppose the black segment on the far
left corresponds to a value of 0.2. Then 1 − 0.2 gives us the input value for
the leftmost side of the car. With this calculation we can see that the input
values of Level 0 are [0.8, 1, 1, 1, 0.8]. We multiply these values by weight
values generated between -1 and 1 for each input. Then we sum all the
input * weight values and compare them against the bias value, which we also
generated between -1 and 1.

If sum(input^i * weight^i) > bias then `level.outputs[i] = 1`, otherwise
`level.outputs[i] = 0`.

Our Level 1 outputs are formed from these results. These outputs are assigned as
shown below to make the car move:

```js
this.controls.forward = outputs[0];
this.controls.left    = outputs[1];
this.controls.right   = outputs[2];
this.controls.reverse = outputs[3];
```

<b><i>Our layer structure:</i></b>

```
4 (forward - left - right - reverse)
6 (hidden layer)
5 (car sensors)
```

> **Updated:** the current network is `7 (5 sensors + own speed + light signal)
> → 8 (hidden) → 4`, and the four outputs are thresholded at 0 because `tanh`
> returns a continuous value. See section 0.1.

###### How the neural network system works can also be examined in the videos.

### What is a Genetic Algorithm and How Does It Work?

Genetic algorithms are an optimisation method that works using mechanisms
similar to the evolutionary mechanisms observed in nature. They are a method of
searching for the best solution that iterates in multi-dimensional space in
order to optimise against a given cost function, and that is based on the
principle that the chromosome producing the best result in each iteration
survives. Instead of producing a single solution to a problem, genetic
algorithms produce a solution set made up of different solutions. In this way
many points in the search space are evaluated at the same time, and the
probability of reaching a global solution increases as a result. The solutions
in the solution set are completely independent of one another.

<img width="379" alt="Screenshot 2023-01-27 at 10 24 56" src="https://user-images.githubusercontent.com/57062662/215031050-2db819ca-5b81-453c-9c6d-72483e3b6c85.png">

#### In the photo below we see 100 AI-driven cars produced by the genetic algorithm

<img width="277" alt="Screenshot 2023-01-27 at 10 25 20" src="https://user-images.githubusercontent.com/57062662/215031122-538a931d-612b-45ba-a638-ca0d18b5f57a.png">

If we do not like how the cars are progressing and want to start their training
over from scratch, we press the "Discard Best Brain" button, which deletes the
`bestBrain` value from `localStorage`.

### In-Game Information

- <b><i>All-time Best Distance:</i></b> the greatest distance reached across all runs.
- <b><i>Generation:</i></b> which generation of the genetic algorithm is currently running.
- <b><i>Generation time:</i></b> how many seconds the current generation has been running.
- <b><i>Best car velocity:</i></b> the speed of the leading car.
- <b><i>Alive cars:</i></b> how many of our AI cars are still driving without having crashed.
- <b><i>Best distance:</i></b> the furthest the car has travelled in the current generation.
- <b><i>Best fitness:</i></b> the fitness score of the current generation's best car.

## 5. Experimental Work

> **Note:** this section describes the original manual workflow. Selection and
> saving are now automatic — see sections 0.1 and 0.2.

First we decided how many cars to use in the genetic algorithm. The higher this
value the better, but if we make it too large the computer can start to stutter.
Suppose we start the game with 100 cars. Since the cars have no foundation
(no successful data) at the beginning, very few of them will move successfully.
Whenever the user judges the car's movements to suit the goal, they should click
the "Save Best Brain" button on screen. This saves the "brain" data of the most
successful car (the car in front) into `localStorage`. From then on, when the
game restarts, all 100 cars will move according to the best brain data. If a
saved "best" brain exists, every car except the one in front is put through the
"mutate" operation again — because if we did not do this the cars would move
identically every round and no improvement could be recorded. Assuming all the
cars drive without crashing, every car except the front one is mutated again
while also taking the best car's data into account, and the chance of moving from
the current state to a better one increases considerably. As soon as a better
state is observed, the user at the computer presses "Save Best Brain" again. By
repeating these steps we obtain autonomous cars that drive better and better.
Here we select at a rate of 0.1 on each mutation.

### Legacy saved brains (no longer loadable)

The brain data below was recorded with the original 5-input network. The current
network takes 10 inputs, so these values are **rejected on load** by
`NeuralNetwork.fromJSON` and the simulation starts from a random population
instead. They are kept here only as a record of the original results; pasting a
brain by hand is no longer necessary, because the genetic algorithm now trains
automatically across generations.

To use them with the original version, the inputs were placed as follows after
opening `index.html` in Chrome:

1. Right-click on the Chrome page and choose Inspect.
2. Find the Application tab as shown below, enter the given input against
   `bestBrain`, and refresh the page.

<img width="546" alt="Screenshot 2023-01-27 at 10 27 07" src="https://user-images.githubusercontent.com/57062662/215031386-5c1d9215-a79b-49e8-a855-7c021065b8dd.png">

#### Without traffic lights (driving past the cars)

{"levels":[{"inputs":[0.02348594003872262,0.03164406782139184,0.2322865061966437,0,0.2655939633704416],"outputs":[0,1,1,1,1,1],"biases":[0.09297822152 049931,-0.27328541600108514,-0.0016449456029122358,-0.07520656222112074,-0.4293541169661284,- 0.3279127487240576],"weights":[[0.08769463411039442,-0.20525787940326268,0.14003961700644643,0.1043851277217048,- 0.3691710400429605,0.08670607795802623],[-0.207367828508094,0.3186562300725941,0.01976640242774084,-0.037625436278908464,- 0.37022452046616816,0.46572623925739637],[-0.43415981306444246,0.21383820561735722,0.08857513143532102,0.40329523646451904,- 0.2768191036060498,-0.6546733418568735],[0.18390659190470998,-0.18376744175635684,0.33262673764568707,- 0.41549834474630754,0.07097723970439074,- 0.39325880565738],[0.0883400257723848,0.3603886130223335,0.39608344908309856,0.2519185589250716,0.5654514339214273,0.13714958969467506]]},{"i nputs":[0,1,1,1,1,1],"outputs":[1,0,0,0],"biases":[0.0655654788051658,-0.2785955246880485,-0.22709448652322511,-0.34075734505366667],"weights":[[- 0.2583349349315358,-0.05001926042616811,-0.29167150334354647,-0.4280593375553857],[-0.22514419549554004,0.22632512782210507,- 0.08469268132174286,-0.4014008445903336],[0.06066213058058589,-0.39585891085793845,0.4317851598219412,- 0.18259075317802892],[0.5293400662344856,-0.28914400156473896,-0.3967479095136819,- 0.11204912637868336],[0.3876441931466752,0.23576970208265807,-0.3733612977049337,0.2277439404109114],[0.6307003641235356,-0.2901409587795225,- 0.10241756055337975,-0.4088241251678542]]}]}

#### With traffic lights (safe driving, obeying the lights and not hitting the cars)

{"levels":[{"inputs":[0.7444987497309877,0.8044478091469646,0.8193333333331398,0.8044478091469646,0.7444987497309877],"outputs":[1,1,1,1,1,0],"biases" :[0.8501546413716821,-0.7974595945236459,0.06617523635956672,-0.816712516428708,-0.962972830724471,- 0.39914939884476763],"weights":[[0.28870271802701836,-0.5577688494317377,0.07430366600156853,-0.03786300697932225,-0.11478857318347252,- 0.007552085447248877],[0.4712967582121296,-0.7038136002061226,0.01321525425357506,0.8052413263148939,0.6100810403056125,- 0.07006998345196726],[0.5927978044443415,0.5872374816500855,0.9446095731936673,-0.7548382121234812,0.6428356690327779,-0.4931115498057693],[- 0.5981189393340189,0.7979328499099527,0.7544250857871335,-0.5863041964081737,0.23603969786854898,-0.8669689747486905],[0.5151245607832291,- 0.548864694638644,0.0822303478877405,0.3361966427556694,0.8115812704085349,- 0.6880336120545487]]},{"inputs":[1,1,1,1,1,0],"outputs":[0,1,1,0],"biases":[-0.43263842298623134,-0.40846030433845365,- 0.46833411650573187,0.7980254685671557],"weights":[[0.04266209240667722,-0.22146399981300058,0.7741490354684679,0.45389850275297294],[- 0.06415399138160147,0.7896831661421895,0.46568285790042235,-0.4002620639333967],[-0.6421402556635472,0.19008116167447486,0.692333522803839,- 0.45875859281127696],[0.7344138307368712,0.3103788897957188,0.40996749454615533,-0.11615256817383415],[- 0.6435557766155903,0.40838040261621134,0.6247186503093534,-0.6279384391310641],[0.580556543942049,-0.8536022905015582,- 0.44381107646226364,0.5162635080285016]]}]}

### Video links of the successful drives we achieved

1. https://drive.google.com/file/d/1pTZOl7Uy7D8IkM_cBJ4DoxEFOkOTZjRY/view?usp=sharing
2. https://drive.google.com/file/d/139DbY0bWFFXtyDJemQ-CFzTU5NG3t7CP/view?usp=sharing
3. https://drive.google.com/file/d/1e0-DKuUMPDVGTEddV40PT_dod3mIMHh5/view?usp=sharing

## 6. Conclusion

#### Success of the Project

The project turned out quite successfully. Using both a neural network and a
genetic algorithm, we managed to create cars that can drive themselves while
obeying certain rules.

#### Its Function in Real Life

It can be used in all vehicles in real life. Today's autonomous cars are an
example of this. It is a popular subject that has been worked on a great deal
recently. It can be useful to everyone who cannot drive, who struggles on long
journeys, who does not enjoy driving but has to travel, and so on.

#### Benefit to People and Living Things

It saves people time. Autonomous cars obey all the rules, so they reduce the risk
of accidents and prevent harm to living things. Traffic congestion is reduced.

#### What We Gained

In this project we learned how artificial neural networks work and how their
formula is applied, and how a genetic algorithm and a neural network can be
combined to increase performance. Because we did all of this without using a
library, we gained more from it than we otherwise would have. Using artificial
intelligence by building a game also taught us a great deal about web
development.

##### Note 1

We used a genetic algorithm to increase success. We saved every result that came
close to success and made the machine produce better results starting from that
best state each time.

##### Note 2

The subject we chose is currently very popular all over the world. What makes
ours different is that we did not use a library in our project. Many of the
projects we had the chance to examine were built with Python or a Raspberry Pi
with library support. Because we wanted to examine the subject in depth, we
preferred to proceed without using a library.
