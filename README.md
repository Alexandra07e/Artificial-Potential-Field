# Artificial Potential Field – Robot Path Planning

An interactive 2D simulation of a point robot that finds its way from **START** to **STOP** through polygonal obstacles using the **Artificial Potential Field (APF)** method. It is written in Python with **NumPy** for the math and **Matplotlib** for the visuals and animation.

The target pulls the robot toward it and the obstacles push it away. At every step the robot moves along the resultant force. When it gets stuck in a **local minimum**, which is the classic weakness of APF (for example inside a U-shaped obstacle), a **lateral (tangential) force** turns on by itself and steers it around the obstacle.

---

## Features

- **Attractive and repulsive forces**, both linear in distance
- **Automatic local-minimum escape** with a lateral force tangent to nearby obstacles
- **Smart side selection**: the planner simulates going around both the left and the right side and picks the shorter path that reaches the target (you can also force a side)
- **Live animation**: the robot moves step by step, with arrows for the attractive, repulsive, lateral and resultant forces
- **Field visualization**: repulsive iso-distance lines around obstacles (at `d0/3`, `2·d0/3` and `d0`) and attractive circles around the target
- **Real-time parameter tuning** from the keyboard. The robot carries on from where it is, so you see the effect right away
- **Interactive map editing**: draw new obstacles, delete them, move START and STOP with the mouse
- **Three built-in maps**: 3 obstacles, 5 obstacles, and a U-shaped trap
- **Safety handling**: collision detection, workspace walls, adaptive repulsion very close to obstacles

---

## The Method

Let **q** be the robot position, **q_goal** the target, *d* the distance to the closest point of an obstacle and *d₀* the obstacle's radius of influence.

| Component | Potential | Force |
|---|---|---|
| Attractive | U_a = ½ · ξ · ‖q − q_goal‖² | F_a = −ξ · (q − q_goal) |
| Repulsive (only if *d* < *d₀*) | U_r = ½ · η · (d₀ − d)² | F_r = η · (d₀ − d) · **n̂** |
| Lateral (only when stuck) | – | tangent to nearby obstacle edges, scaled by `lat` |

**n̂** is the unit vector pointing from the obstacle toward the robot. The forces of all obstacles add up as vectors:

```
F_total = F_a + Σ F_r + F_lat
```

At every iteration the robot takes a fixed-length step along the resultant:

```
x ← x + step · Fx / |F|
y ← y + step · Fy / |F|
```

### Local-minimum escape

1. The robot counts as **stuck** if it has barely moved over the last 60 steps, or if the resultant force is exactly zero.
2. The **lateral force** turns on. It is tangent to the obstacles closest to the robot, and its strength is `lat` × the attractive force (so `lat > 2` beats the attraction).
3. The side (left or right) is chosen automatically by simulating both and keeping the shorter successful path.
4. The lateral force turns off once the robot is clearly closer to the target than it was where it got stuck.

### Possible outcomes

| Status (in code) | Meaning |
|---|---|
| `ajuns` | Reached the target |
| `aproape` | Stopped near the target (the target is within `d0` of an obstacle, so try a smaller `d0`) |
| `blocat` | Stuck in a local minimum, even with the lateral force |
| `coliziune` | Hit an obstacle (attraction beat repulsion, so increase `eta`/`d0` or decrease `xi`) |
| `max_pasi` | Reached the maximum number of steps |

---

## Project Structure

```
Artificial-Potential-Field/
├── viz.py          # Main entry point: interactive visualization, animation, keyboard/mouse controls
├── planner.py      # Robot motion loop, local-minimum detection, lateral-force logic, collision checks
├── fields.py       # Attractive, repulsive and lateral potentials and forces
├── geometry.py     # Closest point / distance from a point to a segment and to a polygon
├── maps.py         # Built-in maps (3 obstacles, 5 obstacles, U-shape) + quick map preview
├── parameters.py   # Default parameters, their limits and keyboard increments
└── test.py         # Quick headless run on all maps (prints the final status)
```

> Note: the comments and identifiers in the code are in Romanian (for example *forta_atractie* = attractive force, *harta* = map, *tinta* = target).

---

## Getting Started

### Requirements

- Python 3.8+
- NumPy
- Matplotlib

### Run

```bash
python viz.py      # interactive simulation (starts on the U-shaped map)
python maps.py     # preview the three built-in maps side by side
python test.py     # headless run on all maps, prints the status of each
```

---

## Controls

### Keyboard

| Key | Action |
|---|---|
| `↑` / `↓` | Increase / decrease attraction **ξ** (`xi`) |
| `→` / `←` | Increase / decrease influence distance **d₀** (`d0`) |
| `W` / `S` | Increase / decrease repulsion **η** (`eta`) |
| `D` / `A` | Increase / decrease lateral-force strength (`lat`) |
| `P` | Cycle the side the robot goes around on: auto → left → right |
| `Shift` + key | Bigger step (×5) |
| `R` | Reset the parameters and restart from START |
| `Space` | Pause / resume |
| `3` / `5` / `U` | Switch map |
| `C` | Reload the original map (removes drawn obstacles) |
| `Z` | Undo the last obstacle |

### Mouse

| Action | How |
|---|---|
| Draw an obstacle | Press `O`, click to add vertices, `Enter` or right-click to close the polygon, `Esc` to finish |
| Delete an obstacle | Right-click on it |
| Move START | Press `I`, then click (the simulation restarts) |
| Move STOP | Press `F`, then click |

---

## Parameters

Defaults are in [`parameters.py`](parameters.py):

| Parameter | Default | Range | Description |
|---|---|---|---|
| `xi` | 1.0 | 0.1 – 5.0 | Attractive force gain |
| `eta` | 10.0 | 1.0 – 50.0 | Repulsive force gain |
| `d0` | 1.0 | 0.2 – 3.0 | Obstacle radius of influence |
| `lat` | 4.0 | 0.0 – 10.0 | Lateral force as a multiple of the attraction (0 = off) |
| `parte` | 0 | −1 / 0 / +1 | Side to go around on: left / auto / right |
| `pas` | 0.05 | – | Robot step length |

---

## Built-in Maps

| Map | Description |
|---|---|
| **3** | Triangle, quadrilateral and rectangle between START (1, 1) and STOP (9, 9) |
| **5** | Five obstacles of different shapes, which make a denser environment |
| **U** | A U-shaped trap opening toward START, the classic local-minimum case |

The workspace is a 10 × 10 square, and its edges act as walls.

---

## Authors

- **Tănase Alexandra**
- **Tîrsîna Nicoleta**

Politehnica University of Timișoara, Faculty of Automation and Computer Engineering
