# PRD: Machine Learning for Fluid Flow Around an Obstacle

## 1. Project Overview

Build a computational-physics project that studies whether machine-learning models can learn and predict fluid flow around an obstacle.

The canonical problem is two-dimensional incompressible flow around a cylinder.

The project should combine:

* Fluid mechanics
* Numerical simulation
* Computational fluid dynamics (CFD)
* Machine learning
* Reduced-order modeling
* Generalization across physical regimes

The project evolves in **stages** (see [DATA_SOURCES.md](./DATA_SOURCES.md)):

1. **Stage 1 (first):** Learn **temporal evolution** of cylinder wake dynamics at fixed
   conditions (e.g. Re ≈ 100, small public dataset).
2. **Stage 2 (later):** Unstructured **mesh** data (MeshGraphNets cylinder_flow) and
   graph-based surrogates.
3. **Stage 3 (later):** **Generalization** across physical conditions (e.g. CFDBench
   Reynolds / boundary / geometry sweeps) when storage allows.

**Primary research question (initial):**

> **Can a machine-learning model learn the temporal evolution of vortex shedding around a cylinder and remain stable over multi-step rollouts?**

**Extended research question (after Stage 3 or multi-Re own CFD):**

> **How well can surrogate models reproduce temporal evolution and physically relevant quantities, and how does performance degrade outside the physical conditions seen in training?**

The project must retain the physical interpretation of the problem rather than treating CFD output as arbitrary image data.

---

## 2. Goals

1. Generate or obtain CFD simulations of flow around a cylinder.
2. Understand the relationship between:

   * Reynolds number
   * Velocity
   * Viscosity
   * Pressure
   * Vorticity
3. Build a dataset containing flow fields over time.
4. Train ML models to predict future flow states.
5. Evaluate spatial and temporal prediction accuracy.
6. **Eventually** test generalization to unseen physical conditions (Re, BC, geometry)
   — not required for Stage 1.
7. Compare ML predictions against physical quantities such as drag, lift, and vorticity
   when the dataset or solver provides them.

---

## 3. Non-Goals

The first version should NOT attempt:

* Full 3D turbulence simulation.
* Industrial-scale CFD.
* Real-time CFD.
* Replacing a professional CFD solver.
* Making unsupported claims about ML outperforming CFD.

---

## 4. Physical Problem

Consider incompressible flow around a circular cylinder.

The governing equations are the incompressible Navier–Stokes equations:

```text
∂u/∂t + (u · ∇)u = -∇p/ρ + ν∇²u
∇ · u = 0
```

where:

```text
u = velocity field
p = pressure
ρ = density
ν = kinematic viscosity
```

The main dimensionless parameter is the Reynolds number:

```text
Re = U D / ν
```

where:

```text
U = characteristic flow velocity
D = cylinder diameter
ν = kinematic viscosity
```

---

## 5. Simulation Domain

Use a 2D rectangular domain:

```text
inlet
  →
  →
  →       ○ cylinder
  →             → wake
  →
  →
```

The cylinder should be positioned away from the inlet and outlet boundaries.

Configurable parameters:

```text
domain_width
domain_height
cylinder_diameter
inlet_velocity
viscosity
simulation_time
time_step
mesh_resolution
```

---

## 6. CFD Backend

The project should support one of:

* OpenFOAM
* FEniCS
* Dedalus
* custom finite-difference solver

Start with a low-resolution solver for experimentation.

The CFD solver must produce:

```text
velocity_x
velocity_y
pressure
vorticity
```

at each timestep.

---

## 7. Dataset

Each sample should contain:

```text
Reynolds number
time
velocity field
pressure field
vorticity field
```

Store metadata separately.

Recommended dataset structure:

```text
dataset/
├── simulations/
│   ├── re_50/
│   ├── re_100/
│   ├── re_200/
│   ├── re_500/
│   └── ...
└── metadata.csv
```

Each simulation should have clearly documented physical parameters.

Public CFD datasets may also be used where appropriate.

### 7.1 Staged data acquisition (recommended)

| Stage | Source | Role | Approx. size |
|-------|--------|------|----------------|
| **1** | [Zenodo Re=100 cylinder flow](https://zenodo.org/records/18669296) | Pipeline + one-step / multi-step ML | ~1 MB |
| **2** | DeepMind MeshGraphNets `cylinder_flow` | GNN / mesh-based ML | Multi-GB |
| **3** | CFDBench (cylinder subset) | Condition generalization | ~13 GB interpolated |

Full URLs, mirrors, and import notes: [DATA_SOURCES.md](./DATA_SOURCES.md).

**Stage 1** uses **time-based** train/val/test splits on a single trajectory.
**Multi-Re splits** (§11) apply when Stage 3 or in-house multi-Re simulations are used.

In-house CFD (§6) remains valuable but is **not** required before Stage 1 ML.

---

## 8. Baseline Analysis

Before training ML models, calculate:

### Reynolds number

```text
Re = U D / ν
```

### Drag coefficient

```text
Cd = Fd / (0.5 ρ U² D)
```

### Lift coefficient

```text
Cl = Fl / (0.5 ρ U² D)
```

### Vortex shedding frequency

Calculate the dominant frequency of lift fluctuations.

Estimate the Strouhal number:

```text
St = f D / U
```

These quantities should be used to validate the simulation.

---

## 9. ML Tasks

Implement the following tasks progressively.

### Task A — Flow-field reconstruction

Given partial information about the flow, reconstruct:

```text
velocity field
pressure field
vorticity field
```

---

### Task B — One-step prediction

Given:

```text
flow(t)
```

predict:

```text
flow(t + Δt)
```

---

### Task C — Multi-step prediction

Given an initial flow state, recursively predict:

```text
t1 → t2 → t3 → ... → tn
```

Measure how prediction error accumulates.

---

### Task D — Parameter-conditioned prediction

Provide the model with:

```text
current flow
Reynolds number
```

and predict future flow.

This allows a single model to represent multiple physical regimes.

---

## 10. Models

Implement simple baselines first.

### Baseline 1

Persistence:

```text
flow(t+1) = flow(t)
```

### Baseline 2

Linear prediction.

### Model 1

CNN-based predictor.

### Model 2

ConvLSTM or temporal CNN.

### Stretch Model

Fourier Neural Operator or another neural operator architecture.

The purpose is to determine whether increasing model complexity produces meaningful improvements.

---

## 11. Train/Test Split

Do NOT randomly split individual frames only within a single continuous simulation
without a documented protocol — that causes **temporal leakage**.

### Stage 1 (single trajectory, fixed Re)

Use **contiguous time blocks** (or rolling windows with strict causality) for
train / val / test — e.g. early times for training, later times for testing rollout.

### Stage 3 / multi-Re own CFD

Use **simulation-level splits by Reynolds number** (or by condition id for CFDBench).

Example (multi-Re):

```text
Training:
Re = 50, 75, 100, 150, 200

Validation:
Re = 125, 175

Testing:
Re = 250, 300, 400
```

This tests interpolation and extrapolation.

---

## 12. Evaluation Metrics

### Field error

Mean squared error:

```text
MSE(u_pred, u_true)
```

### Relative L2 error

```text
||u_pred-u_true||₂ / ||u_true||₂
```

### Physical quantity error

Compare predicted vs true:

```text
Cd
Cl
St
```

### Long-horizon stability

Measure how quickly predictions diverge from the actual CFD trajectory.

---

## 13. Required Visualizations

Generate:

1. CFD velocity field.
2. CFD vorticity field.
3. Pressure field.
4. Ground-truth vs predicted flow.
5. Prediction error field.
6. Vortex-shedding animation.
7. Lift coefficient vs time.
8. Drag coefficient vs time.
9. Fourier spectrum of lift.
10. Error vs prediction horizon.
11. Error vs Reynolds number.
12. Generalization heatmap.

---

## 14. Research Experiments

### Experiment 1

Can the model reproduce the flow field at Reynolds numbers seen during training?

### Experiment 2

Can it interpolate between Reynolds numbers?

### Experiment 3

Can it extrapolate to Reynolds numbers never seen during training?

### Experiment 4

Does predicting vorticity directly improve long-term prediction?

### Experiment 5

Does a physics-informed loss improve predictions?

Possible additional loss:

```text
L = L_data + λ L_divergence
```

where `L_divergence` penalizes violation of:

```text
∇ · u = 0
```

---

## 15. Project Architecture

```text
fluid-ml/
│
├── README.md
├── pyproject.toml
│
├── src/
│   ├── physics/
│   │   ├── navier_stokes.py
│   │   ├── reynolds.py
│   │   └── coefficients.py
│   │
│   ├── cfd/
│   │   ├── solver.py
│   │   └── mesh.py
│   │
│   ├── data/
│   │   ├── generation.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   ├── convlstm.py
│   │   └── neural_operator.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── plots.py
│
├── configs/
├── experiments/
├── tests/
└── results/
```

---

## 16. Technology

Recommended:

* Python
* NumPy
* SciPy
* PyTorch
* Matplotlib
* pandas

Potential CFD tools:

* OpenFOAM
* FEniCS
* Dedalus

Potential ML tools:

* PyTorch
* NeuralOperator

---

## 17. Reproducibility

Every simulation must store:

```text
Reynolds number
viscosity
velocity
geometry
mesh resolution
time step
random seed
```

Every ML experiment must store:

```text
model configuration
dataset version
training parameters
random seed
```

---

## 18. Success Criteria

The project is successful if:

1. The CFD solver produces physically plausible flow.
2. Vortex shedding is observable in the appropriate regime.
3. Drag/lift behavior can be measured.
4. ML can predict future flow better than trivial baselines.
5. Prediction error is quantified over multiple time horizons.
6. Generalization to unseen Reynolds numbers is explicitly evaluated.
7. The model's physical limitations are documented.

---

## 19. Stretch Goals

* Physics-informed neural networks.
* Neural operators.
* Reduced-order modeling using POD.
* Koopman operator models.
* Active learning for selecting CFD simulations.
* Learn a reduced dynamical system.
* Predict flow at much higher Reynolds numbers.
* Use ML as a surrogate to accelerate optimization.
* Optimize cylinder geometry to minimize drag.

---

## 20. Final Research Question

> **How well can machine-learning surrogate models reproduce the temporal evolution and physically relevant quantities of flow around a cylinder, and how does predictive performance degrade when evaluated outside the physical conditions represented in training data?**

Stage 1 emphasizes **temporal evolution and rollout stability** at fixed Re; Stage 3
(or own multi-Re CFD) emphasizes **distribution shift** across conditions.

---

## 21. Licensing and attribution

The project is **open source**. Shared artifacts should remain usable for research and education while requiring **appropriate credit** to the author and project.

| Material | License | Reference |
|----------|---------|-----------|
| Source code, configs, tests | Apache 2.0 | Repository `LICENSE` |
| Datasets, simulation exports, checkpoints, figures, metrics | CC BY 4.0 | Repository `LICENSE-DATA` |
| Attribution requirements | — | `NOTICE`, [ATTRIBUTION.md](./ATTRIBUTION.md) |

When publishing datasets or results (§7, §17), include enough metadata (version, commit, manifest) that others can cite the exact artifact. See [LEGAL.md](./LEGAL.md) and [CITATION.cff](../CITATION.cff).

**Copyright:** Sebastian Russo (update NOTICE and attribution docs if maintainers change).
