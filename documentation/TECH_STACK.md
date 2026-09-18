# Technology Stack

This document defines the **approved technology choices** for the Fluid Flow Around an Obstacle (FFAO) ML project. It complements [PRD.md](./PRD.md) (requirements) and [ARCHITECTURE.md](./ARCHITECTURE.md) (system design and layout).

---

## Design principles

1. **Python-first** — One language for CFD orchestration, data, ML, and evaluation.
2. **Physics-aware** — Fields and metrics stay tied to Navier–Stokes quantities (velocity, pressure, vorticity, Re, Cd, Cl, St), not generic image pipelines.
3. **Backend-agnostic CFD** — Simulations export a common field format; ML code does not depend on a specific solver.
4. **Reproducibility** — Every simulation and ML run is fully specified by config + seed + dataset version.
5. **Progressive complexity** — Start with low-resolution solvers and simple models; add FNO, PINNs, and OpenFOAM only when justified.

---

## Stack summary

| Layer | Primary choice | Alternatives / later |
|--------|----------------|----------------------|
| Language | Python 3.11+ | — |
| Packaging | `pyproject.toml`, **uv** or pip | conda/mamba for Dedalus/CUDA stacks |
| Configuration | **Hydra** + YAML under `configs/` | OmegaConf-only if Hydra is too heavy |
| Numerics | **NumPy**, **SciPy** | Numba (custom solver hot loops) |
| CFD (v1) | **Dedalus** or **custom finite-difference** | FEniCSx, **OpenFOAM** (via adapter) |
| Field I/O | **Zarr** + **xarray** | NetCDF4, HDF5 (h5py) |
| Tabular metadata | **pandas** | — |
| Deep learning | **PyTorch 2.x** | — |
| Neural operators | **neuraloperator** (FNO) | Custom U-Net / AFNO if needed |
| Training (optional) | **PyTorch Lightning** | Plain PyTorch loops in v1 |
| Visualization | **Matplotlib**, **Seaborn** | Plotly (interactive), imageio (GIF/video) |
| Testing | **pytest** | — |
| Lint / format | **ruff** | pre-commit hooks |
| Experiment tracking (optional) | **MLflow** or **Weights & Biases** | JSON logs in `results/` only |
| Data versioning (optional) | **DVC** | Manual dataset manifests |

---

## Core runtime

### Python

- **Version:** 3.11 or 3.12 (3.11 recommended for broad scientific wheel support).
- **Virtual environment:** project-local venv via `uv venv` or `python -m venv`; document activation in README.

### Project metadata

- **`pyproject.toml`:** project name, dependencies (grouped: `cfd`, `ml`, `dev`), optional `[tool.ruff]`, `[tool.pytest.ini_options]`.
- **Lockfile (recommended):** `uv.lock` or export from conda for reproducible CI.

---

## Computational fluid dynamics

### Primary path (v1)

| Tool | When to use |
|------|-------------|
| **Dedalus** | Default for 2D incompressible flow around a cylinder; scripted Re sweeps; spectral accuracy at moderate resolution. |
| **Custom finite-difference solver** | Minimal dependencies, full control, coursework-friendly baseline; lives in `src/cfd/`. |

### Secondary / stretch paths

| Tool | When to use |
|------|-------------|
| **FEniCS / FEniCSx** | FEM-based meshes, variational formulations, or comparison with literature FEM setups. |
| **OpenFOAM** | External validation, finer industrial-style meshes; requires **WSL2** or Linux on Windows. |

### Solver output contract

Regardless of backend, each timestep must provide (on a structured 2D grid):

- `velocity_x`, `velocity_y`
- `pressure`
- `vorticity` (computed in-solver or post-processed consistently)

Stored with explicit physical parameters: Re, U, ν, D, domain size, Δt, mesh resolution, seed.

---

## Data and storage

| Component | Role |
|-----------|------|
| **xarray** | Labeled dimensions `(time, y, x)` and variables/channels for ML loaders. |
| **Zarr** | Chunked storage for long time series; good for partial reads during training. |
| **pandas** | `metadata.csv`: one row per simulation (Re, paths, split, physics params). |
| **PyTorch `Dataset`** | Windowing for one-step / multi-step prediction; Re conditioning as extra input channels or embeddings. |

**On-disk layout** (see PRD §7 and ARCHITECTURE.md):

```text
dataset/
├── simulations/
│   ├── re_050/
│   ├── re_100/
│   └── ...
├── metadata.csv
└── manifest.json          # optional: version, checksums, creation config
```

---

## Machine learning

### Framework

- **PyTorch 2.x** — CNN, ConvLSTM, conditioning on Re, physics-informed losses (∇·u on the grid).
- **CUDA** — Use when available; CPU fallback must remain possible for small grids and tests.

### Model tiers (PRD §10)

| Tier | Implementation |
|------|----------------|
| Persistence / linear baselines | NumPy; optional **scikit-learn** for linear maps on flattened or POD coefficients |
| CNN one-step predictor | PyTorch (`src/models/cnn.py`) |
| Temporal model | ConvLSTM or temporal CNN (`src/models/convlstm.py`) |
| Neural operator | **neuraloperator** FNO (`src/models/neural_operator.py`) |

### Training

- **v1:** `src/training/train.py` with explicit loops, checkpointing, and config from Hydra.
- **Optional:** PyTorch Lightning for multi-experiment scaling and structured logging.

### Splits

- **Simulation-level splits by Reynolds number** — never random per-frame splits (PRD §11).

---

## Physics, metrics, and visualization

| Need | Library |
|------|---------|
| Re, Cd, Cl, St, FFT for shedding | NumPy, SciPy |
| Field MSE, relative L² | NumPy / PyTorch |
| Static plots, spectra, error curves | Matplotlib |
| Heatmaps (e.g. generalization) | Seaborn |
| Animations (vorticity, shedding) | Matplotlib animation or **imageio** |

Post-processing that computes drag/lift from fields should live in `src/physics/` and be reused by CFD validation and ML evaluation.

---

## Configuration and experiments

| Tool | Role |
|------|------|
| **Hydra** | Compose `configs/default.yaml` + overrides; output run dir under `results/` or `experiments/runs/`. |
| **YAML** | Simulation domains, Re lists, model hyperparameters, train/val/test Re sets. |

Each ML experiment records: model config, dataset version/manifest, optimizer, epochs, seed, git commit (optional).

---

## Development and quality

| Tool | Role |
|------|------|
| **pytest** | Unit tests for physics formulas, dataset splits, metric definitions |
| **ruff** | Lint + format |
| **pre-commit** (optional) | Enforce ruff on commit |
| **Jupyter** | Exploratory CFD sanity checks and figure prototyping (not production training path) |

---

## Platform notes (Windows)

| Scenario | Recommendation |
|----------|----------------|
| Dedalus + PyTorch (CPU/GPU) | Prefer **WSL2 Ubuntu** or Linux for smoothest Dedalus install; Windows-native possible with care. |
| OpenFOAM | **WSL2** or Docker; expose case I/O through `src/cfd/adapters/openfoam.py`. |
| Custom FD solver | Runs natively on Windows with NumPy only. |

Document the chosen platform in the project README once the primary CFD path is fixed.

---

## Optional dependencies (dependency groups)

Suggested `pyproject.toml` optional groups:

```text
[project.optional-dependencies]
core = ["numpy", "scipy", "pandas", "xarray", "zarr", "matplotlib", "seaborn"]
ml = ["torch", "neuraloperator"]
cfd-dedalus = ["dedalus"]          # platform-specific install notes
dev = ["pytest", "ruff", "pre-commit", "hydra-core"]
track = ["mlflow"]                 # or wandb
```

Install examples:

```bash
pip install -e ".[core,ml,dev]"
# CFD backend added when ready:
pip install -e ".[core,ml,dev,cfd-dedalus]"
```

---

## Mapping to PRD §16

| PRD recommendation | This stack |
|--------------------|------------|
| Python, NumPy, SciPy, PyTorch, Matplotlib, pandas | Adopted as core |
| OpenFOAM, FEniCS, Dedalus | Dedalus/custom first; others via adapters |
| NeuralOperator | Adopted for FNO stretch goal |

---

## Version pinning policy

- Pin **major.minor** for PyTorch and CUDA in documentation; exact pins in lockfile or `requirements-lock.txt`.
- Pin **dedalus** only after the first successful environment recipe is recorded in README or `docs/setup/`.
- Dataset format version in `manifest.json` when preprocessing logic changes.

---

## Related documents

- [PRD.md](./PRD.md) — product and research requirements
- [ARCHITECTURE.md](./ARCHITECTURE.md) — components, data flow, repository layout
