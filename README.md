# FFAO ML Project

**Fluid Flow Around an Obstacle** — research codebase for **2D incompressible flow
around a cylinder**: staged public datasets, optional in-house CFD, and ML models
that respect physical fields (velocity, pressure, vorticity) rather than treating
simulations as generic images.

## Goals

1. Build a reproducible pipeline from **flow data → datasets → predictors → evaluation**.
2. Start with **temporal evolution** at fixed conditions (Stage 1, Re ≈ 100).
3. Later explore **mesh-based** (Stage 2) and **condition generalization** (Stage 3)
   when storage and prior stages allow.

Research questions and success criteria: [documentation/PRD.md](documentation/PRD.md).  
Dataset stages and URLs: [documentation/DATA_SOURCES.md](documentation/DATA_SOURCES.md).

## Status

| Layer | State |
|-------|--------|
| Package `ffaoml`, physics, Hydra configs, manifests, CI | Ready |
| Stage 1 data on disk (`dataset/`) | Not imported yet |
| ML training | Not started |

**Suggested next step:** import the small **Zenodo Re=100** dataset (Stage 1) into
`dataset/` using the conventions in [CONTRACTS.md](documentation/CONTRACTS.md) and
[REPRODUCIBILITY.md](documentation/REPRODUCIBILITY.md). In parallel, a CFD **solver
stub** can validate the export schema before full simulations.

## Python package

Import name: **`ffaoml`**, [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) under `src/ffaoml/`. Layout and components:
[documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md).

### Install (development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[core,dev]"
```

Optional extras: `ml`, `cfd-dedalus`, `track` (see `pyproject.toml`).  
Platforms: [documentation/setup/PLATFORMS.md](documentation/setup/PLATFORMS.md).

### Verify

```bash
python -c "import ffaoml; print(ffaoml.__version__)"
pytest -m "not slow and not gpu and not cfd"
python scripts/compose_config.py
ruff check src tests && ruff format --check src tests
```

CI: [documentation/setup/CI.md](documentation/setup/CI.md) (GitHub Actions).

## Documentation

| Document | Description |
|----------|-------------|
| [documentation/PRD.md](documentation/PRD.md) | Requirements and research questions |
| [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) | System design and repository layout |
| [documentation/CONTRACTS.md](documentation/CONTRACTS.md) | CFD / dataset / ML tensor contracts |
| [documentation/DATA_SOURCES.md](documentation/DATA_SOURCES.md) | Staged datasets (Stage 1 active) |
| [documentation/REPRODUCIBILITY.md](documentation/REPRODUCIBILITY.md) | Manifests and checkpoint bundles |
| [documentation/TECH_STACK.md](documentation/TECH_STACK.md) | Technology choices |
| [documentation/LEGAL.md](documentation/LEGAL.md) | Licensing overview |
| [documentation/ATTRIBUTION.md](documentation/ATTRIBUTION.md) | How to give credit |

## License and attribution

This project is **open source**. You may use the code and published results if you
**give appropriate credit**.

| Material | License |
|----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Datasets, checkpoints, figures, metrics | [CC BY 4.0](LICENSE-DATA) |

See [NOTICE](NOTICE) and [documentation/ATTRIBUTION.md](documentation/ATTRIBUTION.md).
Academic citation: [CITATION.cff](CITATION.cff) — keep `version` aligned with
`pyproject.toml` when tagging releases (e.g. `foundation-v0.1`).

**Copyright © 2026 Sebastian Russo**
