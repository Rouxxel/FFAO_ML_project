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
After `main.py --run`, see [documentation/EXPECTED_RESULTS.md](documentation/EXPECTED_RESULTS.md)
for where outputs land.

## Repository layout (source only)

Tracked project tree **excluding** gitignored pipeline outputs (`dataset/`, `log/`,
`results/`, `.cache/`, `outputs/`, `.local_test_runs/`, virtualenvs, caches):

```text
FFAO_ML_project/
├── configs/                    # Hydra: dataset, model, train, eval, simulation
│   ├── config.yaml
│   ├── dataset/
│   ├── eval/
│   ├── model/
│   ├── simulation/
│   └── train/
├── documentation/              # PRD, architecture, contracts, data, legal
│   ├── setup/                  # CI and platform notes
│   └── EXPECTED_RESULTS.md     # Pipeline artifacts (generated; not in git)
├── experiments/                # Stage runbooks and experiment notes
├── scripts/                    # CLI: import, train, eval, validate, clean
├── src/
│   ├── ffaoml/                 # Package: data, ML, training, pipeline, CFD, physics
│   │   ├── cfd/                # Solver adapters (stub, FD, Dedalus, OpenFOAM hook)
│   │   ├── data/               # I/O, catalog, Zenodo/LBM import, Zarr layout
│   │   ├── evaluation/         # Baselines, metrics, rollout, plots
│   │   ├── ml/                 # FlowDataset, preprocessing, splits
│   │   ├── models/             # CNN, ConvLSTM, FNO, MeshGraphNet
│   │   ├── pipeline/           # stage1.py + stage2.py (used by main.py)
│   │   ├── physics/            # NS helpers, Reynolds, torch ops
│   │   ├── training/           # Train loops, losses, checkpoints
│   │   └── validation/         # Stage 1 CFD validation figures
│   └── utils/                  # Logging, secure_file_io
├── tests/
│   ├── fixtures/               # Mini HDF5 and dataset builders
│   └── test_*.py
├── .github/workflows/          # CI and security workflows
├── main.py                     # Stage 1 (default) and Stage 2 (--stage2) entry point
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
├── LICENSE / LICENSE-DATA / NOTICE
└── README.md
```

Planning notes at repo root (`CFD_DATA_TASKS.md`, `FOUNDATION_TASKS.md`,
`ML_EVAL_TASKS.md`) may exist locally; they are optional and not required to run the pipeline.

## Status

| Layer | State |
|-------|--------|
| Package `ffaoml`, physics, Hydra configs, manifests, CI | Ready |
| Stage 1 import, Zarr layout, validation figures | Ready (scripts below) |
| ML `FlowDataset` / normalization (Stage 1 temporal splits) | Ready |
| CNN / ConvLSTM training, baselines, rollout eval | Ready (`ml-stage1-v0.1`) |
| Stage 2 MeshGraphNets import, mesh validation, train/eval | Ready (see below) |

## One-command pipeline (`main.py`)

From the repo root (after `pip install -r requirements.txt` or `pip install -e ".[core,dev,ml]"`):

```bash
python main.py --dry-run          # preview only — does not train or download
python main.py --run --local-file path/to/cylinder_re100_grid64_last100.h5
python main.py --run --with-convlstm --epochs 80 --force
```

Runtime logs: **`log/ffao_ml_YYYY-MM-DD.log`** (also echoed to the console). Import via
``from ffaoml.app_logging import log_handler``.

Bare `python main.py` logs a usage guide and exits (no work is done). Use **`--run`**
to execute; **`--dry-run`** to list steps without running them.

Steps: **import** → **CFD validation** → **baseline eval** → **CNN train/eval**;
optional **ConvLSTM** + multistep compare, **FNO** (`--with-fno`). Use `--only` /
`--from` on `main.py --run`, or run scripts under `scripts/` for one step at a time.

### Stage 2 pipeline (MeshGraphNets)

Import requires **TensorFlow** for TFRecords; training defaults to **CPU** (use
`train.device=cuda` or `--device cuda` on the train script for full shards).

```bash
python main.py --stage2 --dry-run
python main.py --stage2 --run --max-trajectories 2
# equivalent Hydra selector:
python main.py --run dataset=stage2_meshgraphnets --max-trajectories 2
# dedicated script (same orchestrator):
python scripts/run_stage2_pipeline.py --run --max-trajectories 2
```

Phases: **import** → **mesh_validation** → **train_meshgn** → **eval_meshgn**. Runbook:
[experiments/stage2_meshgraphnets.md](experiments/stage2_meshgraphnets.md).

Reset Stage 2 artifacts: `python scripts/clean_pipeline_artifacts.py --preset stage2 --yes`

### Reset local pipeline outputs

To re-test import → train from scratch (pipeline still supports skip-if-done when
artifacts remain):

```bash
python scripts/clean_pipeline_artifacts.py --list
python scripts/clean_pipeline_artifacts.py --preset pipeline --dry-run
python scripts/clean_pipeline_artifacts.py --preset pipeline --yes
```

Selective cleanup: `--target dataset-generated`, `--target dataset-zenodo`,
`--target runs`, `--run stage1_cnn`, `--preset models`, etc. (see script docstring).

## Stage 1 data (Zenodo Re ≈ 100)

Catalog and attribution: [documentation/DATA_SOURCES.md](documentation/DATA_SOURCES.md).  
On-disk layout: [documentation/CONTRACTS.md](documentation/CONTRACTS.md).

### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
# Full stack (core + dev + ML): see also requirements.txt
python -m pip install -r requirements.txt
# Or minimal CI-style install without PyTorch:
# python -m pip install -e ".[core,dev]"
```

### 2. Import into `dataset/`

The [Zenodo record](https://zenodo.org/records/18669296) often ships **code only**
(no HDF5). The pipeline **auto-fallback** runs an LBM generator when Zenodo/cache
has no file:

```bash
python main.py --dry-run          # shows import strategy (zenodo vs generate)
python main.py --run              # import → … (auto fallback → generated_data)
python main.py --run --generate-data   # force LBM → dataset/generated_data/
python main.py --run --only generate   # generate + import only
```

On-disk layout after import:

| Source | Directory |
|--------|-----------|
| Zenodo / cache / `--local-file` | `dataset/zenodo_data/` |
| LBM fallback / `--generate-data` | `dataset/generated_data/` |

Each contains `simulations/re_100_zenodo/fields.zarr`, `metadata.csv`, `manifest.json`.
Legacy flat `dataset/` is still detected if present.

If you already have an HDF5 (official or third-party), pass it explicitly:

```bash
python scripts/download_stage1_zenodo.py --local-file path/to/cylinder_re100_grid64_last100.h5
```

This writes `dataset/simulations/re_100_zenodo/fields.zarr`, `dataset/metadata.csv`,
and `dataset/manifest.json` (see `configs/dataset/stage1_zenodo.yaml`).

**Upstream data credit:** Addiucci, L. (2026). *Physics-Constrained Convolutional
Autoencoders for 2D Cylinder Flow at Re=100* (Version 1.0) [Dataset]. Zenodo.
https://doi.org/10.5281/zenodo.18669296 (CC BY 4.0; Copyright © 2026 Luca Addiucci).

More detail: [experiments/stage1_zenodo_import.md](experiments/stage1_zenodo_import.md).

### 3. CFD validation figures

```bash
python scripts/validate_stage1_zenodo.py
```

Outputs under `results/cfd_validation/stage1_zenodo/` (snapshots, shedding GIF,
`SUMMARY.md`, `metrics.json`).

### 4. ML data loading (Phase 0)

After import, Python can build one-step windows without hard-coded paths:

```python
from hydra import compose, initialize_config_dir
from ffaoml.config import config_dir
from ffaoml.ml import build_flow_datasets, fit_preprocess_stats

with initialize_config_dir(config_dir="configs", version_base="1.3"):
    cfg = compose(config_name="config")
stats = fit_preprocess_stats(cfg)  # train time range only
datasets = build_flow_datasets(cfg, stats=stats)
sample = datasets["train"][0]  # input/target tensors (C, H, W)
```

Switch to multi-Re simulation splits (Stage 3): `dataset=splits` in Hydra overrides.

### 5. Baseline evaluation (ML Phase 1)

```bash
python scripts/evaluate.py
python scripts/evaluate.py --run-id baseline_val --split val
```

Writes `results/runs/<run_id>/metrics.json` with persistence and linear rollout
curves (`configs/eval/default.yaml`).

### 6. Train one-step CNN (ML Phase 2)

Requires the `[ml]` extra (`torch`):

```bash
pip install -e ".[core,dev,ml]"
python scripts/train.py --run-id cnn_stage1 --epochs 50
```

Writes `model.pt`, `config.yaml`, `preprocess_stats.json`, `dataset_manifest.json`
(when present), and `training_summary.json` (includes comparison vs persistence).

### 7. Model evaluation figures (ML Phase 3)

```bash
python scripts/evaluate_model.py --run-dir results/runs/cnn_stage1
```

Writes `<run-dir>/figures/` (`vorticity_pred_vs_true.png`, `error_vs_horizon.png`,
`rollout_stability.png`) and `model_eval_metrics.json`. Re-wise heatmaps are deferred
to Stage 3.

### 8. ConvLSTM and multi-step compare (ML Phase 4)

```bash
python scripts/train_convlstm.py --run-id convlstm_stage1 --epochs 50
# Optional truncated unroll during training:
python scripts/train_convlstm.py --run-id convlstm_unroll4 --unroll-steps 4

python scripts/compare_multistep.py \
  --cnn-run results/runs/cnn_stage1 \
  --convlstm-run results/runs/convlstm_stage1
```

Writes `multistep_compare.json` and `figures/cnn_vs_convlstm_horizon.png` with MSE
at horizons `{1, 10, 25, 50}` (`configs/eval/default.yaml`). See
[experiments/stage1_multistep_drift.md](experiments/stage1_multistep_drift.md).

### 9. Reproducibility (ML Phase 5)

Training runs persist `config.yaml`, `seed` (top-level Hydra + `training_summary.json`),
`dataset_manifest.json`, and `bundle_manifest.json` (config and dataset hashes).
Check a run:

```bash
python scripts/validate_run_bundle.py --run-dir results/runs/cnn_stage1
```

End-to-end Stage 1 ML notes: [experiments/stage1_ml_temporal_re100.md](experiments/stage1_ml_temporal_re100.md).

### 10. Multi-Re conditioning (ML Phase 6)

Requires multiple simulations in `dataset/metadata.csv` (own CFD or stub data).
Use `dataset=splits` and `model=cnn_re`:

```bash
python scripts/train.py --run-id cnn_multire dataset=splits model=cnn_re --epochs 50
python scripts/evaluate_re_generalization.py --run-dir results/runs/cnn_multire
```

See [experiments/stage3_re_generalization.md](experiments/stage3_re_generalization.md).

### 11. FNO and reconstruction (ML Phase 7)

```bash
python scripts/train.py --run-id fno_stage1 --model fno --epochs 50
# Physics-informed loss: train.loss.divergence_weight=0.01 in Hydra overrides
python scripts/train.py --run-id reconstruct_stage1 --model reconstruct --epochs 50
```

Details: [experiments/fno_stretch.md](experiments/fno_stretch.md).

## Python package

Import name: **`ffaoml`**, [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) under `src/ffaoml/`. Layout and components:
[documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md).

### Verify

```bash
python -c "import ffaoml; print(ffaoml.__version__)"
pytest -m "not slow and not gpu and not cfd"
python scripts/compose_config.py
ruff check src tests scripts && ruff format --check src tests scripts
```

CI: [documentation/setup/CI.md](documentation/setup/CI.md) (GitHub Actions).

### Release tags (optional)

- `foundation-v0.1` — package, configs, CI baseline  
- `data-stage1-v0.1` — Stage 1 import + validation (after local import/validation)  
- `ml-stage1-v0.1` — Stage 1 ML Phases 0–4 (baselines, CNN, ConvLSTM, compare); tag after `pytest` with `[ml]`

## Documentation

| Document | Description |
|----------|-------------|
| [documentation/PRD.md](documentation/PRD.md) | Requirements and research questions |
| [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) | System design and repository layout |
| [documentation/CONTRACTS.md](documentation/CONTRACTS.md) | CFD / dataset / ML tensor contracts |
| [documentation/EXPECTED_RESULTS.md](documentation/EXPECTED_RESULTS.md) | Pipeline output paths and success checklist |
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
`pyproject.toml` when tagging releases.

**Copyright © 2026 Sebastian Russo**
