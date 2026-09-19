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
| Stage 1 import, Zarr layout, validation figures | Ready (scripts below) |
| ML `FlowDataset` / normalization (Stage 1 temporal splits) | Ready |
| CNN / ConvLSTM training, baselines, rollout eval | Ready (`ml-stage1-v0.1`) |

## Stage 1 data (Zenodo Re ≈ 100)

Catalog and attribution: [documentation/DATA_SOURCES.md](documentation/DATA_SOURCES.md).  
On-disk layout: [documentation/CONTRACTS.md](documentation/CONTRACTS.md).

### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[core,dev]"
# Optional for DataLoader smoke tests and training:
python -m pip install -e ".[ml]"
```

### 2. Import into `dataset/`

```bash
python scripts/download_stage1_zenodo.py
```

If the HDF5 is not listed on the [Zenodo record](https://zenodo.org/records/18669296)
yet, place `cylinder_re100_grid64_last100.h5` under `.cache/zenodo_stage1/` or pass:

```bash
python scripts/download_stage1_zenodo.py --local-file path/to/cylinder_re100_grid64_last100.h5
```

This writes `dataset/simulations/re_100_zenodo/fields.zarr`, `dataset/metadata.csv`,
and `dataset/manifest.json` (see `configs/dataset/stage1_zenodo.yaml`).

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
