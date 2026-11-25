# Expected pipeline results (Stage 1)

What `python main.py --run` (default CNN path) produces on disk, and how to tell a run
succeeded. Contracts: [CONTRACTS.md](./CONTRACTS.md). Reset outputs:
[../scripts/clean_pipeline_artifacts.py](../scripts/clean_pipeline_artifacts.py).

---

## Is the run satisfactory?

A **successful default end-to-end run** ends with JSON summary `"status": "ok"`, an empty
`skipped_phases` list (unless you intentionally skipped steps), and all artifacts below
present for each executed phase.

| Phase | Pass signal |
|-------|-------------|
| **import** | `dataset/.../manifest.json`, `metadata.csv`, `simulations/.../fields.zarr` |
| **cfd_validation** | `results/cfd_validation/stage1_zenodo/SUMMARY.md`, figures, `metrics.json` |
| **baseline_eval** | `results/runs/stage1_baseline/metrics.json` |
| **train_cnn** | `results/runs/stage1_cnn/model.pt`, `training_summary.json` |
| **eval_cnn** | `results/runs/stage1_cnn/model_eval_metrics.json`, `figures/*.png` |

Benign warnings you may see: Zarr “consolidated metadata” (format 3), matplotlib
`tight_layout` on validation figures. They do not invalidate the run.

**Your 2026-09-19 run** used `upstream_strategy: zenodo_unavailable_lbm_fallback` → data
under `dataset/generated_data/`. All five phases completed with the files listed in
[Your run snapshot](#your-run-snapshot-lbm-fallback) below.

Optional phases **not** run unless requested: `--with-convlstm`, `--with-fno` (extra run
dirs and compare JSON).

---

## Generated layout (overview)

Paths are relative to the **repository root**. Everything here is **gitignored** (see
`.gitignore`); it is recreated by the pipeline.

```text
├── .cache/
│   └── zenodo_stage1/              # LBM / download cache (HDF5 before Zarr import)
│       └── cylinder_re100_grid64_last100.h5
├── dataset/
│   ├── generated_data/             # LBM fallback or --generate-data
│   │   ├── manifest.json           # Import provenance (source_id, DOI, hashes)
│   │   ├── metadata.csv            # One row per simulation (Re, split, path, grid…)
│   │   └── simulations/
│   │       └── re_100_zenodo/
│   │           └── fields.zarr/      # Chunked u, v, p, ω, time, grid coords
│   └── zenodo_data/                # Zenodo, cache copy, or --local-file import
│       ├── manifest.json
│       ├── metadata.csv
│       └── simulations/re_100_zenodo/fields.zarr/
├── log/
│   └── ffao_ml_YYYY-MM-DD.log      # Same messages as console (daily file)
├── results/
│   ├── cfd_validation/
│   │   └── stage1_zenodo/          # CFD sanity checks on imported fields
│   │       ├── SUMMARY.md
│   │       ├── metrics.json
│   │       ├── velocity_snapshots.png
│   │       ├── vorticity_snapshots.png
│   │       ├── shedding_spectrum.png
│   │       └── shedding_animation.gif
│   └── runs/
│       ├── stage1_baseline/        # Persistence / linear rollout (default run id)
│       │   ├── config.yaml
│       │   ├── preprocess_stats.json
│       │   └── metrics.json
│       └── stage1_cnn/             # Default CNN run id (--cnn-run-id)
│           ├── config.yaml         # Resolved Hydra config for this run
│           ├── preprocess_stats.json
│           ├── dataset_manifest.json   # Snapshot of dataset/.../manifest.json at train time
│           ├── bundle_manifest.json
│           ├── model.pt            # Best checkpoint used for eval
│           ├── checkpoint_epoch_*.pt   # Periodic checkpoints (every 10 epochs)
│           ├── training_summary.json   # Val MSE vs persistence, seed, hashes
│           ├── model_eval_metrics.json
│           └── figures/
│               ├── vorticity_pred_vs_true.png
│               ├── error_vs_horizon.png
│               └── rollout_stability.png
```

**Legacy:** a flat `dataset/manifest.json` + `dataset/simulations/...` layout is still
supported by the code if you imported before the `zenodo_data` / `generated_data` split.

---

## What each area contains

### `.cache/zenodo_stage1/`

Intermediate **HDF5** written by the LBM generator (or copied from a Zenodo download).
Import reads this and builds Zarr under `dataset/`. Safe to delete with
`clean_pipeline_artifacts.py --target cache` if you want to force regeneration; import
can rebuild from LBM when Zenodo has no attachment.

### `dataset/generated_data/` vs `dataset/zenodo_data/`

| Directory | When |
|-----------|------|
| `zenodo_data/` | Official or user-supplied HDF5, or successful Zenodo/cache import |
| `generated_data/` | LBM fallback, `--generate-data`, or `--only generate` + import |

Both use the same **simulation id** folder name (`re_100_zenodo` for Stage 1 catalog entry).

- **`manifest.json`** — import metadata (stage, `source_id`, Zenodo DOI, config/git hashes).
- **`metadata.csv`** — simulation catalog row(s); ML splits by **time** within Re=100 use
  `configs/dataset/stage1_zenodo.yaml` → `temporal_split`.
- **`fields.zarr/`** — Zarr groups: `velocity_x`, `velocity_y`, `pressure`, `vorticity`,
  `time`, `x`, `y` (see [CONTRACTS.md](./CONTRACTS.md)).

### `results/cfd_validation/stage1_zenodo/`

Qualitative CFD checks: snapshot panels, vortex shedding GIF/spectrum, human-readable
`SUMMARY.md`, numeric `metrics.json` (e.g. Strouhal-related stats when computable).

### `results/runs/stage1_baseline/`

Non-learned baselines on the val split: `metrics.json` with persistence and linear
rollout curves (`configs/eval/default.yaml` horizons).

### `results/runs/stage1_cnn/`

Full ML bundle for the default one-step CNN:

| File | Role |
|------|------|
| `model.pt` | Weights for inference and `eval_cnn` |
| `training_summary.json` | Epochs, seed, val MSE, beats-persistence flag |
| `model_eval_metrics.json` | Rollout / horizon metrics from eval phase |
| `figures/` | Standard eval plots from `evaluate_model` |
| `checkpoint_epoch_*.pt` | Training checkpoints (optional for analysis) |

Validate bundle integrity: `python scripts/validate_run_bundle.py --run-dir results/runs/stage1_cnn`.

### `log/`

Application log from `ffaoml.app_logging` / `src/utils/custom_logger.py`. Not required for
reproducibility of metrics; useful for debugging.

---

## Optional outputs (not in default `main.py --run`)

```text
results/runs/
├── stage1_convlstm/          # --with-convlstm
├── stage1_fno/               # --with-fno
└── …/multistep_compare.json  # after compare_multistep (ConvLSTM path)

outputs/                      # Hydra multirun when using scripts/train.py with Hydra cwd
.multirun/
.local_test_runs/             # pytest scratch (clean preset `all`)
```

---

## Your run snapshot (LBM fallback)

After `python main.py --run` on 2026-09-19, these paths were present:

| Artifact | Path |
|----------|------|
| Zarr store | `dataset/generated_data/simulations/re_100_zenodo/fields.zarr` |
| Dataset manifest | `dataset/generated_data/manifest.json` |
| LBM cache HDF5 | `.cache/zenodo_stage1/cylinder_re100_grid64_last100.h5` |
| CFD validation | `results/cfd_validation/stage1_zenodo/` (6 files) |
| Baseline | `results/runs/stage1_baseline/` (`config.yaml`, `metrics.json`, `preprocess_stats.json`) |
| CNN train + eval | `results/runs/stage1_cnn/` (`model.pt`, 5× `checkpoint_epoch_*.pt`, `figures/` × 3, JSON summaries) |
| Log | `log/ffao_ml_2026-09-19.log` |

**Conclusion:** this is a **complete, satisfactory** default Stage 1 pipeline run.

---

## Stage 2 (MeshGraphNets) — expected layout

After `python scripts/run_stage2_pipeline.py --run` (or `python main.py --stage2 --run`):

| Phase | Pass signal |
|-------|-------------|
| **import** | `dataset/meshgraphnets_data/manifest.json`, `metadata.csv`, `trajectories/<sim_id>/` |
| **mesh_validation** | `results/cfd_validation/stage2_meshgraphnets/SUMMARY.md`, `metrics.json` |
| **train_meshgn** | `results/runs/stage2_meshgn/model.pt`, `training_summary.json`, bundle JSON |
| **eval_meshgn** | `results/runs/stage2_meshgn/model_eval_metrics.json`, `figures/error_vs_horizon.png` |

```text
dataset/meshgraphnets_data/
├── manifest.json
├── metadata.csv
└── trajectories/
    └── cylinder_flow_train_00000/
        ├── mesh_pos.npy, cells.npy, node_type.npy
        ├── velocity.npy, pressure.npy, meta.json
results/cfd_validation/stage2_meshgraphnets/
results/runs/stage2_meshgn/
    ├── model.pt, config.yaml, preprocess_stats.json
    ├── dataset_manifest.json, bundle_manifest.json
    ├── model_eval_metrics.json
    └── figures/
        ├── error_vs_horizon.png
        └── rollout_stability.png
.cache/meshgraphnets_cylinder/   # TFRecord shards (optional keep)
```

Mesh metrics are **not** directly comparable to Stage 1 grid CNN/FNO numbers without
interpolation; see `note_stage1_comparison` in `model_eval_metrics.json`.

---

## Quick checks

```bash
python scripts/validate_run_bundle.py --run-dir results/runs/stage1_cnn
python scripts/clean_pipeline_artifacts.py --list    # sizes and presence
```

Re-run Stage 1 from scratch: `python scripts/clean_pipeline_artifacts.py --preset pipeline --yes`
then `python main.py --run`.

Re-run Stage 2: `--preset stage2` (mesh dataset, mesh cache, stage2 CFD figures, all runs).

---

## Stage 3 (CFDBench multi-Re) — expected layout

After `python scripts/run_stage3_pipeline.py --run` (or `python main.py --stage3 --run`):

| Phase | Pass signal |
|-------|-------------|
| **import** | `dataset/cfdbench_data/manifest.json`, `metadata.csv`, Zarr stores per `simulations/*` |
| **grid_validation** | `results/cfd_validation/stage3_cfdbench/SUMMARY.md`, `metrics.json`, `figures/` |
| **train_cnn_re** | `results/runs/stage3_cnn_re/model.pt`, `preprocess_stats.json`, bundle JSON |
| **eval_re** | `re_generalization_metrics.json`, `figures/re_generalization_heatmap.png` |

```text
dataset/cfdbench_data/
├── manifest.json
├── metadata.csv
└── simulations/
    └── cfdbench_cylinder_prop_*/
results/cfd_validation/stage3_cfdbench/
results/runs/stage3_cnn_re/
    ├── model.pt, config.yaml, preprocess_stats.json
    ├── re_generalization_metrics.json
    └── figures/
        ├── re_generalization_heatmap.png
        └── error_vs_horizon_by_re.png
.cache/cfdbench/   # HF snapshot (optional keep)
```

Re-run Stage 3: `--preset stage3` (CFDBench dataset, CFDBench cache, stage3 CFD figures, all runs).

Local release gate: `python scripts/verify_stage3_local.py` — see
[STAGE3_RELEASE.md](./STAGE3_RELEASE.md).
