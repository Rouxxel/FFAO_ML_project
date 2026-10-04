# Stage 3 release gate (local)

Stage 3 is **not** fully exercised in GitHub Actions: CI uses the `cfdbench_mini`
fixture and stub multi-Re data. Before tagging a Stage 3 data or ML release, run
the checks below on a machine with **~15 GB** free disk for the interpolated
CFDBench subset.

## Prerequisites

```bash
pip install -e ".[core,dev,ml]"
```

## End-to-end pipeline

```bash
python main.py --stage3 --run --max-cases 20
# or
python scripts/run_stage3_pipeline.py --run --max-cases 20
```

Phases: **import** → **grid_validation** → **train_cnn_re** → **eval_re**.

Expected artifacts:

| Phase | Pass signal |
|-------|-------------|
| import | `dataset/cfdbench_data/manifest.json`, `metadata.csv`, Zarr under `simulations/` |
| grid_validation | `results/cfd_validation/stage3_cfdbench/SUMMARY.md`, `metrics.json` |
| train_cnn_re | `results/runs/stage3_cnn_re/model.pt`, `training_summary.json` |
| eval_re | `results/runs/stage3_cnn_re/re_generalization_metrics.json`, heatmap figures |

## Automated local gate

After import (and optionally after full train + eval):

```bash
python scripts/verify_stage3_local.py
python scripts/verify_stage3_local.py --skip-eval   # dataset / test_re only
```

The gate checks that imported Reynolds numbers cover `test_re` from
`configs/dataset/stage3_cfdbench.yaml` (250, 300, 400 within `re_tolerance`)
and that `re_generalization_metrics.json` lists **test** / experiment 3 entries
when `model.pt` exists.

## Git tags (manual at release time)

Apply only when the checklist above passes on your machine. Tags are **not**
created by CI.

**Data import snapshot** (manifest + imported Zarr layout frozen in docs):

```bash
git tag -a data-stage3-v0.1 -m "Stage 3 CFDBench import layout v0.1"
```

**ML bundle** (trained `stage3_cnn_re` with Re generalization metrics):

```bash
git tag -a ml-stage3-v0.1 -m "Stage 3 cnn_re Re generalization v0.1"
git push origin data-stage3-v0.1 ml-stage3-v0.1
```

Record the commit hash, `dataset/cfdbench_data/manifest.json` hash, and
`bundle_manifest.json` from the run directory in your release notes.

## Known limits (v0.1)

- Cylinder **prop** subset only; other CFDBench families are out of scope.
- Grid QC validation does not compute Cd/Cl/St.
- Metrics are on **interpolated 64×64** grids; not comparable to Stage 1 Zenodo
  or Stage 2 mesh rollouts without extra post-processing.

See [experiments/stage3_re_generalization.md](../experiments/stage3_re_generalization.md)
and [DATA_SOURCES.md](./DATA_SOURCES.md).
