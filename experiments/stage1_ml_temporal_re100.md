# Stage 1 ML — temporal evolution at Re = 100

## Hypothesis

A grid-based model can learn **one-step** (and optionally **multi-step**) maps
along a single Zenodo cylinder wake trajectory at **Re = 100**, beating persistence
on the held-out **time block** and exposing **long-horizon drift** when rolled out
(PRD §12).

## Data

| Item | Value |
|------|--------|
| Source | [Zenodo](https://doi.org/10.5281/zenodo.18669296) — Addiucci (2026), CC BY 4.0 (`cylinder_re100_grid64_last100.h5`) |
| Import | `scripts/download_stage1_zenodo.py` |
| On-disk id | `re_100_zenodo` (`configs/dataset/stage1_zenodo.yaml`) |
| Splits | Contiguous time indices: train `[0,70)`, val `[70,85)`, test `[85,100)` |

**Upstream credit:** cite the Zenodo record and related PO-CAE code when publishing;
see [documentation/ATTRIBUTION.md](../documentation/ATTRIBUTION.md#stage-1-zenodo-re-100-flow-fields).

## Models (ml-stage1-v0.1 scope)

1. **Baselines** — persistence / linear (`scripts/evaluate.py`).
2. **FlowCNN** — one-step; recursive rollout for horizons `{1, 10, 25, 50}`.
3. **FlowConvLSTM** — carried hidden state; optional `train.unroll_steps > 1`.

## Reproducibility checklist

Each training run under `results/runs/<run_id>/` should include:

| Artifact | Purpose |
|----------|---------|
| `config.yaml` | Resolved Hydra config (includes `seed`) |
| `dataset_manifest.json` | Copy of `dataset/manifest.json` at train time |
| `bundle_manifest.json` | `config_hash`, `dataset_manifest_hash`, `seed`, `git_commit` |
| `training_summary.json` | Val MSE vs persistence + same hashes |
| `preprocess_stats.json` | Normalization fit on **train** time range only |

Validate locally:

```bash
python scripts/validate_run_bundle.py --run-dir results/runs/cnn_stage1
```

Record **git commit** (tag `ml-stage1-v0.1` when publishing this milestone) and
both hashes in papers or experiment tables.

## Suggested command sequence

```bash
pip install -e ".[core,dev,ml]"
python scripts/download_stage1_zenodo.py --local-file path/to/cylinder_re100_grid64_last100.h5
python scripts/validate_stage1_zenodo.py

python scripts/evaluate.py --run-id baseline_val --split val
python scripts/train.py --run-id cnn_stage1 --epochs 100
python scripts/evaluate_model.py --run-dir results/runs/cnn_stage1

python scripts/train_convlstm.py --run-id convlstm_stage1 --epochs 100
python scripts/compare_multistep.py \
  --cnn-run results/runs/cnn_stage1 \
  --convlstm-run results/runs/convlstm_stage1
```

## Notes

- Multi-step comparison and drift interpretation:
  [stage1_multistep_drift.md](./stage1_multistep_drift.md).
- Re-wise generalization is **out of scope** until Stage 3 (`dataset=splits`).
- Release tag **`ml-stage1-v0.1`**: create after `pytest` passes with `[ml]` on the
  commit you used for published numbers.
