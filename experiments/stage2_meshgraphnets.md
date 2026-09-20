# Stage 2 ML — MeshGraphNets `cylinder_flow`

## Hypothesis

On DeepMind **cylinder_flow** unstructured meshes, a **MeshGraphNet-style** model can
learn one-step node dynamics (velocity + pressure) and beat **persistence** on held-out
**trajectories**, with measurable **rollout drift** at longer horizons.

## Data

| Item | Value |
|------|--------|
| Source | [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) `cylinder_flow` TFRecords |
| Import | `scripts/download_stage2_meshgraphnets.py` or pipeline **import** phase |
| On-disk root | `dataset/meshgraphnets_data/` (`configs/dataset/stage2_meshgraphnets.yaml`) |
| Splits | Trajectory-level **train / val / test** from upstream shards (`metadata.csv`) |

**Citation:** Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*; cite the
official dataset URLs in [documentation/DATA_SOURCES.md](../documentation/DATA_SOURCES.md).

## Models (ml-stage2-v0.1 scope)

1. **MeshGraphNet (v1)** — encoder / message passing / decoder (`src/ffaoml/models/meshgraphnet.py`).
2. **Baselines** — persistence on normalized node features (train + eval summaries).
3. **Rollout** — autoregressive horizons `{1, 10, 50, 100}` on val trajectories (configurable).

## Orchestration

| Entry | Use when |
|-------|----------|
| `python scripts/run_stage2_pipeline.py --run` | Dedicated Stage 2 script (recommended) |
| `python main.py --stage2 --run` | Same pipeline from repo root |
| `python main.py --run dataset=stage2_meshgraphnets` | Hydra dataset override selects Stage 2 |

Phases: **import** → **mesh_validation** → **train_meshgn** → **eval_meshgn**.

Smoke import (laptop):

```bash
python scripts/run_stage2_pipeline.py --run --max-trajectories 2 --split train --split val
```

Full train split is multi-GB and **GPU**-friendly; default training config uses **CPU**.

## Known limits

- **TensorFlow** required only for TFRecord **import** (not for training/inference).
- Mesh and Stage 1 **grid** metrics are **not** directly comparable without interpolation.
- Default `batch_size: 1` for mesh graphs; large graphs need memory-aware caps.
- CI uses `tests/fixtures/meshgraphnets_mini/` only (no TFRecord download).

## Reproducibility checklist

Training run `results/runs/stage2_meshgn/` should include:

| Artifact | Purpose |
|----------|---------|
| `config.yaml` | Resolved Hydra config (`dataset=stage2_meshgraphnets`, …) |
| `preprocess_stats.json` | Velocity/pressure normalization (train trajectories only) |
| `dataset_manifest.json` | Snapshot of `dataset/meshgraphnets_data/manifest.json` |
| `bundle_manifest.json` | `config_hash`, `dataset_manifest_hash`, `seed`, `git_commit` |
| `training_summary.json` | Val MSE vs persistence |
| `model_eval_metrics.json` | Rollout curves + `note_stage1_comparison` |

Qualitative data check: `results/cfd_validation/stage2_meshgraphnets/SUMMARY.md`.

## Suggested command sequence

```bash
python scripts/run_stage2_pipeline.py --dry-run
python scripts/run_stage2_pipeline.py --run --max-trajectories 2
python scripts/evaluate_mesh_model.py --run-dir results/runs/stage2_meshgn
python scripts/clean_pipeline_artifacts.py --preset stage2 --dry-run
```
