# Stage 3 — Reynolds generalization (PRD §14 experiments 1–3)

## Data source and citation

Stage 3 v1 uses the **CFDBench** interpolated cylinder subset (~**13.4 GB** on disk;
do not mirror the full ~460 GB raw release).

| Resource | URL |
|----------|-----|
| Paper / project | [CFDBench (GitHub)](https://github.com/luo-yining/CFDBench) |
| Download | [luoyining/CFDBench on Hugging Face](https://huggingface.co/datasets/luoyining/CFDBench) |

When publishing results, cite the CFDBench authors and note that this repo imports
**64×64** `u`, `v`, `p` (and derived ω) into the contract Zarr layout under
`dataset/cfdbench_data/`.

## Setup

- Hydra: `model=cnn_re` (`condition_on_re: true`).
- Dataset group:
  - **`dataset=stage3_cfdbench`** — imported CFDBench under `dataset/cfdbench_data/`
    (default Stage 3 E2E).
  - **`dataset=splits`** — same Re lists; use with stub/own-CFD under a custom
    `dataset.output_root`.
- Simulation-level splits: `train_re`, `val_re`, `test_re` in
  `configs/dataset/stage3_cfdbench.yaml` / `configs/dataset/splits.yaml`.
- Normalization and Re scaling use **training Reynolds only** (`train_re`).

## Experiments mapped to splits

| PRD experiment | Config list | Question |
|----------------|-------------|----------|
| **1** | `train_re` | Error at Reynolds numbers seen during training |
| **2** | `val_re` | Interpolation between trained Re |
| **3** | `test_re` | Extrapolation beyond trained Re |

## Commands — one-shot pipeline (recommended)

```bash
python main.py --stage3 --run --max-cases 20
# equivalent:
python scripts/run_stage3_pipeline.py --run --max-cases 20
```

Phases: `import` → `grid_validation` → `train_cnn_re` → `eval_re`. Use
`--skip-cfd-validation` on `main.py --stage3` to skip grid QC figures.

## Commands — CFDBench (step by step)

```bash
python scripts/download_stage3_cfdbench.py --max-cases 20
python scripts/validate_stage3_cfdbench.py

python scripts/train.py --run-id stage3_cnn_re \
  dataset=stage3_cfdbench model=cnn_re --epochs 50

python scripts/evaluate_re_generalization.py \
  --run-dir results/runs/stage3_cnn_re

python scripts/verify_stage3_local.py
```

Local release tags and checklist: [documentation/STAGE3_RELEASE.md](../documentation/STAGE3_RELEASE.md).

Tune `dataset.train_re` / `val_re` / `test_re` if imported cases do not match PRD
integers exactly (`import.re_tolerance` in `stage3_cfdbench.yaml`).

## Commands — stub / own multi-Re CFD

```bash
# Populate dataset/simulations/* and metadata.csv (own CFD or stub backend).
python scripts/train.py --run-id cnn_multire \
  dataset=splits model=cnn_re --epochs 100

python scripts/evaluate_re_generalization.py \
  --run-dir results/runs/cnn_multire
```

Outputs (under `results/runs/stage3_cnn_re/`):

- `re_generalization_metrics.json` — per-simulation one-step MSE, regime tags, rollout curves
- `figures/re_generalization_heatmap.png` — PRD §13 item 12 style summary
- `figures/error_vs_horizon_by_re.png` — rollout MSE vs horizon per Reynolds (optional `--no-rollout`)

## Known limits (v0.1)

- **Subset:** cylinder `prop` cases only; BC/geometry sweeps elsewhere in CFDBench are deferred.
- **Re coverage:** import must include simulations near every `train_re` / `val_re` /
  `test_re` in config (`import.re_tolerance`); use `--re` filters or raise `max_cases`
  if `verify_stage3_local.py` fails on `test_re`.
- **Metrics:** one-step and rollout MSE on normalized grids; not directly comparable to
  Stage 1 Zenodo CNN numbers or Stage 2 mesh rollouts without alignment work.
- **CI:** fixture/`cfdbench_mini` only; full HF download is manual.

## Tests

CI uses `tests/fixtures/build_multi_re_stub_dataset.py` (stub solver, no full CFD) and
`tests/test_stage3_pipeline.py` (orchestrator dry-run).
