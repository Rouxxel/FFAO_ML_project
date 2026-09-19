# Stage 3 — Reynolds generalization (PRD §14 experiments 1–3)

## Setup

- Hydra: `dataset=splits`, `model=cnn_re` (`condition_on_re: true`).
- Simulation-level splits: `train_re`, `val_re`, `test_re` in
  `configs/dataset/splits.yaml`.
- Normalization and Re scaling use **training Reynolds only** (`train_re`).

## Experiments mapped to splits

| PRD experiment | Config list | Question |
|----------------|-------------|----------|
| **1** | `train_re` | Error at Reynolds numbers seen during training |
| **2** | `val_re` | Interpolation between trained Re |
| **3** | `test_re` | Extrapolation beyond trained Re |

## Commands (own multi-Re CFD or stub data)

```bash
# Populate dataset/simulations/* and metadata.csv (own CFD or stub backend).
python scripts/train.py --run-id cnn_multire \
  dataset=splits model=cnn_re --epochs 100

python scripts/evaluate_re_generalization.py \
  --run-dir results/runs/cnn_multire
```

Outputs:

- `re_generalization_metrics.json` — per-Re one-step MSE and regime tags
- `figures/re_generalization_heatmap.png` — PRD §13 item 12 style summary

## Tests

CI uses `tests/fixtures/build_multi_re_stub_dataset.py` (stub solver, no full CFD).
