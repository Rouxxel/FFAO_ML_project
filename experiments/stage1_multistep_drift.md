# Stage 1 — long-horizon drift (PRD §12)

## Question

How quickly do learned predictors diverge from the true Re=100 cylinder trajectory
when rolled out autoregressively?

## Models compared

| Model | Rollout mechanism |
|-------|-------------------|
| **CNN** | One-step `FlowCNN`; each step feeds the previous **prediction** (no memory). |
| **ConvLSTM** | `FlowConvLSTM`; hidden/cell state **carried** across steps. |
| **Persistence** | `flow(t+k) ≈ flow(t)` baseline. |

## Metrics

- Field **MSE** and **relative L²** at horizons in `configs/eval/default.yaml`:
  `{1, 10, 25, 50}` (capped by split length on short fixtures).
- Full per-step curves: `results/runs/<run>/figures/error_vs_horizon.png` (CNN eval)
  and `multistep_compare.json` from `scripts/compare_multistep.py`.

## Expected behavior (literature-aligned)

1. **Horizon 1** — CNN and ConvLSTM should be close if both are trained on
   one-step (or short unroll) loss; neither should lose to persistence on val
   when training succeeded (`beats_persistence` in `training_summary.json`).
2. **Horizons 10–50** — Recursive CNN error often **super-linear** in horizon
   because each step compounds discretization and learned bias; vorticity phases
   drift first.
3. **ConvLSTM** — Carrying convolutional memory can **slow** error growth when
   `train.unroll_steps > 1` with teacher forcing exposes the model to multi-step
   targets during training; gains are not guaranteed on Stage 1’s single trajectory.

## Commands

```bash
python scripts/download_stage1_zenodo.py --local-file path/to/cylinder_re100_grid64_last100.h5
python scripts/train.py --run-id cnn_stage1 --epochs 100
python scripts/train_convlstm.py --run-id convlstm_stage1 --epochs 100
# Optional multi-step training loss:
python scripts/train_convlstm.py --run-id convlstm_unroll4 --unroll-steps 4 --epochs 100

python scripts/compare_multistep.py \
  --cnn-run results/runs/cnn_stage1 \
  --convlstm-run results/runs/convlstm_stage1
```

Record git commit and dataset manifest hash from each run directory when publishing
a table of horizon errors.

## Interpretation checklist

- [ ] Report `at_report_horizons.mse` for CNN, ConvLSTM, and persistence.
- [ ] Note eval split (`eval.split`, default `val`) and `rollout_horizon`.
- [ ] If CNN beats ConvLSTM at h=1 but loses at h≥25, attribute to **compounding**
  vs **temporal state** (PRD §12 long-horizon stability).
- [ ] Tie qualitative drift to vorticity panels from `scripts/evaluate_model.py`.
