# Stretch — FNO and physics-informed loss (Phase 7)

## FNO one-step predictor

Hydra model group `fno` wraps [neuraloperator](https://github.com/neuraloperator/neuraloperator)
`FNO` for grid-to-grid maps (PRD §10 stretch model).

```bash
pip install -e ".[ml]"
python scripts/train.py --run-id fno_stage1 --model fno --epochs 100
python scripts/evaluate_model.py --run-dir results/runs/fno_stage1
```

Tune `model.n_modes`, `model.n_layers`, and `model.hidden_channels` in
`configs/model/fno.yaml`. Modes must not exceed half the grid resolution per axis.

## Divergence penalty (PRD §14 experiment 5)

```yaml
train:
  loss:
    divergence_weight: 0.01
  physics:
    dx: <from simulation metadata>
    dy: <from simulation metadata>
```

The penalty uses central differences on predicted ``velocity_x`` / ``velocity_y``
(`ffaoml.physics.torch_ops`).

## Task A — reconstruction (optional)

`model=reconstruct` uses velocity-only input (2 channels) to predict all four
CONTRACT channels at the same time index.

```bash
python scripts/train.py --run-id reconstruct_stage1 --model reconstruct --epochs 50
```
