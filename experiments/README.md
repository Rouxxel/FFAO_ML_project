# Experiments

This folder holds **human-readable experiment notes** (hypotheses, parameter
choices, result summaries). It is not a substitute for versioned configuration
or run artifacts.

## Where outputs go

Automated training and evaluation runs write under:

```text
results/runs/<run_id>/
```

Typical contents:

- `config.yaml` — resolved Hydra (or equivalent) configuration
- `checkpoints/` — model weights
- `metrics.json` or `metrics.jsonl` — scalar evaluation results
- `figures/` — plots for papers or reports

Hydra multirun sweeps may also create `multirun/` or `outputs/` at the repo
root during local development; those paths are gitignored.

## Suggested workflow

1. Copy or reference a config from `configs/` (added in later scaffolding).
2. Record the command line and git commit hash in a note file here, e.g.
   `experiments/2026-09-18_cnn_baseline.md`.
3. Link to the corresponding `results/runs/...` directory after the run finishes.

Keep large binaries and datasets out of git (`dataset/`, `results/` are
ignored). Published artifacts should follow [LICENSE-DATA](../LICENSE-DATA) and
[ATTRIBUTION.md](../documentation/ATTRIBUTION.md).
