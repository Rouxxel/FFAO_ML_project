"""
#############################################################################
### Evaluation runner
###
### @file runner.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Hydra-driven baseline evaluation and ``metrics.json`` export.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.config import write_resolved_config
from ffaoml.data.loading import load_split_tensor
from ffaoml.evaluation.baselines import BaselineName
from ffaoml.evaluation.rollout import (
    one_step_baseline_metrics,
    rollout_curve,
    sample_at_report_horizons,
)
from ffaoml.manifests import git_short_commit, hash_config
from ffaoml.ml.preprocessing import (
    PreprocessStats,
    fit_preprocess_stats,
    normalize_fields,
    save_preprocess_stats,
)

"""CONSTANTS-----------------------------------------------------------"""
METRICS_FILENAME = "metrics.json"

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class EvaluationResult:
    """Paths and payload from ``run_baseline_evaluation``."""

    output_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


"""RUNNER-----------------------------------------------------------"""


def _load_normalized_split(
    cfg: DictConfig,
    split: str,
    stats: PreprocessStats,
) -> np.ndarray:
    raw = load_split_tensor(cfg, split)
    return normalize_fields(raw, stats)


def run_baseline_evaluation(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> EvaluationResult:
    """
    Evaluate persistence and linear baselines; write ``metrics.json``.

    Parameters:
        cfg (DictConfig): Composed config (``eval``, ``dataset``, ``paths``).
        output_dir (str | Path): Run directory under ``results/runs/``.
        repo_root (str | Path | None): For git commit in metrics.

    Returns:
        EvaluationResult: Output paths and metrics dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = str(cfg.eval.split)
    horizon = int(cfg.eval.rollout_horizon)
    report_horizons = [int(h) for h in cfg.eval.report_horizons]
    baseline_names: list[BaselineName] = [str(b) for b in cfg.eval.baselines]

    stats = fit_preprocess_stats(cfg)
    save_preprocess_stats(out / "preprocess_stats.json", stats)
    series = _load_normalized_split(cfg, split, stats)

    baseline_results: dict[str, Any] = {}
    for name in baseline_names:
        one_step = one_step_baseline_metrics(series, name)
        curve = rollout_curve(series, name, horizon)
        baseline_results[name] = {
            "one_step": one_step,
            "rollout": {
                "horizons": list(curve.horizons),
                "mse": list(curve.mse),
                "relative_l2": list(curve.relative_l2),
                "n_starts": curve.n_starts,
                "at_report_horizons": sample_at_report_horizons(curve, report_horizons),
            },
        }

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "kind": "baseline_evaluation",
        "split": split,
        "rollout_horizon_requested": horizon,
        "rollout_horizon_effective": len(
            baseline_results[baseline_names[0]]["rollout"]["horizons"]
        ),
        "baselines": baseline_results,
        "config_hash": hash_config(cfg),
        "git_commit": git_short_commit(repo_root),
        "normalization": {
            "fit_split": stats.fit_split,
            "time_range": [stats.time_start, stats.time_end],
        },
    }

    metrics_path = out / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_resolved_config(cfg, out)

    return EvaluationResult(output_dir=out, metrics_path=metrics_path, metrics=metrics)
