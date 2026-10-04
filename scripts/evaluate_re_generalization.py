#!/usr/bin/env python3
"""
#############################################################################
### Evaluate Reynolds generalization (Stage 3 splits)
###
### @file evaluate_re_generalization.py
### @author Sebastian Russo
### @date 2026
#############################################################################

PRD §14 experiments 1–3: per-Re one-step error and generalization heatmap.

Example::

    python scripts/train.py --run-id stage3_cnn_re \\
        dataset=stage3_cfdbench model=cnn_re --epochs 50
    python scripts/evaluate_re_generalization.py \\
        --run-dir results/runs/stage3_cnn_re
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.evaluation.re_generalization import run_re_generalization_evaluation

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Write ``re_generalization_metrics.json`` and heatmap under the run directory.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Per-Re one-step errors for multi-Re CNN runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run with model.pt (e.g. results/runs/stage3_cnn_re).",
    )
    parser.add_argument(
        "--temporal-split",
        default=None,
        help="Temporal window for each sim (default: normalization.fit_split).",
    )
    parser.add_argument(
        "--no-rollout",
        action="store_true",
        help="Skip multi-step rollout curves per Re.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=None,
        help="Override eval.rollout_horizon from saved config.",
    )
    args = parser.parse_args()

    result = run_re_generalization_evaluation(
        args.run_dir,
        eval_split=args.temporal_split,
        include_rollout=not args.no_rollout,
        rollout_horizon=args.rollout_horizon,
    )
    log_handler.info("Metrics: %s", result.metrics_path)
    log_handler.info("Heatmap: %s", result.heatmap_path)
    if result.rollout_horizon_path is not None:
        log_handler.info("Rollout by Re: %s", result.rollout_horizon_path)


if __name__ == "__main__":
    main()
