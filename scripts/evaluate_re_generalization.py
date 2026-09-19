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

    python scripts/train.py --run-id cnn_multire \\
        dataset=splits model=cnn_re --epochs 50
    python scripts/evaluate_re_generalization.py \\
        --run-dir results/runs/cnn_multire
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
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
        help="Training run with model.pt and dataset=splits config.",
    )
    parser.add_argument(
        "--temporal-split",
        default=None,
        help="Temporal window for each sim (default: normalization.fit_split).",
    )
    args = parser.parse_args()

    result = run_re_generalization_evaluation(
        args.run_dir,
        eval_split=args.temporal_split,
    )
    print(f"Metrics: {result.metrics_path}")
    print(f"Heatmap: {result.heatmap_path}")


if __name__ == "__main__":
    main()
