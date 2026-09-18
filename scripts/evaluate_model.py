#!/usr/bin/env python3
"""
#############################################################################
### Evaluate trained CNN (figures)
###
### @file evaluate_model.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Generate Stage 1 ML evaluation figures for an existing training run.

Example::

    python scripts/train.py --run-id cnn_stage1 --epochs 50
    python scripts/evaluate_model.py --run-dir results/runs/cnn_stage1
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.evaluation.model_report import run_model_evaluation

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Build vorticity and rollout stability figures under ``<run-dir>/figures/``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="ML evaluation figures for a CNN run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory (contains model.pt and config.yaml).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override eval split (train|val|test).",
    )
    args = parser.parse_args()

    result = run_model_evaluation(args.run_dir, split=args.split)
    print(f"Figures: {result.figures_dir}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
