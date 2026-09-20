#!/usr/bin/env python3
"""
#############################################################################
### Evaluate trained MeshGraphNet (Stage 2)
###
### @file evaluate_mesh_model.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Rollout metrics and horizon figures for an existing mesh training run.

Example::

    python scripts/train_meshgraphnet.py --run-id stage2_meshgn
    python scripts/evaluate_mesh_model.py --run-dir results/runs/stage2_meshgn
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.evaluation.evaluate_mesh_model import run_mesh_model_evaluation

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """Build rollout figures under ``<run-dir>/figures/``."""
    parser = argparse.ArgumentParser(
        description="MeshGraphNet rollout evaluation for a training run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory (model.pt + config.yaml).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override eval split (train|val|test).",
    )
    args = parser.parse_args()

    result = run_mesh_model_evaluation(args.run_dir, split=args.split)
    log_handler.info("Figures: %s", result.figures_dir)
    log_handler.info("Metrics: %s", result.metrics_path)


if __name__ == "__main__":
    main()
