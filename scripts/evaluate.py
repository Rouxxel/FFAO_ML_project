#!/usr/bin/env python3
"""
#############################################################################
### Baseline evaluation CLI
###
### @file evaluate.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Run persistence / linear baselines on the configured temporal split and write
``metrics.json`` under ``results/runs/<run_id>/``.

Example::

    python scripts/download_stage1_zenodo.py --local-file path/to/data.h5
    python scripts/evaluate.py
    python scripts/evaluate.py --run-id baseline_smoke --split test
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.evaluation.runner import run_baseline_evaluation

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compose Hydra config, run baselines, print metrics path.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Evaluate naive flow baselines.")
    parser.add_argument(
        "--run-id",
        default="baseline_eval",
        help="Subdirectory under results/runs/",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override eval.split (train|val|test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override entire output directory.",
    )
    args = parser.parse_args()

    overrides: list[str] = []
    if args.split is not None:
        overrides.append(f"eval.split={args.split}")

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    seed_from_config(cfg)
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = run_directory(cfg.paths.runs_root, args.run_id)

    result = run_baseline_evaluation(cfg, out_dir, repo_root=REPO_ROOT)
    log_handler.info("Metrics: %s", result.metrics_path)
    for name, payload in result.metrics["baselines"].items():
        one = payload["one_step"]
        log_handler.info("  %s one-step MSE: %.6f", name, one["mse"])


if __name__ == "__main__":
    main()
