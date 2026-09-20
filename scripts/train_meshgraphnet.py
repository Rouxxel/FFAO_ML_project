#!/usr/bin/env python3
"""
#############################################################################
### Train MeshGraphNet (Stage 2)
###
### @file train_meshgraphnet.py
### @author Sebastian Russo
### @date 2026
#############################################################################

One-step mesh node prediction. Default CPU; GPU recommended for the full train
split after import.

Example::

    python scripts/download_stage2_meshgraphnets.py --split train --max-trajectories 2
    python scripts/download_stage2_meshgraphnets.py --split val --max-trajectories 1
    python scripts/train_meshgraphnet.py --run-id stage2_meshgn
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.training.train_meshgraphnet import run_meshgraphnet_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """Compose Stage 2 config and run MeshGraphNet training."""
    parser = argparse.ArgumentParser(description="Train Stage 2 MeshGraphNet.")
    parser.add_argument("--run-id", default="stage2_meshgn", help="Run folder name.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override train.epochs."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override train.device (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/runs/<run_id>.",
    )
    args = parser.parse_args()

    overrides: list[str] = [
        "dataset=stage2_meshgraphnets",
        "model=meshgraphnet",
        "train=meshgraphnet",
        "eval=meshgraphnet",
    ]
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.device is not None:
        overrides.append(f"train.device={args.device}")

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    seed_from_config(cfg)
    out_dir = args.output_dir or run_directory(cfg.paths.runs_root, args.run_id)
    result = run_meshgraphnet_training(cfg, out_dir, repo_root=REPO_ROOT)

    log_handler.info("Model: %s", result.model_path)
    log_handler.info("Summary: %s", result.summary_path)
    log_handler.info("Val MSE: %.6f", result.best_val_mse)
    log_handler.info("Persistence val MSE: %.6f", result.persistence_val_mse)
    log_handler.info("Beats persistence: %s", result.beats_persistence)


if __name__ == "__main__":
    main()
