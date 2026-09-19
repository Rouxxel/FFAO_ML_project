#!/usr/bin/env python3
"""
#############################################################################
### Train one-step CNN (Stage 1)
###
### @file train.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Train ``FlowCNN`` on temporal train/val splits and write a checkpoint bundle under
``results/runs/<run_id>/``.

Example::

    python scripts/download_stage1_zenodo.py --local-file path/to/data.h5
    python scripts/train.py --run-id cnn_stage1 --epochs 50
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.training.train import run_cnn_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compose Hydra config and run CNN training.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train Stage 1 one-step FlowCNN.")
    parser.add_argument("--run-id", default="cnn_train", help="Run folder name.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override train.epochs."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hydra model group (e.g. cnn, fno, reconstruct).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/runs/<run_id>.",
    )
    args = parser.parse_args()

    overrides: list[str] = []
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.model is not None:
        overrides.append(f"model={args.model}")

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    seed_from_config(cfg)
    out_dir = args.output_dir or run_directory(cfg.paths.runs_root, args.run_id)
    result = run_cnn_training(cfg, out_dir, repo_root=REPO_ROOT)

    print(f"Model: {result.model_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Val MSE: {result.best_val_mse:.6f}")
    print(f"Persistence val MSE: {result.persistence_val_mse:.6f}")
    print(f"Beats persistence: {result.beats_persistence}")


if __name__ == "__main__":
    main()
