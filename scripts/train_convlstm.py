#!/usr/bin/env python3
"""
#############################################################################
### Train ConvLSTM (Stage 1)
###
### @file train_convlstm.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Train ``FlowConvLSTM`` on temporal splits. Optional truncated unroll via
``train.unroll_steps`` and ``train.teacher_forcing``.

Example::

    python scripts/train_convlstm.py --run-id convlstm_stage1 --epochs 50
    python scripts/train_convlstm.py --unroll-steps 4 --epochs 80
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.training.train_convlstm import run_convlstm_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compose Hydra config with ``model=convlstm`` and run training.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train Stage 1 FlowConvLSTM.")
    parser.add_argument("--run-id", default="convlstm_train", help="Run folder name.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override train.epochs."
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=None,
        help="Override train.unroll_steps (multi-step loss).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/runs/<run_id>.",
    )
    args = parser.parse_args()

    overrides: list[str] = ["model=convlstm"]
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.unroll_steps is not None:
        overrides.append(f"train.unroll_steps={args.unroll_steps}")

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    seed_from_config(cfg)
    out_dir = args.output_dir or run_directory(cfg.paths.runs_root, args.run_id)
    result = run_convlstm_training(cfg, out_dir, repo_root=REPO_ROOT)

    print(f"Model: {result.model_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Val MSE: {result.best_val_mse:.6f}")
    print(f"Persistence val MSE: {result.persistence_val_mse:.6f}")
    print(f"Beats persistence: {result.beats_persistence}")


if __name__ == "__main__":
    main()
