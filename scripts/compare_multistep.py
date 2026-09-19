#!/usr/bin/env python3
"""
#############################################################################
### Compare CNN recursive rollout vs ConvLSTM
###
### @file compare_multistep.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Write ``multistep_compare.json`` with MSE at ``eval.report_horizons``.

Example::

    python scripts/train.py --run-id cnn_stage1
    python scripts/train_convlstm.py --run-id convlstm_stage1
    python scripts/compare_multistep.py \\
        --cnn-run results/runs/cnn_stage1 \\
        --convlstm-run results/runs/convlstm_stage1
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.evaluation.multistep_compare import run_multistep_comparison

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compare multi-step errors for two trained runs.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="CNN recursive vs ConvLSTM multi-step rollout."
    )
    parser.add_argument(
        "--cnn-run",
        type=Path,
        required=True,
        help="CNN training run directory.",
    )
    parser.add_argument(
        "--convlstm-run",
        type=Path,
        required=True,
        help="ConvLSTM training run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write multistep_compare.json (default: convlstm run).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override eval split (train|val|test).",
    )
    args = parser.parse_args()

    result = run_multistep_comparison(
        args.cnn_run,
        args.convlstm_run,
        output_dir=args.output_dir,
        split=args.split,
    )
    log_handler.info("Metrics: %s", result.metrics_path)
    if result.horizon_figure is not None:
        log_handler.info("Figure: %s", result.horizon_figure)


if __name__ == "__main__":
    main()
