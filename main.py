#!/usr/bin/env python3
"""
#############################################################################
### FFAO ML — Stage 1 end-to-end pipeline
###
### @file main.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Run import → CFD validation → baselines → CNN train/eval (optional ConvLSTM, FNO).

Required:
pip install -e ".[core,dev,ml]"

``python main.py`` alone does **not** run anything — you must pass ``--run`` or ``--dry-run``.

Quick reference::

    # Plan (no download, training, or eval)
    python main.py --dry-run

    # Full pipeline (import: Zenodo/cache/local → else LBM → dataset/generated_data/)
    python main.py --run
    python main.py --run --generate-data
    python main.py --run --local-file path/to/cylinder_re100_grid64_last100.h5
    python main.py --run --force

    # Optional ConvLSTM / FNO
    python main.py --run --with-convlstm --with-fno --epochs 80 --fno-epochs 50

    # Single phase (--run required)
    python main.py --run --only import
    python main.py --run --only generate
    python main.py --run --only train_cnn --epochs 50
    python main.py --run --from train_cnn

    # Clean slate before re-run
    python scripts/clean_pipeline_artifacts.py --preset pipeline --yes
"""

# Native imports
from __future__ import annotations

import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler, shutdown_logger
from ffaoml.pipeline.stage1 import (
    PHASE_ORDER,
    PipelinePhase,
    PrerequisiteError,
    Stage1PipelineOptions,
    print_summary,
    run_stage1_pipeline,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parent

"""CLI-----------------------------------------------------------"""

_USAGE_GUIDE = """
================================================================================
WARNING: `python main.py` does not run the pipeline.

  --dry-run   Show which steps would run (no download, training, or evaluation).
  --run       Actually execute the pipeline (required for real work).

Quick reference (full pipeline)
--------------------------------------------------------------------------------
  python main.py --dry-run
  python main.py --run
  python main.py --run --generate-data
  python main.py --run --local-file path/to/cylinder_re100_grid64_last100.h5
  python main.py --run --force
  python main.py --run --with-convlstm --with-fno --epochs 80 --fno-epochs 50

Single phase (--run required)
--------------------------------------------------------------------------------
  python main.py --run --only generate
  python main.py --run --only import
  python main.py --run --only import --local-file path/to/data.h5
  python main.py --run --only cfd_validation
  python main.py --run --only baseline_eval
  python main.py --run --only train_cnn --epochs 50
  python main.py --run --only eval_cnn
  python main.py --run --only train_convlstm --with-convlstm
  python main.py --run --only compare_multistep --with-convlstm
  python main.py --run --from train_cnn

Phases: generate, import, cfd_validation, baseline_eval, train_cnn, eval_cnn,
        train_convlstm, compare_multistep, train_fno, eval_fno

Import (auto): Zenodo/cache/local HDF5 → dataset/zenodo_data/; on failure LBM
→ dataset/generated_data/. --generate-data forces LBM on import; --only generate
runs LBM file generation only (no import).

Individual scripts (same steps, run manually)
--------------------------------------------------------------------------------
  python scripts/generate_stage1_cylinder_h5.py   # if Zenodo has no HDF5
  python scripts/download_stage1_zenodo.py [--local-file ...]
  python scripts/validate_stage1_zenodo.py
  python scripts/evaluate.py [--run-id ...]
  python scripts/train.py [--run-id ...] [--model cnn|fno|reconstruct]
  python scripts/evaluate_model.py --run-dir results/runs/<run_id>
  python scripts/train_convlstm.py [--run-id ...]
  python scripts/compare_multistep.py --cnn-run ... --convlstm-run ...
  python scripts/evaluate_re_generalization.py --run-dir ...  (multi-Re)

Reset local outputs (see scripts/clean_pipeline_artifacts.py docstring)
--------------------------------------------------------------------------------
  python scripts/clean_pipeline_artifacts.py --list
  python scripts/clean_pipeline_artifacts.py --preset pipeline --dry-run
  python scripts/clean_pipeline_artifacts.py --preset pipeline --yes
  python scripts/clean_pipeline_artifacts.py --target dataset-generated --yes

Install first: pip install -e ".[core,dev,ml]"
================================================================================
"""


def print_usage_guide() -> None:
    """Log a short catalog of valid invocations when no action flag is set."""
    log_handler.warning("%s", _USAGE_GUIDE.strip())


def build_parser() -> argparse.ArgumentParser:
    """
    Build the root pipeline argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    phase_choices = [p.value for p in PipelinePhase]
    parser = argparse.ArgumentParser(
        description="FFAO ML Stage 1: data import through training and evaluation.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the pipeline (required; bare main.py only prints help).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even when outputs already exist (no Y/N prompt).",
    )
    parser.add_argument(
        "--local-file",
        type=Path,
        default=None,
        help="Existing HDF5 → import under dataset/zenodo_data/.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Force LBM generation → dataset/generated_data/ (skip Zenodo).",
    )
    parser.add_argument(
        "--lbm-fast",
        action="store_true",
        help="Short LBM run when generating (smoke tests only).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset directory (default: configs paths.dataset_root).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for CNN / ConvLSTM.",
    )
    parser.add_argument(
        "--fno-epochs",
        type=int,
        default=50,
        help="Training epochs when --with-fno is set.",
    )
    parser.add_argument(
        "--cnn-run-id",
        default="stage1_cnn",
        help="Folder name under results/runs/ for the CNN.",
    )
    parser.add_argument(
        "--baseline-run-id",
        default="stage1_baseline",
        help="Folder name under results/runs/ for baseline metrics.",
    )
    parser.add_argument(
        "--convlstm-run-id",
        default="stage1_convlstm",
        help="Folder name under results/runs/ for ConvLSTM.",
    )
    parser.add_argument(
        "--fno-run-id",
        default="stage1_fno",
        help="Folder name under results/runs/ for FNO.",
    )
    parser.add_argument(
        "--skip-cfd-validation",
        action="store_true",
        help="Skip CFD validation figures under results/cfd_validation/.",
    )
    parser.add_argument(
        "--with-convlstm",
        action="store_true",
        help="Also train ConvLSTM and write multistep_compare.json.",
    )
    parser.add_argument(
        "--with-fno",
        action="store_true",
        help="Also train and evaluate an FNO model.",
    )
    parser.add_argument(
        "--only",
        choices=phase_choices,
        help="Run a single pipeline phase.",
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=phase_choices,
        help="Start at this phase and run it plus all later phases.",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Extra Hydra overrides (e.g. train.batch_size=4).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry for the Stage 1 pipeline.

    Parameters:
        argv (list[str] | None): Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.run and not args.dry_run:
        print_usage_guide()
        return 2

    if args.only and args.start_from:
        parser.error("Use either --only or --from, not both.")

    opts = Stage1PipelineOptions(
        repo_root=REPO_ROOT,
        local_upstream=args.local_file,
        dataset_root=args.dataset_root,
        baseline_run_id=args.baseline_run_id,
        cnn_run_id=args.cnn_run_id,
        convlstm_run_id=args.convlstm_run_id,
        fno_run_id=args.fno_run_id,
        train_epochs=args.epochs,
        fno_epochs=args.fno_epochs,
        skip_cfd_validation=args.skip_cfd_validation,
        with_convlstm=args.with_convlstm,
        with_fno=args.with_fno,
        hydra_overrides=list(args.hydra_overrides),
        force=args.force,
        dry_run=args.dry_run,
        generate_data=args.generate_data,
        lbm_fast=args.lbm_fast,
    )
    only = PipelinePhase(args.only) if args.only else None
    start_from = PipelinePhase(args.start_from) if args.start_from else None

    try:
        results = run_stage1_pipeline(opts, only=only, start_from=start_from)
    except PrerequisiteError as exc:
        log_handler.error("%s", exc)
        return 1

    print_summary(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutdown_logger()
