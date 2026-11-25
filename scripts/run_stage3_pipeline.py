#!/usr/bin/env python3
"""
#############################################################################
### Stage 3 end-to-end pipeline (script entry)
###
### @file run_stage3_pipeline.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import CFDBench → grid validation → Re-conditioned CNN train → Re generalization eval.

Requires ``huggingface_hub`` for download unless ``--local-data-root`` points at an
existing CFDBench ``data/`` tree. Training defaults to CPU; use Hydra
``train.device=cuda`` for GPU.

Example::

    python scripts/run_stage3_pipeline.py --dry-run
    python scripts/run_stage3_pipeline.py --run --max-cases 20
    python scripts/run_stage3_pipeline.py --run --only train_cnn_re --epochs 10
"""

# Native imports
from __future__ import annotations

import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler, shutdown_logger
from ffaoml.pipeline.stage1 import PrerequisiteError
from ffaoml.pipeline.stage3 import (
    Stage3PipelineOptions,
    Stage3PipelinePhase,
    print_summary,
    run_stage3_pipeline,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    phase_choices = [p.value for p in Stage3PipelinePhase]
    parser = argparse.ArgumentParser(
        description="FFAO ML Stage 3: CFDBench import through Re generalization eval.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the pipeline (required).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even when outputs already exist.",
    )
    parser.add_argument(
        "--run-id",
        default="stage3_cnn_re",
        dest="cnn_run_id",
        help="Folder under results/runs/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the Re-conditioned CNN.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap CFDBench cases imported this run.",
    )
    parser.add_argument(
        "--re",
        dest="re_filter",
        action="append",
        type=float,
        help="Only import Reynolds near this value (repeatable).",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help="Skip HF download; use existing CFDBench data root.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-fetch CFDBench snapshot under .cache/cfdbench/.",
    )
    parser.add_argument(
        "--skip-grid-validation",
        action="store_true",
        help="Skip qualitative grid figures under results/cfd_validation/.",
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
        help="Extra Hydra overrides (e.g. train.device=cuda).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.run and not args.dry_run:
        log_handler.warning(
            "No action: pass --dry-run to preview or --run to execute Stage 3."
        )
        return 2

    if args.only and args.start_from:
        parser.error("Use either --only or --from, not both.")

    opts = Stage3PipelineOptions(
        repo_root=REPO_ROOT,
        cnn_run_id=args.cnn_run_id,
        train_epochs=args.epochs,
        max_cases=args.max_cases,
        re_filter=args.re_filter,
        local_data_root=args.local_data_root,
        force_download=args.force_download,
        skip_grid_validation=args.skip_grid_validation,
        hydra_overrides=list(args.hydra_overrides),
        force=args.force,
        dry_run=args.dry_run,
    )
    only = Stage3PipelinePhase(args.only) if args.only else None
    start_from = Stage3PipelinePhase(args.start_from) if args.start_from else None

    try:
        results = run_stage3_pipeline(opts, only=only, start_from=start_from)
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
