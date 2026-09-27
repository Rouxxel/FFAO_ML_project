#!/usr/bin/env python3
"""
#############################################################################
### Stage 2 end-to-end pipeline (script entry)
###
### @file run_stage2_pipeline.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import MeshGraphNets data → mesh validation → train → eval.

Requires TensorFlow for the **import** phase (TFRecord parse). Training defaults to CPU;
use ``--device cuda`` or Hydra overrides for GPU.

Example::

    python scripts/run_stage2_pipeline.py --dry-run
    python scripts/run_stage2_pipeline.py --run --max-trajectories 2
    python scripts/run_stage2_pipeline.py --run --only train_meshgn --epochs 10
"""

# Native imports
from __future__ import annotations

import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler, shutdown_logger
from ffaoml.pipeline.stage1 import PrerequisiteError
from ffaoml.pipeline.stage2 import (
    Stage2PipelineOptions,
    Stage2PipelinePhase,
    print_summary,
    run_stage2_pipeline,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    phase_choices = [p.value for p in Stage2PipelinePhase]
    parser = argparse.ArgumentParser(
        description="FFAO ML Stage 2: MeshGraphNets import through train/eval.",
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
        "--mesh-run-id",
        default="stage2_meshgn",
        help="Folder under results/runs/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for MeshGraphNet.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap trajectories per TFRecord split (smoke imports).",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        dest="import_splits",
        help="Import shard(s); default all three.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-fetch TFRecords under .cache/meshgraphnets_cylinder/.",
    )
    parser.add_argument(
        "--skip-mesh-validation",
        action="store_true",
        help="Skip qualitative mesh figures under results/cfd_validation/.",
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
            "No action: pass --dry-run to preview or --run to execute Stage 2."
        )
        return 2

    if args.only and args.start_from:
        parser.error("Use either --only or --from, not both.")

    splits = tuple(args.import_splits or ("train", "val", "test"))
    opts = Stage2PipelineOptions(
        repo_root=REPO_ROOT,
        mesh_run_id=args.mesh_run_id,
        train_epochs=args.epochs,
        import_splits=splits,
        max_trajectories_per_split=args.max_trajectories,
        force_download=args.force_download,
        skip_mesh_validation=args.skip_mesh_validation,
        hydra_overrides=list(args.hydra_overrides),
        force=args.force,
        dry_run=args.dry_run,
    )
    only = Stage2PipelinePhase(args.only) if args.only else None
    start_from = Stage2PipelinePhase(args.start_from) if args.start_from else None

    try:
        results = run_stage2_pipeline(opts, only=only, start_from=start_from)
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
