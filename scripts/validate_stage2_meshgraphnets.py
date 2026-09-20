#!/usr/bin/env python3
"""
#############################################################################
### Stage 2 mesh CFD validation
###
### @file validate_stage2_meshgraphnets.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Mesh + |v| figures under ``results/cfd_validation/stage2_meshgraphnets/``.

Example::

    python scripts/download_stage2_meshgraphnets.py --split train --max-trajectories 1
    python scripts/validate_stage2_meshgraphnets.py
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir
from ffaoml.validation.stage2_meshgraphnets import run_stage2_mesh_validation

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 meshgraphnets qualitative validation plots.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sim-id", default=None, help="Trajectory folder name.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["dataset=stage2_meshgraphnets"],
        )

    result = run_stage2_mesh_validation(
        cfg,
        output_dir=args.output_dir,
        sim_id=args.sim_id,
        fps=args.fps,
        max_animation_frames=args.max_frames,
        write_animation=not args.no_animation,
    )
    log_handler.info("Summary: %s", result.summary_md)
    log_handler.info("Metrics: %s", result.metrics_json)
    log_handler.info("Snapshots: %s", result.mesh_velocity_snapshots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
