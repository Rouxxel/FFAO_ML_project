#!/usr/bin/env python3
"""
#############################################################################
### Stage 3 CFDBench / multi-Re grid validation
###
### @file validate_stage3_cfdbench.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Velocity / vorticity figures under ``results/cfd_validation/stage3_cfdbench/``.

Example::

    python scripts/download_stage3_cfdbench.py --max-cases 3 --local-data-root data
    python scripts/validate_stage3_cfdbench.py
    python scripts/validate_stage3_cfdbench.py --sim-id re_100_cfdbench_prop_case0000
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir
from ffaoml.validation.stage3_cfdbench import run_stage3_cfdbench_validation

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3 multi-Re grid qualitative validation plots.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sim-id", default=None, help="Single simulation id.")
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=None,
        help="Cap number of metadata rows to plot.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--dataset",
        default="stage3_cfdbench",
        choices=["stage3_cfdbench", "splits"],
        help="Hydra dataset group (use splits for stub multi-Re root).",
    )
    parser.add_argument(
        "--dataset-output-root",
        type=Path,
        default=None,
        help="Override dataset.output_root (absolute path).",
    )
    args = parser.parse_args()

    overrides = [f"dataset={args.dataset}"]
    if args.dataset_output_root is not None:
        overrides.append(f"dataset.output_root={args.dataset_output_root.as_posix()}")

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    try:
        result = run_stage3_cfdbench_validation(
            cfg,
            output_dir=args.output_dir,
            sim_id=args.sim_id,
            max_simulations=args.max_simulations,
            fps=args.fps,
            max_animation_frames=args.max_frames,
            write_animation=not args.no_animation,
        )
    except FileNotFoundError as exc:
        log_handler.error("%s", exc)
        return 1

    log_handler.info("Summary: %s", result.summary_md)
    log_handler.info("Metrics: %s", result.metrics_json)
    for sid, paths in result.per_sim_figures.items():
        log_handler.info("Figures[%s]: %s", sid, paths)
    if result.vorticity_animation is not None:
        log_handler.info("Animation: %s", result.vorticity_animation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
