#!/usr/bin/env python3
"""
#############################################################################
### Stage 1 Zenodo CFD validation
###
### @file validate_stage1_zenodo.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Build vorticity figures and shedding animation under
``results/cfd_validation/stage1_zenodo/`` (requires imported ``dataset/``).

Example::

    python scripts/download_stage1_zenodo.py --local-file path/to/data.h5
    python scripts/validate_stage1_zenodo.py
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.config import config_dir
from ffaoml.validation.stage1_zenodo import run_stage1_validation

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compose Hydra config and write validation artifacts.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Stage 1 Zenodo vorticity validation plots and report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/cfd_validation/stage1_zenodo",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")

    result = run_stage1_validation(
        cfg,
        output_dir=args.output_dir,
        fps=args.fps,
        max_animation_frames=args.max_frames,
    )
    print(f"Summary: {result.summary_md}")
    print(f"Metrics: {result.metrics_json}")
    if result.strouhal_number is not None:
        print(
            f"St ≈ {result.strouhal_number:.4f} "
            f"(f ≈ {result.dominant_frequency_hz:.4f} Hz)"
        )


if __name__ == "__main__":
    main()
