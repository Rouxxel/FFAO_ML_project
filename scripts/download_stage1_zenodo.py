#!/usr/bin/env python3
"""
#############################################################################
### Download and import Stage 1 Zenodo data
###
### @file download_stage1_zenodo.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Resolve upstream ``cylinder_re100_grid64_last100.h5`` (Zenodo, cache, or
``--local-file``), convert to ``dataset/simulations/re_100_zenodo/fields.zarr``,
and write ``metadata.csv`` plus ``dataset/manifest.json``.

If Zenodo has no HDF5 attachment, run ``scripts/generate_stage1_cylinder_h5.py``
first (writes ``.cache/zenodo_stage1/``).

Example::

    python scripts/generate_stage1_cylinder_h5.py
    python scripts/download_stage1_zenodo.py
    python scripts/download_stage1_zenodo.py --local-file path/to/data.h5
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir
from ffaoml.data.sources.stage1_import import Stage1ImportMode, run_stage1_import

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Compose ``dataset=stage1_zenodo`` and run the import pipeline.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Import Stage 1 Zenodo Re=100 data.")
    parser.add_argument(
        "--local-file",
        type=Path,
        default=None,
        help="Use an existing upstream file instead of downloading from Zenodo.",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra root config name (default: config).",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Force LBM fallback (dataset/generated_data/).",
    )
    parser.add_argument(
        "--lbm-fast",
        action="store_true",
        help="Short LBM when generating (smoke only).",
    )
    args = parser.parse_args()

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name=args.config_name)

    mode = Stage1ImportMode.GENERATED if args.generate_data else Stage1ImportMode.AUTO
    result, dataset_root = run_stage1_import(
        cfg,
        REPO_ROOT,
        mode=mode,
        local_upstream=args.local_file,
        lbm_fast=args.lbm_fast,
    )
    log_handler.info("Dataset root: %s", dataset_root)
    log_handler.info(
        "Imported %s steps on %sx%s grid.",
        result.n_steps,
        result.ny,
        result.nx,
    )
    log_handler.info("Zarr store: %s", result.store_path)
    log_handler.info("Manifest: %s", result.manifest_path)
    log_handler.info("Upstream: %s", result.upstream_path)


if __name__ == "__main__":
    main()
