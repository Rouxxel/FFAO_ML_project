#!/usr/bin/env python3
"""
#############################################################################
### Download and import Stage 1 Zenodo data
###
### @file download_stage1_zenodo.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fetch the Re≈100 cylinder trajectory from Zenodo (record 18669296 by default),
convert to ``dataset/simulations/re_100_zenodo/fields.zarr``, and write
``metadata.csv`` plus ``dataset/manifest.json``.

Example::

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
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config

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
    args = parser.parse_args()

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name=args.config_name)

    result = import_stage1_from_config(
        cfg,
        local_upstream=args.local_file,
        repo_root=REPO_ROOT,
    )
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
