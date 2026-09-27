#!/usr/bin/env python3
"""
#############################################################################
### Download and import Stage 2 MeshGraphNets data
###
### @file download_stage2_meshgraphnets.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fetch ``cylinder_flow`` TFRecord shards (when missing) and export trajectories under
``dataset/meshgraphnets_data/``.

Requires TensorFlow to parse TFRecords.

Example::

    python scripts/download_stage2_meshgraphnets.py --split train --max-trajectories 1
    python scripts/download_stage2_meshgraphnets.py --split train --split val \\
        --force-download
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir
from ffaoml.data.sources.meshgraphnets_cylinder import run_meshgraphnets_import

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import MeshGraphNets cylinder_flow trajectories.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        dest="splits",
        help="Shard to import (repeatable). Default: all three.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap trajectories per split (smoke tests).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download TFRecords even when cached under .cache/.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional meta.json next to cache (else built-in template).",
    )
    args = parser.parse_args()
    splits = tuple(args.splits or ("train", "val", "test"))

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["dataset=stage2_meshgraphnets"],
        )

    try:
        result = run_meshgraphnets_import(
            cfg,
            splits=splits,
            max_trajectories_per_split=args.max_trajectories,
            repo_root=REPO_ROOT,
            force_download=args.force_download,
            meta_path=args.meta,
        )
    except ImportError as exc:
        log_handler.error("%s", exc)
        return 1

    log_handler.info(
        "Imported %d trajectories into %s (manifest %s)",
        result.trajectories_imported,
        result.dataset_root,
        result.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
