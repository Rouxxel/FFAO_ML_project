"""
#############################################################################
### Checkpoint bundle helpers
###
### @file bundle.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Finalize training runs with reproducibility metadata
(``documentation/REPRODUCIBILITY.md``).
"""

# Native imports
from __future__ import annotations

import shutil
from pathlib import Path

# Third-party imports
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import dataset_root_from_config
from ffaoml.manifests import (
    RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME,
    dataset_manifest_path,
    write_checkpoint_bundle_manifest,
)

"""BUNDLE-----------------------------------------------------------"""


def copy_dataset_manifest_snapshot(
    cfg: DictConfig,
    run_dir: str | Path,
) -> Path | None:
    """
    Copy ``<dataset_root>/manifest.json`` into the run dir as ``dataset_manifest.json``.

    Returns:
        Path | None: Destination path when the source manifest exists.
    """
    src = dataset_manifest_path(dataset_root_from_config(cfg))
    if not src.is_file():
        return None
    dest = Path(run_dir) / RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME
    shutil.copy2(src, dest)
    return dest


def finalize_training_bundle(
    cfg: DictConfig,
    run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write ``bundle_manifest.json`` after config and dataset manifest are in place.

    Parameters:
        cfg (DictConfig): Resolved training config (includes ``seed``).
        run_dir (str | Path): Training output directory.
        repo_root (str | Path | None): For ``git_commit`` in the bundle manifest.

    Returns:
        Path: Path to ``bundle_manifest.json``.
    """
    return write_checkpoint_bundle_manifest(run_dir, cfg, repo_root=repo_root)
