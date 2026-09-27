"""
#############################################################################
### Stage 1 on-disk dataset layout
###
### @file stage1_layout.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Resolve ``dataset/zenodo_data`` vs ``dataset/generated_data`` (and legacy flat
``dataset/``) for imports and ML loaders.
"""

# Native imports
from __future__ import annotations

from pathlib import Path

# Project imports
from ffaoml.data.io import FIELD_STORE_NAME
from ffaoml.manifests import dataset_manifest_path

"""CONSTANTS-----------------------------------------------------------"""
DATASET_BASE_NAME = "dataset"
ZENODO_DATA_DIR = "zenodo_data"
GENERATED_DATA_DIR = "generated_data"
DEFAULT_SIMULATION_ID = "re_100_zenodo"


def dataset_base(repo_root: str | Path) -> Path:
    """Repository ``dataset/`` parent directory."""
    return Path(repo_root) / DATASET_BASE_NAME


def zenodo_dataset_root(repo_root: str | Path) -> Path:
    """``dataset/zenodo_data``- Zenodo or cached/local official HDF5 imports."""
    return dataset_base(repo_root) / ZENODO_DATA_DIR


def generated_dataset_root(repo_root: str | Path) -> Path:
    """``dataset/generated_data``- LBM fallback imports."""
    return dataset_base(repo_root) / GENERATED_DATA_DIR


def import_complete(
    dataset_root: Path,
    simulation_id: str = DEFAULT_SIMULATION_ID,
) -> bool:
    """True when manifest and ``fields.zarr`` exist under *dataset_root*."""
    manifest = dataset_manifest_path(dataset_root)
    zarr_dir = dataset_root / "simulations" / simulation_id / FIELD_STORE_NAME
    return manifest.is_file() and zarr_dir.is_dir()


def resolve_active_dataset_root(
    repo_root: str | Path,
    *,
    simulation_id: str = DEFAULT_SIMULATION_ID,
) -> Path | None:
    """
    Pick an existing Stage 1 dataset root for training (newest layout first).

    Returns:
        Path | None: First complete import among zenodo, generated, legacy flat.
    """
    base = dataset_base(repo_root)
    candidates = (
        zenodo_dataset_root(repo_root),
        generated_dataset_root(repo_root),
        base,
    )
    for root in candidates:
        if import_complete(root, simulation_id=simulation_id):
            return root
    return None


def default_import_target(
    repo_root: str | Path,
    *,
    prefer_generated: bool = False,
) -> Path:
    """Where the next import should write when no override is set."""
    if prefer_generated:
        return generated_dataset_root(repo_root)
    return zenodo_dataset_root(repo_root)
