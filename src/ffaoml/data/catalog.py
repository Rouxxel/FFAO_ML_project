"""
#############################################################################
### Dataset catalog and path resolution
###
### @file catalog.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Resolve ``dataset/`` paths from Hydra config and ``metadata.csv`` so loaders do
not hard-code simulation directories.
"""

# Native imports
from __future__ import annotations

from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Project imports
from ffaoml.data.io import open_field_store
from ffaoml.data.metadata import metadata_csv_path, read_metadata_rows

"""PATHS-----------------------------------------------------------"""


def dataset_root_from_config(cfg: DictConfig) -> Path:
    """
    Return the configured dataset root directory.

    Parameters:
        cfg (DictConfig): Composed config with ``dataset.output_root``.

    Returns:
        Path: Dataset root (typically ``dataset/``).
    """
    return Path(str(cfg.dataset.output_root))


def simulation_id_from_config(cfg: DictConfig) -> str:
    """Return ``dataset.simulation_id`` for Stage 1 single-trajectory layouts."""
    return str(cfg.dataset.simulation_id)


def simulation_directory(
    dataset_root: str | Path,
    sim_id: str,
) -> Path:
    """
    Absolute path to one simulation export folder.

    Parameters:
        dataset_root (str | Path): ``dataset/`` root.
        sim_id (str): Row id (e.g. ``re_100_zenodo``).

    Returns:
        Path: ``<dataset_root>/simulations/<sim_id>/``.
    """
    return Path(dataset_root) / "simulations" / sim_id


def resolve_simulation_dir(cfg: DictConfig, sim_id: str | None = None) -> Path:
    """
    Resolve the simulation directory from config (and optional override id).

    Parameters:
        cfg (DictConfig): Composed Hydra config.
        sim_id (str | None): Override; defaults to ``cfg.dataset.simulation_id``.

    Returns:
        Path: Simulation directory containing ``fields.zarr``.
    """
    root = dataset_root_from_config(cfg)
    sid = sim_id or simulation_id_from_config(cfg)
    return simulation_directory(root, sid)


"""SPLITS-----------------------------------------------------------"""


def temporal_index_range(cfg: DictConfig, split: str) -> tuple[int, int]:
    """
    Return ``[start, end)`` time indices for a named temporal split.

    Parameters:
        cfg (DictConfig): Config with ``dataset.temporal_split``.
        split (str): ``train``, ``val``, or ``test``.

    Returns:
        tuple[int, int]: Inclusive start, exclusive end.

    Raises:
        KeyError: If ``split`` is not defined.
    """
    temporal = cfg.dataset.temporal_split
    if split not in temporal:
        raise KeyError(f"unknown temporal split '{split}'")
    bounds = temporal[split]
    return int(bounds[0]), int(bounds[1])


"""METADATA-----------------------------------------------------------"""


def lookup_metadata_row(
    dataset_root: str | Path,
    sim_id: str,
) -> dict[str, Any]:
    """
    Find one ``metadata.csv`` row by ``sim_id``.

    Parameters:
        dataset_root (str | Path): Dataset root.
        sim_id (str): Simulation identifier.

    Returns:
        dict[str, Any]: Row values.

    Raises:
        KeyError: If no matching row exists.
    """
    for row in read_metadata_rows(dataset_root):
        if str(row["sim_id"]) == sim_id:
            return row
    raise KeyError(f"sim_id {sim_id!r} not found in {metadata_csv_path(dataset_root)}")


"""LOADING-----------------------------------------------------------"""


def open_simulation_from_config(
    cfg: DictConfig,
    sim_id: str | None = None,
):
    """
    Open ``fields.zarr`` for the configured Stage 1 simulation.

    Parameters:
        cfg (DictConfig): Composed Hydra config.
        sim_id (str | None): Optional override simulation id.

    Returns:
        xarray.Dataset: Lazy field dataset.
    """
    sim_dir = resolve_simulation_dir(cfg, sim_id=sim_id)
    return open_field_store(sim_dir)
