"""
#############################################################################
### ML-oriented field loading
###
### @file loading.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Load CONTRACT-shaped numpy tensors from Zarr using Hydra temporal splits (Stage 1).
"""

# Native imports
from __future__ import annotations

from collections.abc import Sequence

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.catalog import open_simulation_from_config, temporal_index_range
from ffaoml.data.io import read_time_window, stack_fields_tensor

"""LOADERS-----------------------------------------------------------"""


def load_split_tensor(
    cfg: DictConfig,
    split: str,
    *,
    channels: Sequence[str] | None = None,
    sim_id: str | None = None,
) -> np.ndarray:
    """
    Load a temporal window as ``(n_time, n_channels, height, width)``.

    Parameters:
        cfg (DictConfig): Composed config with ``dataset.temporal_split``.
        split (str): ``train``, ``val``, or ``test``.
        channels (Sequence[str] | None): Channel order; defaults to CONTRACT list.
        sim_id (str | None): Optional simulation id override.

    Returns:
        np.ndarray: Float32 tensor ready for PyTorch ``Dataset`` wrappers.
    """
    start, end = temporal_index_range(cfg, split)
    dataset = open_simulation_from_config(cfg, sim_id=sim_id)
    window = read_time_window(dataset, start, end)
    return stack_fields_tensor(window, channels=channels)


def load_pooled_split_tensor(
    cfg: DictConfig,
    split: str,
    *,
    channels: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Load and concatenate trajectories for a split (multi-Re or Stage 1).

    Parameters:
        cfg (DictConfig): Composed config.
        split (str): ``train``, ``val``, or ``test``.
        channels (Sequence[str] | None): Channel order override.

    Returns:
        np.ndarray: ``(n_time_total, C, H, W)`` concatenated along time.
    """
    if bool(cfg.dataset.get("use_re_splits", False)):
        from ffaoml.ml.splits import re_split_simulation_ids

        chunks = [
            load_simulation_tensor(cfg, sim_id, split, channels=channels)
            for sim_id in re_split_simulation_ids(cfg)[split]
        ]
        if not chunks:
            raise ValueError(f"no simulations for split={split}")
        return np.concatenate(chunks, axis=0)
    return load_split_tensor(cfg, split, channels=channels)


def load_simulation_tensor(
    cfg: DictConfig,
    sim_id: str,
    split: str,
    *,
    channels: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Load one simulation's temporal window for a named split.

    Parameters:
        cfg (DictConfig): Composed config with ``dataset.temporal_split``.
        sim_id (str): ``metadata.csv`` simulation id.
        split (str): ``train``, ``val``, or ``test`` time bounds.
        channels (Sequence[str] | None): Channel order override.

    Returns:
        np.ndarray: ``(n_time, n_channels, height, width)`` float32 tensor.
    """
    return load_split_tensor(cfg, split, channels=channels, sim_id=sim_id)


def split_shape(cfg: DictConfig, split: str) -> tuple[int, int, int, int]:
    """
    Return ``(n_time, n_channels, ny, nx)`` for a split without loading all data.

    Parameters:
        cfg (DictConfig): Composed config.
        split (str): Temporal split name.

    Returns:
        tuple[int, int, int, int]: Expected batch tensor shape.
    """
    start, end = temporal_index_range(cfg, split)
    n_time = end - start
    n_channels = len(DEFAULT_FIELD_CHANNELS)
    dataset = open_simulation_from_config(cfg)
    ny = int(dataset.sizes["y"])
    nx = int(dataset.sizes["x"])
    return n_time, n_channels, ny, nx
