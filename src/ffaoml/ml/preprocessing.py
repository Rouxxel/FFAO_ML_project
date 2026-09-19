"""
#############################################################################
### Field preprocessing and normalization
###
### @file preprocessing.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fit per-channel mean and std on the **training time window only** (Stage 1) or
training Reynolds list (Stage 3), then apply to all splits without leakage.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.catalog import temporal_index_range
from ffaoml.data.loading import load_simulation_tensor, load_split_tensor
from ffaoml.ml.splits import re_split_simulation_ids

"""CONSTANTS-----------------------------------------------------------"""
STATS_FILENAME = "preprocess_stats.json"

"""TYPES-----------------------------------------------------------"""


@dataclass
class PreprocessStats:
    """Per-channel normalization fit on the training split only."""

    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fit_split: str
    time_start: int
    time_end: int
    schema_version: int = 1
    re_min: float | None = None
    re_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``preprocess_stats.json``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessStats:
        """
        Load stats from a JSON-compatible dict.

        Parameters:
            data (dict[str, Any]): Parsed ``preprocess_stats.json``.

        Returns:
            PreprocessStats: Validated instance.
        """
        re_min = data.get("re_min")
        re_max = data.get("re_max")
        return cls(
            channels=tuple(data["channels"]),
            mean=tuple(float(x) for x in data["mean"]),
            std=tuple(float(x) for x in data["std"]),
            fit_split=str(data["fit_split"]),
            time_start=int(data["time_start"]),
            time_end=int(data["time_end"]),
            schema_version=int(data.get("schema_version", 1)),
            re_min=float(re_min) if re_min is not None else None,
            re_max=float(re_max) if re_max is not None else None,
        )


"""FIT / APPLY-----------------------------------------------------------"""


def fit_preprocess_stats(cfg: DictConfig) -> PreprocessStats:
    """
    Estimate per-channel mean and std on the configured training split.

    Parameters:
        cfg (DictConfig): Composed Hydra config.

    Returns:
        PreprocessStats: Statistics for ``normalize_fields``.
    """
    fit_split = str(cfg.dataset.normalization.fit_split)
    start, end = temporal_index_range(cfg, fit_split)
    if bool(cfg.dataset.get("use_re_splits", False)):
        from ffaoml.ml.conditioning import fit_re_scaling

        sim_ids = re_split_simulation_ids(cfg)[fit_split]
        if not sim_ids:
            raise ValueError(f"no train simulations for fit_split={fit_split}")
        chunks = [load_simulation_tensor(cfg, sim_id, fit_split) for sim_id in sim_ids]
        tensor = np.concatenate(chunks, axis=0)
        re_bounds = fit_re_scaling(cfg)
        re_min, re_max = re_bounds.re_min, re_bounds.re_max
    else:
        tensor = load_split_tensor(cfg, fit_split)
        re_min, re_max = None, None
    mean = tensor.mean(axis=(0, 2, 3))
    std = tensor.std(axis=(0, 2, 3)) + 1e-8
    return PreprocessStats(
        channels=DEFAULT_FIELD_CHANNELS,
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in std),
        fit_split=fit_split,
        time_start=start,
        time_end=end,
        re_min=re_min,
        re_max=re_max,
    )


def normalize_fields(
    tensor: np.ndarray,
    stats: PreprocessStats,
) -> np.ndarray:
    """
    Apply per-channel normalization to ``(T, C, H, W)`` or ``(C, H, W)`` data.

    Parameters:
        tensor (np.ndarray): Field tensor.
        stats (PreprocessStats): Fitted statistics.

    Returns:
        np.ndarray: Normalized array (same shape).
    """
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1, 1, 1)
    if tensor.ndim == 4:
        mean = mean[None, ...]
        std = std[None, ...]
    return (tensor - mean) / std


"""PERSIST-----------------------------------------------------------"""


def save_preprocess_stats(path: str | Path, stats: PreprocessStats) -> Path:
    """
    Write ``preprocess_stats.json`` for checkpoint bundles.

    Parameters:
        path (str | Path): Destination file.
        stats (PreprocessStats): Fitted stats.

    Returns:
        Path: Written file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_preprocess_stats(path: str | Path) -> PreprocessStats:
    """
    Load stats from ``preprocess_stats.json``.

    Parameters:
        path (str | Path): JSON file path.

    Returns:
        PreprocessStats: Parsed stats.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PreprocessStats.from_dict(data)
