"""
#############################################################################
### Task A - flow-field reconstruction datasets
###
### @file reconstruction.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Predict full CONTRACT channels from velocity-only input at the same time index.
"""

# Native imports
from __future__ import annotations

from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.catalog import lookup_metadata_row
from ffaoml.data.loading import load_simulation_tensor, load_split_tensor
from ffaoml.ml.preprocessing import (
    PreprocessStats,
    fit_preprocess_stats,
    normalize_fields,
)
from ffaoml.ml.splits import re_split_simulation_ids

"""CONSTANTS-----------------------------------------------------------"""
VELOCITY_CHANNEL_INDICES = (
    DEFAULT_FIELD_CHANNELS.index("velocity_x"),
    DEFAULT_FIELD_CHANNELS.index("velocity_y"),
)

"""DATASET-----------------------------------------------------------"""


class FlowReconstructionDataset:
    """
    Same-time reconstruction: input ``(u, v)``, target all four channels.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str,
        stats: PreprocessStats,
    ) -> None:
        """
        Parameters:
            cfg (DictConfig): Composed Hydra config.
            split (str): Temporal or Re split name.
            stats (PreprocessStats): Normalization from training data only.
        """
        self.cfg = cfg
        self.split = split
        self.stats = stats
        self._samples: list[tuple[float, np.ndarray]] = []
        use_re = bool(cfg.dataset.get("use_re_splits", False))
        if use_re:
            root = cfg.dataset.output_root
            for sim_id in re_split_simulation_ids(cfg)[split]:
                row = lookup_metadata_row(root, sim_id)
                re_val = float(row["re"])
                raw = load_simulation_tensor(cfg, sim_id, split)
                series = normalize_fields(raw, stats)
                for t in range(series.shape[0]):
                    self._samples.append((re_val, series[t]))
        else:
            re_val = float(cfg.dataset.re)
            raw = load_split_tensor(cfg, split)
            series = normalize_fields(raw, stats)
            for t in range(series.shape[0]):
                self._samples.append((re_val, series[t]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return velocity-only input and full-field target at time ``t``.

        Parameters:
            index (int): Sample index.

        Returns:
            dict[str, Any]: Keys ``input``, ``target``, ``time_index``, ``re``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(index)
        re_val, frame = self._samples[index]
        velocity = frame[list(VELOCITY_CHANNEL_INDICES)]
        return {
            "input": velocity.astype(np.float32),
            "target": frame.astype(np.float32),
            "time_index": index,
            "re": re_val,
        }


def build_reconstruction_datasets(
    cfg: DictConfig,
    *,
    stats: PreprocessStats | None = None,
) -> dict[str, FlowReconstructionDataset]:
    """
    Build Task A datasets for train/val/test splits.

    Parameters:
        cfg (DictConfig): Composed config (``model.task: reconstruction``).
        stats (PreprocessStats | None): Pre-fitted normalization stats.

    Returns:
        dict[str, FlowReconstructionDataset]: ``train``, ``val``, ``test``.
    """
    fitted = stats if stats is not None else fit_preprocess_stats(cfg)
    out: dict[str, FlowReconstructionDataset] = {}
    for name in ("train", "val", "test"):
        if name in cfg.dataset.temporal_split:
            out[name] = FlowReconstructionDataset(cfg, name, fitted)
    return out
