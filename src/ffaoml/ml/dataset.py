"""
#############################################################################
### FlowDataset — one-step prediction windows
###
### @file dataset.py
### @author Sebastian Russo
### @date 2026
#############################################################################

PyTorch-compatible samples: flow at time ``t`` → flow at ``t + Δt`` on the Stage 1
temporal split (``use_re_splits: false``).
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.data.loading import load_split_tensor
from ffaoml.ml.preprocessing import (
    PreprocessStats,
    fit_preprocess_stats,
    normalize_fields,
)

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class FlowSample:
    """One (input, target) pair at a physical time index."""

    input: np.ndarray
    target: np.ndarray
    time_index: int
    re: float


"""DATASET-----------------------------------------------------------"""


class FlowDataset:
    """
    One-step field prediction dataset for a temporal split.

    Normalization stats must be fit on the training split only and shared across
    train/val/test ``FlowDataset`` instances.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str,
        stats: PreprocessStats,
        *,
        delta_steps: int = 1,
    ) -> None:
        """
        Parameters:
            cfg (DictConfig): Composed Hydra config.
            split (str): ``train``, ``val``, or ``test`` temporal window.
            stats (PreprocessStats): Normalization from training data only.
            delta_steps (int): Prediction horizon in time indices (default 1).
        """
        if bool(cfg.dataset.get("use_re_splits", False)):
            raise NotImplementedError(
                "FlowDataset for use_re_splits is deferred to Stage 3 / own CFD."
            )
        if delta_steps < 1:
            raise ValueError("delta_steps must be >= 1")
        self.cfg = cfg
        self.split = split
        self.stats = stats
        self.delta_steps = delta_steps
        self.re = float(cfg.dataset.re)
        raw = load_split_tensor(cfg, split)
        self._series = normalize_fields(raw, stats)
        self._start_index = 0

    def __len__(self) -> int:
        return max(0, self._series.shape[0] - self.delta_steps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return a sample dict suitable for PyTorch ``DataLoader``.

        Parameters:
            index (int): Sample index within the split.

        Returns:
            dict[str, Any]: Keys ``input``, ``target``, ``time_index``, ``re``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(index)
        t = index
        inp = self._series[t]
        tgt = self._series[t + self.delta_steps]
        sample = FlowSample(
            input=inp,
            target=tgt,
            time_index=t + self._start_index,
            re=self.re,
        )
        return {
            "input": sample.input.astype(np.float32),
            "target": sample.target.astype(np.float32),
            "time_index": sample.time_index,
            "re": sample.re,
        }


class FlowUnrollDataset:
    """
    One input frame and a stack of ``unroll_steps`` target frames for ConvLSTM.

    Shares normalization with ``FlowDataset`` on the same temporal split.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str,
        stats: PreprocessStats,
        *,
        unroll_steps: int = 1,
    ) -> None:
        """
        Parameters:
            cfg (DictConfig): Composed Hydra config.
            split (str): ``train``, ``val``, or ``test``.
            stats (PreprocessStats): Normalization from training data only.
            unroll_steps (int): Number of future frames in each sample.
        """
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        self.unroll_steps = unroll_steps
        self.re = float(cfg.dataset.re)
        raw = load_split_tensor(cfg, split)
        self._series = normalize_fields(raw, stats)

    def __len__(self) -> int:
        return max(0, self._series.shape[0] - self.unroll_steps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return input at ``t`` and targets ``t+1..t+unroll_steps``.

        Parameters:
            index (int): Sample index within the split.

        Returns:
            dict[str, Any]: Keys ``input``, ``targets``, ``time_index``, ``re``.
        """
        if index < 0 or index >= len(self):
            raise IndexError(index)
        t = index
        targets = self._series[t + 1 : t + 1 + self.unroll_steps]
        return {
            "input": self._series[t].astype(np.float32),
            "targets": targets.astype(np.float32),
            "time_index": t,
            "re": self.re,
        }


def build_flow_datasets(
    cfg: DictConfig,
    *,
    delta_steps: int = 1,
    stats: PreprocessStats | None = None,
) -> dict[str, FlowDataset]:
    """
    Build train/val/test ``FlowDataset`` objects with shared normalization.

    Parameters:
        cfg (DictConfig): Composed config.
        delta_steps (int): Steps between input and target frames.
        stats (PreprocessStats | None): Pre-fitted stats; computed on train if omitted.

    Returns:
        dict[str, FlowDataset]: Keys ``train``, ``val``, ``test``.
    """
    fitted = stats if stats is not None else fit_preprocess_stats(cfg)
    out: dict[str, FlowDataset] = {}
    for name in ("train", "val", "test"):
        if name in cfg.dataset.temporal_split:
            out[name] = FlowDataset(
                cfg,
                name,
                fitted,
                delta_steps=delta_steps,
            )
    return out


def build_flow_unroll_datasets(
    cfg: DictConfig,
    *,
    unroll_steps: int,
    stats: PreprocessStats | None = None,
) -> dict[str, FlowUnrollDataset]:
    """
    Build train/val/test ``FlowUnrollDataset`` objects with shared normalization.

    Parameters:
        cfg (DictConfig): Composed config.
        unroll_steps (int): Future frames per sample.
        stats (PreprocessStats | None): Pre-fitted stats; computed on train if omitted.

    Returns:
        dict[str, FlowUnrollDataset]: Keys ``train``, ``val``, ``test``.
    """
    fitted = stats if stats is not None else fit_preprocess_stats(cfg)
    out: dict[str, FlowUnrollDataset] = {}
    for name in ("train", "val", "test"):
        if name in cfg.dataset.temporal_split:
            out[name] = FlowUnrollDataset(
                cfg,
                name,
                fitted,
                unroll_steps=unroll_steps,
            )
    return out
