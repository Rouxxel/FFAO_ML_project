"""
#############################################################################
### Physical-condition (Reynolds) conditioning
###
### @file conditioning.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Embed normalized Reynolds number into CNN inputs (Task D / Stage 3 splits).
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass

# Third-party imports
import numpy as np
from omegaconf import DictConfig

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class ReScaling:
    """Affine map from physical Re to ``[-1, 1]`` using training Reynolds only."""

    re_min: float
    re_max: float

    def scale(self, re: float) -> float:
        """
        Normalize a Reynolds number.

        Parameters:
            re (float): Physical Re.

        Returns:
            float: Value in ``[-1, 1]`` (0 when train range is degenerate).
        """
        if self.re_max <= self.re_min:
            return 0.0
        return 2.0 * (float(re) - self.re_min) / (self.re_max - self.re_min) - 1.0


"""FIT-----------------------------------------------------------"""


def fit_re_scaling(cfg: DictConfig) -> ReScaling:
    """
    Fit Re bounds on ``dataset.train_re`` (no val/test leakage).

    Parameters:
        cfg (DictConfig): ``dataset=splits`` config with ``train_re``.

    Returns:
        ReScaling: Bounds for ``scale`` and channel broadcast.
    """
    train_res = [float(r) for r in cfg.dataset.train_re]
    if not train_res:
        raise ValueError("train_re must be non-empty for Re conditioning")
    return ReScaling(re_min=min(train_res), re_max=max(train_res))


def condition_on_re_enabled(cfg: DictConfig) -> bool:
    """
    Return whether the model config requests Re conditioning.

    Parameters:
        cfg (DictConfig): Composed config with ``model`` group.

    Returns:
        bool: True when ``model.condition_on_re`` is set.
    """
    return bool(cfg.model.get("condition_on_re", False))


"""NUMPY-----------------------------------------------------------"""


def append_re_channel(
    fields: np.ndarray,
    re_scaled: float,
) -> np.ndarray:
    """
    Concatenate a spatially constant Re channel to a field tensor.

    Parameters:
        fields (np.ndarray): ``(C, H, W)`` or ``(T, C, H, W)``.
        re_scaled (float): Normalized Re in ``[-1, 1]``.

    Returns:
        np.ndarray: Tensor with one extra leading channel dimension.
    """
    if fields.ndim == 3:
        _, h, w = fields.shape
        extra = np.full((1, h, w), re_scaled, dtype=np.float32)
        return np.concatenate([fields.astype(np.float32), extra], axis=0)
    if fields.ndim == 4:
        t, _, h, w = fields.shape
        extra = np.full((t, 1, h, w), re_scaled, dtype=np.float32)
        return np.concatenate([fields.astype(np.float32), extra], axis=1)
    raise ValueError(f"expected 3D or 4D fields, got shape {fields.shape}")
