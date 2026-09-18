"""
#############################################################################
### Naive flow baselines
###
### @file baselines.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Persistence and linear temporal extrapolation for one-step and rollout evaluation.
"""

# Native imports
from __future__ import annotations

from typing import Literal

# Third-party imports
import numpy as np

"""TYPES-----------------------------------------------------------"""
BaselineName = Literal["persistence", "linear"]

"""PREDICTORS-----------------------------------------------------------"""


def predict_persistence(current: np.ndarray) -> np.ndarray:
    """
    One-step persistence: ``x̂_{t+1} = x_t``.

    Parameters:
        current (np.ndarray): State at time ``t``, shape ``(C, H, W)``.

    Returns:
        np.ndarray: Predicted next state (copy of ``current``).
    """
    return np.asarray(current, dtype=np.float32).copy()


def predict_linear(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    Linear extrapolation: ``x̂_{t+1} = 2 x_t − x_{t−1}``.

    Parameters:
        previous (np.ndarray): State at ``t-1``.
        current (np.ndarray): State at ``t``.

    Returns:
        np.ndarray: Predicted next state.
    """
    return (2.0 * current - previous).astype(np.float32)


def predict_next(
    name: BaselineName,
    current: np.ndarray,
    previous: np.ndarray | None = None,
) -> np.ndarray:
    """
    Dispatch a baseline one-step predictor.

    Parameters:
        name (BaselineName): ``persistence`` or ``linear``.
        current (np.ndarray): State at time ``t``.
        previous (np.ndarray | None): Required for ``linear``.

    Returns:
        np.ndarray: One-step prediction.

    Raises:
        ValueError: If ``linear`` is requested without ``previous``.
    """
    if name == "persistence":
        return predict_persistence(current)
    if name == "linear":
        if previous is None:
            raise ValueError("linear baseline requires previous frame")
        return predict_linear(previous, current)
    raise ValueError(f"unknown baseline: {name}")
