"""
#############################################################################
### Field prediction metrics
###
### @file metrics.py
### @author Sebastian Russo
### @date 2026
#############################################################################

MSE and relative L² for tensor fields (PRD §8 / ARCHITECTURE §7).
"""

# Native imports
from __future__ import annotations

# Third-party imports
import numpy as np

"""METRICS-----------------------------------------------------------"""


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Mean squared error over all elements.

    Parameters:
        pred (np.ndarray): Predicted field(s).
        true (np.ndarray): Ground truth (same shape).

    Returns:
        float: Scalar MSE.

    Raises:
        ValueError: On shape mismatch.
    """
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    diff = pred.astype(np.float64) - true.astype(np.float64)
    return float(np.mean(diff * diff))


def relative_l2(pred: np.ndarray, true: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Relative L² error ‖pred − true‖₂ / ‖true‖₂.

    Parameters:
        pred (np.ndarray): Predicted field(s).
        true (np.ndarray): Ground truth (same shape).
        eps (float): Denominator floor.

    Returns:
        float: Relative L² norm.

    Raises:
        ValueError: On shape mismatch.
    """
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    diff = pred.astype(np.float64) - true.astype(np.float64)
    num = np.linalg.norm(diff.ravel())
    denom = max(np.linalg.norm(true.astype(np.float64).ravel()), eps)
    return float(num / denom)


def metric_dict(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """
    Compute standard metrics for one prediction pair.

    Parameters:
        pred (np.ndarray): Predicted field.
        true (np.ndarray): Ground truth.

    Returns:
        dict[str, float]: ``mse`` and ``relative_l2`` keys.
    """
    return {
        "mse": mse(pred, true),
        "relative_l2": relative_l2(pred, true),
    }
