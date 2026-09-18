"""
#############################################################################
### Training losses
###
### @file losses.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Field reconstruction losses for one-step predictors (PRD §14 Exp. 5 hooks).
"""

# Native imports
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment,misc]

"""LOSSES-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def field_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error over batch and spatial dimensions.

    Parameters:
        pred (torch.Tensor): Model output ``(B, C, H, W)``.
        target (torch.Tensor): Ground truth next frame.

    Returns:
        torch.Tensor: Scalar loss.
    """
    _require_torch()
    return torch.mean((pred - target) ** 2)


def build_loss_fn(field_mse: bool = True) -> nn.Module:
    """
    Return the primary training loss module.

    Parameters:
        field_mse (bool): When true, use ``nn.MSELoss`` (same as field_mse_loss).

    Returns:
        nn.Module: Callable loss accepting ``(pred, target)``.
    """
    _require_torch()
    if field_mse:
        return nn.MSELoss()
    raise ValueError("at least one loss term must be enabled")
