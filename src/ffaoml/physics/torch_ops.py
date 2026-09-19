"""
#############################################################################
### Torch mirrors of discrete physics operators
###
### @file torch_ops.py
### @author Sebastian Russo
### @date 2026
#############################################################################

GPU-friendly operators for physics-informed training losses.
"""

# Native imports
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

"""OPERATORS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def divergence_2d_torch(
    velocity_x: torch.Tensor,
    velocity_y: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Central-difference divergence on interior cells (batched).

    Parameters:
        velocity_x (torch.Tensor): ``u``, shape ``(B, H, W)``.
        velocity_y (torch.Tensor): ``v``, same shape.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.

    Returns:
        torch.Tensor: Divergence on interior, shape ``(B, H-2, W-2)``.
    """
    _require_torch()
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    du_dx = (velocity_x[:, :, 2:] - velocity_x[:, :, :-2]) / (2.0 * dx)
    dv_dy = (velocity_y[:, 2:, :] - velocity_y[:, :-2, :]) / (2.0 * dy)
    return du_dx[:, 1:-1, :] + dv_dy[:, :, 1:-1]


def divergence_penalty(
    pred: torch.Tensor,
    *,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Mean squared interior divergence of predicted velocity (channels 0, 1).

    Parameters:
        pred (torch.Tensor): ``(B, C, H, W)`` predicted state.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    _require_torch()
    if pred.ndim != 4 or pred.size(-1) < 3 or pred.size(-2) < 3:
        return pred.new_tensor(0.0)
    div = divergence_2d_torch(pred[:, 0], pred[:, 1], dx, dy)
    return torch.mean(div**2)
