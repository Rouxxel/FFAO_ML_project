"""
#############################################################################
### Discrete Navier–Stokes operators
###
### @file navier_stokes.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Central-difference operators on uniform structured 2D grids for validation and
physics-informed losses.
"""

# Native imports
from __future__ import annotations

# Third-party imports
import numpy as np

"""TYPES-----------------------------------------------------------"""
Array2D = np.ndarray

"""OPERATORS-----------------------------------------------------------"""


def divergence_2d(
    velocity_x: Array2D,
    velocity_y: Array2D,
    dx: float,
    dy: float,
    *,
    solid_mask: Array2D | None = None,
) -> Array2D:
    """
    Central-difference divergence ∇·u on a uniform structured grid.

    Interior stencil (indices with full neighbours):

        ∂u/∂x ≈ (u[i, j+1] - u[i, j-1]) / (2 dx)
        ∂v/∂y ≈ (v[i+1, j] - v[i-1, j]) / (2 dy)

    Boundary rows/columns are ``nan`` because neighbours are not stored in
    typical CFD exports.

    Parameters:
        velocity_x (Array2D): ``u`` component, shape ``(ny, nx)``.
        velocity_y (Array2D): ``v`` component, same shape as ``velocity_x``.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        solid_mask (Array2D | None): If set, ``True`` marks solid cells (output
            forced to ``nan`` there).

    Returns:
        Array2D: Divergence field, shape ``(ny, nx)``.

    Raises:
        ValueError: On shape mismatch or non-positive spacing.
    """
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    if velocity_x.shape != velocity_y.shape:
        raise ValueError("velocity_x and velocity_y must have the same shape")
    if velocity_x.ndim != 2:
        raise ValueError("velocity fields must be 2D arrays")

    ny, nx = velocity_x.shape
    div = np.full((ny, nx), np.nan, dtype=float)

    if ny < 3 or nx < 3:
        return div

    du_dx = (velocity_x[:, 2:] - velocity_x[:, :-2]) / (2.0 * dx)
    dv_dy = (velocity_y[2:, :] - velocity_y[:-2, :]) / (2.0 * dy)
    div[1:-1, 1:-1] = du_dx[1:-1, :] + dv_dy[:, 1:-1]

    if solid_mask is not None:
        if solid_mask.shape != velocity_x.shape:
            raise ValueError("solid_mask must match velocity field shape")
        div = np.where(solid_mask, np.nan, div)

    return div
