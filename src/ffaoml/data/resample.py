"""
#############################################################################
### Spatial resampling for field stores
###
### @file resample.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Downsample structured 2D fields (and coordinates) for train-friendly resolution
while preserving original grid spacing in metadata attrs.
"""

# Native imports
from __future__ import annotations

from typing import Any

# Third-party imports
import numpy as np
from scipy.ndimage import zoom

"""RESAMPLE-----------------------------------------------------------"""


def downsample_spatial(
    fields: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_nx: int,
    target_ny: int,
    order: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Resize ``(n_time, ny, nx)`` field stacks to ``(target_ny, target_nx)``.

    Parameters:
        fields (dict[str, np.ndarray]): Channel arrays sharing time/spatial shape.
        x (np.ndarray): X coordinates length ``nx``.
        y (np.ndarray): Y coordinates length ``ny``.
        target_nx (int): Output width.
        target_ny (int): Output height.
        order (int): ``scipy.ndimage.zoom`` spline order (1 = linear).

    Returns:
        tuple: ``(fields_out, x_out, y_out, meta)`` where ``meta`` records
        original sizes and spacing for Zarr attrs.
    """
    if target_nx <= 0 or target_ny <= 0:
        raise ValueError("target_nx and target_ny must be positive")
    sample = next(iter(fields.values()))
    if sample.ndim != 3:
        raise ValueError("field arrays must be 3D (time, y, x)")

    n_time, ny, nx = sample.shape
    if (target_nx, target_ny) == (nx, ny):
        dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        dy = float(np.median(np.diff(y))) if y.size > 1 else 1.0
        meta = {
            "downsampled": False,
            "nx_original": nx,
            "ny_original": ny,
            "dx_original": dx,
            "dy_original": dy,
        }
        return fields, x, y, meta

    zx = target_nx / nx
    zy = target_ny / ny
    out: dict[str, np.ndarray] = {}
    for name, array in fields.items():
        if array.shape != (n_time, ny, nx):
            raise ValueError(f"field '{name}' shape mismatch")
        out[name] = zoom(array, (1.0, zy, zx), order=order).astype(array.dtype)

    x_out = np.linspace(float(x[0]), float(x[-1]), target_nx, dtype=np.float64)
    y_out = np.linspace(float(y[0]), float(y[-1]), target_ny, dtype=np.float64)
    dx_orig = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    dy_orig = float(np.median(np.diff(y))) if y.size > 1 else 1.0
    meta = {
        "downsampled": True,
        "nx_original": int(nx),
        "ny_original": int(ny),
        "dx_original": dx_orig,
        "dy_original": dy_orig,
        "downsample_order": int(order),
    }
    return out, x_out, y_out, meta
