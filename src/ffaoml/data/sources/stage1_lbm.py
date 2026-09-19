"""
#############################################################################
### Stage 1 LBM HDF5 generator (Zenodo HDF5 fallback)
###
### @file stage1_lbm.py
### @author Sebastian Russo
### @date 2026
#############################################################################

D2Q9 lattice Boltzmann cylinder wake at Re≈100; writes Addiucci-compatible HDF5.
"""

# Native imports
from __future__ import annotations

from pathlib import Path

# Third-party imports
import h5py
import numpy as np
import scipy.ndimage as ndimage

# Project imports
from ffaoml.app_logging import log_handler

"""CONSTANTS-----------------------------------------------------------"""
TARGET_N = 64
COLLECT_LAST_N = 100
COLLECT_INTERVAL = 20
TOTAL_STEPS = 8000
DEFAULT_H5_NAME = "cylinder_re100_grid64_last100.h5"


def run_lbm_re100(
    *,
    total_steps: int = TOTAL_STEPS,
    collect_last_n: int = COLLECT_LAST_N,
    collect_interval: int = COLLECT_INTERVAL,
    target_n: int = TARGET_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run D2Q9 LBM and return u, v, vorticity on a ``target_n`` grid.

    Returns:
        tuple: Each array ``(n_time, target_n, target_n)`` float32.
    """
    log_handler.info("LBM Re=100: %s steps on 200x80 lattice...", total_steps)
    nx, ny = 200, 80
    cx, cy, r = nx // 4, ny // 2, ny // 9
    reynolds_no = 100.0
    u_max = 0.1
    viscosity = (u_max * r * 2.0) / reynolds_no
    omega = 1.0 / (3.0 * viscosity + 0.5)

    weights = np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )
    cx_lbm = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy_lbm = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cylinder_mask = (x - cx) ** 2 + (y - cy) ** 2 < r**2

    rho0 = np.ones((nx, ny), dtype=np.float64)
    ux0 = np.zeros((nx, ny), dtype=np.float64)
    uy0 = np.zeros((nx, ny), dtype=np.float64)
    ux0[0, :] = u_max
    f = np.zeros((9, nx, ny), dtype=np.float64)
    feq = np.zeros_like(f)
    for i in range(9):
        cu = 3.0 * (cx_lbm[i] * ux0 + cy_lbm[i] * uy0)
        feq[i] = rho0 * weights[i] * (1.0 + cu + 0.5 * cu**2 - 1.5 * (ux0**2 + uy0**2))
    f[:] = feq
    ux0[cylinder_mask] = 0.0
    uy0[cylinder_mask] = 0.0

    history_u: list[np.ndarray] = []
    history_v: list[np.ndarray] = []
    history_omega: list[np.ndarray] = []
    window_start = total_steps - collect_last_n * collect_interval

    for step in range(total_steps):
        rho = np.sum(f, axis=0)
        rho_safe = np.maximum(rho, 1.0e-12)
        ux = np.sum(f * cx_lbm[:, None, None], axis=0) / rho_safe
        uy = np.sum(f * cy_lbm[:, None, None], axis=0) / rho_safe

        ux[cylinder_mask] = 0.0
        uy[cylinder_mask] = 0.0
        ux[0, :] = u_max
        uy[0, :] = 0.0

        for i in range(9):
            cu = 3.0 * (cx_lbm[i] * ux + cy_lbm[i] * uy)
            cu = np.clip(cu, -20.0, 20.0)
            feq[i] = rho * weights[i] * (1.0 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))
            f[i] = f[i] + omega * (feq[i] - f[i])

        for i in range(9):
            f[i, cylinder_mask] = f[opposite[i], cylinder_mask]

        for i in range(9):
            f[i] = np.roll(np.roll(f[i], cx_lbm[i], axis=0), cy_lbm[i], axis=1)

        f = np.maximum(f, 0.0)
        if not np.isfinite(f).all():
            raise RuntimeError(
                "LBM simulation became non-finite; reduce Reynolds number or "
                "increase grid resolution."
            )

        if step >= window_start and step % collect_interval == 0:
            du_dy = np.gradient(ux, axis=1)
            dv_dx = np.gradient(uy, axis=0)
            vorticity = dv_dx - du_dy

            row0, row1 = cx - ny // 2, cx + ny // 2
            crop_ux = ux[row0:row1, :]
            crop_uy = uy[row0:row1, :]
            crop_vort = vorticity[row0:row1, :]

            zoom_y = target_n / crop_ux.shape[0]
            zoom_x = target_n / crop_ux.shape[1]
            history_u.append(
                ndimage.zoom(crop_ux, (zoom_y, zoom_x), order=3).astype(np.float32)
            )
            history_v.append(
                ndimage.zoom(crop_uy, (zoom_y, zoom_x), order=3).astype(np.float32)
            )
            history_omega.append(
                ndimage.zoom(crop_vort, (zoom_y, zoom_x), order=3).astype(np.float32)
            )

    u = np.array(history_u[-collect_last_n:], dtype=np.float32)
    v = np.array(history_v[-collect_last_n:], dtype=np.float32)
    omega = np.array(history_omega[-collect_last_n:], dtype=np.float32)
    if not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(omega).all()):
        raise RuntimeError("LBM output contains non-finite velocity or vorticity.")
    return u, v, omega


def write_addiucci_h5(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
) -> None:
    """Write ``fields``, ``grid_x``, ``grid_y`` in PO-CAE HDF5 layout."""
    if u.shape != v.shape or u.shape != omega.shape:
        raise ValueError("u, v, vorticity must share the same shape")
    _, ny, nx = u.shape
    fields = np.stack([u, v, omega], axis=1)
    grid_x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    grid_y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields, compression="gzip")
        handle.create_dataset("grid_x", data=grid_x)
        handle.create_dataset("grid_y", data=grid_y)
        handle.attrs["generator"] = "ffaoml.data.sources.stage1_lbm"
        handle.attrs["re"] = 100.0
        handle.attrs["note"] = (
            "Local LBM fallback when Zenodo HDF5 is unavailable; "
            "benchmark definition per Addiucci (2026) Zenodo 18669296."
        )


def generate_stage1_h5(
    output_path: Path,
    *,
    fast: bool = False,
) -> Path:
    """Run full LBM and write HDF5 to *output_path*."""
    steps = 1200 if fast else TOTAL_STEPS
    collect_n = 20 if fast else COLLECT_LAST_N
    interval = COLLECT_INTERVAL if not fast else 10
    u, v, omega = run_lbm_re100(
        total_steps=steps,
        collect_last_n=collect_n,
        collect_interval=interval,
    )
    write_addiucci_h5(output_path, u, v, omega)
    log_handler.info("Wrote LBM HDF5: %s", output_path)
    return output_path
