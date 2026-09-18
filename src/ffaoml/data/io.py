"""
#############################################################################
### Zarr field store I/O
###
### @file io.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Read and write CONTRACTS-compliant simulation fields as xarray Zarr stores
(``fields.zarr`` per simulation directory).
"""

# Native imports
from __future__ import annotations

from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
import xarray as xr

# Project imports
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS

"""CONSTANTS-----------------------------------------------------------"""
FIELD_STORE_NAME = "fields.zarr"

"""PATHS-----------------------------------------------------------"""


def simulation_store_path(simulation_dir: str | Path) -> Path:
    """
    Path to the Zarr store for one simulation.

    Parameters:
        simulation_dir (str | Path): ``dataset/simulations/<sim_id>/``.

    Returns:
        Path: ``.../fields.zarr``.
    """
    return Path(simulation_dir) / FIELD_STORE_NAME


"""EXPORT-----------------------------------------------------------"""


def write_field_store(
    simulation_dir: str | Path,
    *,
    time: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    fields: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    solid_mask: np.ndarray | None = None,
    time_chunk: int | None = None,
) -> Path:
    """
    Write required flow channels to ``fields.zarr``.

    Parameters:
        simulation_dir (str | Path): Output directory for this case.
        time (np.ndarray): Time coordinate, length ``n_steps``.
        y (np.ndarray): Y grid coordinates, length ``ny``.
        x (np.ndarray): X grid coordinates, length ``nx``.
        fields (dict[str, np.ndarray]): Arrays keyed by ``DEFAULT_FIELD_CHANNELS``
            names, each shape ``(n_steps, ny, nx)``.
        attrs (dict[str, Any] | None): Global Zarr attributes (Re, dt, etc.).
        solid_mask (np.ndarray | None): Optional ``(ny, nx)`` bool cylinder mask.
        time_chunk (int | None): Zarr chunk size along time (defaults to full series).

    Returns:
        Path: Written Zarr directory.

    Raises:
        KeyError: If a required channel is missing.
        ValueError: On shape mismatch for fields or mask.
    """
    simulation_dir = Path(simulation_dir)
    simulation_dir.mkdir(parents=True, exist_ok=True)
    store_path = simulation_store_path(simulation_dir)

    for name in DEFAULT_FIELD_CHANNELS:
        if name not in fields:
            raise KeyError(f"missing required field '{name}'")
        arr = fields[name]
        if arr.shape != (time.size, y.size, x.size):
            raise ValueError(
                f"field '{name}' shape {arr.shape} != "
                f"(n_time={time.size}, ny={y.size}, nx={x.size})"
            )

    data_vars = {
        name: (["time", "y", "x"], fields[name]) for name in DEFAULT_FIELD_CHANNELS
    }
    if solid_mask is not None:
        if solid_mask.shape != (y.size, x.size):
            raise ValueError("solid_mask must have shape (ny, nx)")
        data_vars["solid_mask"] = (["y", "x"], solid_mask.astype(bool))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "y": y, "x": x},
        attrs=attrs or {},
    )
    encoding: dict[str, dict[str, tuple[int, ...]]] | None = None
    if time_chunk is not None and time_chunk > 0:
        chunk_t = min(int(time_chunk), int(time.size))
        encoding = {
            name: {"chunks": (chunk_t, y.size, x.size)}
            for name in DEFAULT_FIELD_CHANNELS
        }
        if solid_mask is not None:
            encoding["solid_mask"] = {"chunks": (y.size, x.size)}
    ds.to_zarr(store_path, mode="w", encoding=encoding)
    return store_path


def open_field_store(simulation_dir: str | Path) -> xr.Dataset:
    """
    Open an existing ``fields.zarr`` store.

    Parameters:
        simulation_dir (str | Path): Simulation directory containing the store.

    Returns:
        xr.Dataset: Lazy-loaded field dataset.
    """
    return xr.open_zarr(simulation_store_path(simulation_dir))
