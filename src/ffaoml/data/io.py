"""Zarr / xarray export for simulation fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ffaoml.contracts import DEFAULT_FIELD_CHANNELS

FIELD_STORE_NAME = "fields.zarr"


def simulation_store_path(simulation_dir: str | Path) -> Path:
    return Path(simulation_dir) / FIELD_STORE_NAME


def write_field_store(
    simulation_dir: str | Path,
    *,
    time: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    fields: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    solid_mask: np.ndarray | None = None,
) -> Path:
    """Write CONTRACTS-compliant fields to ``fields.zarr`` under ``simulation_dir``."""
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
    ds.to_zarr(store_path, mode="w")
    return store_path


def open_field_store(simulation_dir: str | Path) -> xr.Dataset:
    return xr.open_zarr(simulation_store_path(simulation_dir))
