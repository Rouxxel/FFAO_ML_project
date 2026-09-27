"""
#############################################################################
### Mesh node-field normalization (Stage 2)
###
### @file mesh_preprocessing.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fit velocity and pressure mean/std on **train trajectories only**, then apply to
all splits without leakage.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import list_mesh_trajectories_for_split
from ffaoml.data.mesh_io import load_mesh_trajectory_dir

"""CONSTANTS-----------------------------------------------------------"""
MESH_STATS_FILENAME = "mesh_preprocess_stats.json"

"""TYPES-----------------------------------------------------------"""


@dataclass
class MeshPreprocessStats:
    """Normalization for mesh node velocity (2) and pressure (1)."""

    velocity_mean: tuple[float, float]
    velocity_std: tuple[float, float]
    pressure_mean: float
    pressure_std: float
    fit_split: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshPreprocessStats:
        return cls(
            velocity_mean=tuple(float(x) for x in data["velocity_mean"]),
            velocity_std=tuple(float(x) for x in data["velocity_std"]),
            pressure_mean=float(data["pressure_mean"]),
            pressure_std=float(data["pressure_std"]),
            fit_split=str(data["fit_split"]),
            schema_version=int(data.get("schema_version", 1)),
        )


"""FIT / APPLY-----------------------------------------------------------"""


def _stack_train_fields(cfg: DictConfig, split: str) -> tuple[np.ndarray, np.ndarray]:
    velocities: list[np.ndarray] = []
    pressures: list[np.ndarray] = []
    for _sim_id, traj_dir in list_mesh_trajectories_for_split(cfg, split):
        traj = load_mesh_trajectory_dir(traj_dir)
        velocities.append(traj.velocity.reshape(-1, 2))
        p = traj.pressure
        if p.ndim == 3:
            p = p.reshape(-1, 1)
        else:
            p = p.reshape(-1, 1)
        pressures.append(p)
    if not velocities:
        raise ValueError(f"no trajectories for split {split!r}")
    vel = np.concatenate(velocities, axis=0)
    pres = np.concatenate(pressures, axis=0)
    return vel, pres


def fit_mesh_preprocess_stats(cfg: DictConfig) -> MeshPreprocessStats:
    """Fit normalization on train-split trajectories only."""
    fit_split = str(cfg.dataset.normalization.get("fit_split", "train"))
    vel, pres = _stack_train_fields(cfg, fit_split)
    v_mean = vel.mean(axis=0)
    v_std = np.maximum(vel.std(axis=0), 1e-8)
    p_mean = float(pres.mean())
    p_std = float(max(pres.std(), 1e-8))
    return MeshPreprocessStats(
        velocity_mean=(float(v_mean[0]), float(v_mean[1])),
        velocity_std=(float(v_std[0]), float(v_std[1])),
        pressure_mean=p_mean,
        pressure_std=p_std,
        fit_split=fit_split,
    )


def normalize_mesh_state(
    velocity: np.ndarray,
    pressure: np.ndarray,
    stats: MeshPreprocessStats,
) -> np.ndarray:
    """
    Normalize and concatenate node features to ``(N, 3)`` or ``(T, N, 3)``.

    Parameters:
        velocity (np.ndarray): ``(..., N, 2)``.
        pressure (np.ndarray): ``(..., N, 1)`` or ``(..., N)``.
        stats (MeshPreprocessStats): Fitted stats.

    Returns:
        np.ndarray: ``(..., N, 3)`` float32.
    """
    v_mean = np.array(stats.velocity_mean, dtype=np.float32)
    v_std = np.array(stats.velocity_std, dtype=np.float32)
    vn = (velocity - v_mean) / v_std
    if pressure.ndim == velocity.ndim - 1:
        pressure = pressure[..., np.newaxis]
    pn = (pressure - stats.pressure_mean) / stats.pressure_std
    return np.concatenate([vn, pn.astype(np.float32)], axis=-1).astype(np.float32)


def save_mesh_preprocess_stats(path: str | Path, stats: MeshPreprocessStats) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    return dest


def load_mesh_preprocess_stats(path: str | Path) -> MeshPreprocessStats:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MeshPreprocessStats.from_dict(data)
