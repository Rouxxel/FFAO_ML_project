"""
#############################################################################
### MeshGraphDataset - one-step node prediction (Stage 2)
###
### @file mesh_dataset.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Trajectory-level train/val/test from ``metadata.csv``; samples are
``(graph_t) → (graph_{t+Δ})`` on mesh nodes.
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import list_mesh_trajectories_for_split
from ffaoml.data.mesh_io import (
    MeshTrajectory,
    edges_from_cells,
    load_mesh_trajectory_dir,
)
from ffaoml.ml.mesh_preprocessing import MeshPreprocessStats, normalize_mesh_state

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class _MeshIndex:
    traj_idx: int
    time_index: int


"""DATASET-----------------------------------------------------------"""


class MeshGraphDataset:
    """One-step mesh node prediction for a metadata split."""

    def __init__(
        self,
        cfg: DictConfig,
        split: str,
        stats: MeshPreprocessStats,
        *,
        delta_steps: int | None = None,
    ) -> None:
        if not bool(cfg.dataset.get("use_trajectory_splits", False)):
            raise ValueError("MeshGraphDataset requires dataset.use_trajectory_splits")
        self.cfg = cfg
        self.split = split
        self.stats = stats
        self.delta_steps = int(
            delta_steps
            if delta_steps is not None
            else cfg.dataset.prediction.delta_steps
        )
        if self.delta_steps < 1:
            raise ValueError("delta_steps must be >= 1")

        self._trajectories: list[tuple[str, Path, MeshTrajectory, np.ndarray]] = []
        self._indices: list[_MeshIndex] = []

        for sim_id, traj_dir in list_mesh_trajectories_for_split(cfg, split):
            traj = load_mesh_trajectory_dir(traj_dir)
            edge_index = edges_from_cells(traj.cells)
            traj_idx = len(self._trajectories)
            self._trajectories.append((sim_id, traj_dir, traj, edge_index))
            for t in range(max(0, traj.n_steps - self.delta_steps)):
                self._indices.append(_MeshIndex(traj_idx=traj_idx, time_index=t))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._indices[index]
        sim_id, _traj_dir, traj, edge_index = self._trajectories[entry.traj_idx]
        t = entry.time_index
        dt = self.delta_steps
        return {
            "input": normalize_mesh_state(
                traj.velocity[t], traj.pressure[t], self.stats
            ),
            "target": normalize_mesh_state(
                traj.velocity[t + dt], traj.pressure[t + dt], self.stats
            ),
            "edge_index": edge_index.copy(),
            "node_type": traj.node_type.copy(),
            "mesh_pos": traj.mesh_pos.copy(),
            "sim_id": sim_id,
            "time_index": t,
        }


def build_mesh_datasets(
    cfg: DictConfig,
    stats: MeshPreprocessStats,
) -> dict[str, MeshGraphDataset]:
    """Build train/val/test mesh datasets sharing the same normalization stats."""
    return {
        split: MeshGraphDataset(cfg, split, stats) for split in ("train", "val", "test")
    }


def collate_mesh_graph_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate mesh samples into a disjoint union batch (PyG-style).

    Returns a ``batch`` vector of length ``sum(N_i)`` when ``len(samples) > 1``.
    """
    if len(samples) == 1:
        s = samples[0]
        n = s["input"].shape[0]
        return {**s, "batch": np.zeros(n, dtype=np.int64)}
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    node_types: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    batch_vec: list[int] = []
    offset = 0
    for batch_id, s in enumerate(samples):
        n = s["input"].shape[0]
        inputs.append(s["input"])
        targets.append(s["target"])
        node_types.append(s["node_type"])
        positions.append(s["mesh_pos"])
        edges.append(s["edge_index"] + offset)
        batch_vec.extend([batch_id] * n)
        offset += n
    return {
        "input": np.concatenate(inputs, axis=0),
        "target": np.concatenate(targets, axis=0),
        "node_type": np.concatenate(node_types, axis=0),
        "mesh_pos": np.concatenate(positions, axis=0),
        "edge_index": np.concatenate(edges, axis=1),
        "batch": np.array(batch_vec, dtype=np.int64),
        "sim_id": [s["sim_id"] for s in samples],
        "time_index": [s["time_index"] for s in samples],
    }
