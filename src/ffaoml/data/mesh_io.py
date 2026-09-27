"""
#############################################################################
### Mesh trajectory I/O (Stage 2)
###
### @file mesh_io.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Load small mesh-graph fixtures and build edge indices from triangle ``cells``.
See ``documentation/CONTRACTS.md``.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np

# Project imports
from ffaoml.contracts import MESH_DYNAMIC_NODE_FIELDS, MESH_STATIC_ARRAYS

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class MeshTrajectory:
    """One cylinder_flow-style trajectory on an unstructured mesh."""

    mesh_pos: np.ndarray
    node_type: np.ndarray
    cells: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    dt: float = 0.01

    @property
    def n_nodes(self) -> int:
        return int(self.mesh_pos.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.velocity.shape[0])


"""I/O-----------------------------------------------------------"""


def load_mesh_trajectory_dir(root: str | Path) -> MeshTrajectory:
    """
    Load arrays from a fixture or import directory layout.

    Expects ``mesh_pos.npy``, ``node_type.npy``, ``cells.npy``, ``velocity.npy``,
    ``pressure.npy``, and optional ``meta.json`` (``dt``).
    """
    base = Path(root)
    mesh_pos = np.load(base / "mesh_pos.npy")
    node_type = np.load(base / "node_type.npy")
    cells = np.load(base / "cells.npy")
    velocity = np.load(base / "velocity.npy")
    pressure = np.load(base / "pressure.npy")
    dt = 0.01
    meta_path = base / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dt = float(meta.get("dt", dt))
    return MeshTrajectory(
        mesh_pos=mesh_pos,
        node_type=node_type,
        cells=cells,
        velocity=velocity,
        pressure=pressure,
        dt=dt,
    )


def validate_mesh_trajectory(traj: MeshTrajectory) -> None:
    """
    Raise ``ValueError`` when array shapes violate the mesh contract.

    Parameters:
        traj (MeshTrajectory): Loaded trajectory.
    """
    n = traj.n_nodes
    t = traj.n_steps
    if traj.mesh_pos.shape != (n, 2):
        raise ValueError(f"mesh_pos expected ({n}, 2), got {traj.mesh_pos.shape}")
    if traj.node_type.shape != (n,):
        raise ValueError(f"node_type expected ({n},), got {traj.node_type.shape}")
    if traj.cells.ndim != 2 or traj.cells.shape[1] != 3:
        raise ValueError(f"cells expected (F, 3), got {traj.cells.shape}")
    if traj.velocity.shape != (t, n, 2):
        raise ValueError(f"velocity expected ({t}, {n}, 2), got {traj.velocity.shape}")
    p_shape = traj.pressure.shape
    if p_shape not in ((t, n), (t, n, 1)):
        raise ValueError(
            f"pressure expected ({t}, {n}) or ({t}, {n}, 1), got {p_shape}"
        )
    for name in MESH_STATIC_ARRAYS:
        if not hasattr(traj, name):
            raise ValueError(f"missing static field {name}")
    for name in MESH_DYNAMIC_NODE_FIELDS:
        if not hasattr(traj, name):
            raise ValueError(f"missing dynamic field {name}")


def edges_from_cells(cells: np.ndarray) -> np.ndarray:
    """
    Undirected edges from triangle indices (PyG ``edge_index`` layout).

    Returns:
        np.ndarray: ``(2, n_edges)`` int64; undirected edges as both directions.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"cells must be (F, 3), got {cells.shape}")
    pairs: set[tuple[int, int]] = set()
    for i, j, k in cells:
        for a, b in ((i, j), (j, k), (k, i)):
            lo, hi = (int(min(a, b)), int(max(a, b)))
            if lo != hi:
                pairs.add((lo, hi))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    undirected = sorted(pairs)
    src = [a for a, _ in undirected] + [b for _, b in undirected]
    dst = [b for _, b in undirected] + [a for a, _ in undirected]
    return np.array([src, dst], dtype=np.int64)
