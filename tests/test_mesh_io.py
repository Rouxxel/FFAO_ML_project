"""
#############################################################################
### Mesh I/O tests (Stage 2 fixture)
###
### @file test_mesh_io.py
### @date 2026
#############################################################################
"""

from pathlib import Path

import numpy as np

from ffaoml.data.mesh_io import (
    edges_from_cells,
    load_mesh_trajectory_dir,
    validate_mesh_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"


def test_load_meshgraphnets_mini_fixture() -> None:
    traj = load_mesh_trajectory_dir(FIXTURE)
    validate_mesh_trajectory(traj)
    assert traj.n_nodes == 24
    assert traj.n_steps == 8
    assert traj.dt == 0.01


def test_edges_from_cells_unique_and_directed_pairs() -> None:
    cells = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_index = edges_from_cells(cells)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 10  # 5 undirected -> 10 directed
    assert edge_index.dtype == np.int64
