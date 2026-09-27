"""
#############################################################################
### Build meshgraphnets_mini fixture
###
### @file build_meshgraphnets_mini.py
### @date 2026
#############################################################################

Regenerate ``tests/fixtures/meshgraphnets_mini/`` for CI mesh I/O tests.
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import numpy as np

from ffaoml.app_logging import log_handler

"""CONSTANTS-----------------------------------------------------------"""
OUTPUT = Path(__file__).resolve().parent / "meshgraphnets_mini"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    n_nodes = 24
    n_cells = 30
    n_steps = 8

    mesh_pos = rng.uniform(0.0, 1.0, size=(n_nodes, 2)).astype(np.float32)
    node_type = np.zeros(n_nodes, dtype=np.int32)
    node_type[0] = 4  # inflow
    node_type[-1] = 5  # outflow
    node_type[4:8] = 1  # obstacle band

    cells = rng.integers(0, n_nodes, size=(n_cells, 3), dtype=np.int32)

    velocity = rng.standard_normal((n_steps, n_nodes, 2), dtype=np.float32) * 0.05
    pressure = rng.standard_normal((n_steps, n_nodes, 1), dtype=np.float32) * 0.01

    np.save(OUTPUT / "mesh_pos.npy", mesh_pos)
    np.save(OUTPUT / "node_type.npy", node_type)
    np.save(OUTPUT / "cells.npy", cells)
    np.save(OUTPUT / "velocity.npy", velocity)
    np.save(OUTPUT / "pressure.npy", pressure)
    meta = {"dt": 0.01, "n_steps": n_steps, "sim_id": "meshgraphnets_mini"}
    meta_path = OUTPUT / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
    log_handler.info("Wrote %s (%s bytes total)", OUTPUT, total)


if __name__ == "__main__":
    main()
