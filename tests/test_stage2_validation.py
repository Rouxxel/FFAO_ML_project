"""
#############################################################################
### Stage 2 mesh validation tests
###
### @file test_stage2_validation.py
### @date 2026
#############################################################################
"""

import json
import shutil
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.data.mesh_io import load_mesh_trajectory_dir
from ffaoml.data.sources.meshgraphnets_cylinder import import_trajectory_from_arrays
from ffaoml.validation.stage2_meshgraphnets import (
    run_stage2_mesh_validation,
    velocity_magnitude,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"
WORK = REPO_ROOT / ".local_test_runs" / "stage2_validation"


def test_stage2_validation_writes_artifacts() -> None:
    out_root = WORK / "meshgraphnets_data"
    out_dir = WORK / "report"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=stage2_meshgraphnets",
                f"dataset.output_root={out_root.as_posix()}",
            ],
        )
    GlobalHydra.instance().clear()
    traj = load_mesh_trajectory_dir(FIXTURE)
    arrays = {
        "mesh_pos": traj.mesh_pos,
        "node_type": traj.node_type,
        "cells": traj.cells,
        "velocity": traj.velocity,
        "pressure": traj.pressure,
    }
    import_trajectory_from_arrays(
        cfg,
        arrays,
        sim_id="cylinder_flow_train_00000",
        split="train",
        repo_root=REPO_ROOT,
    )
    result = run_stage2_mesh_validation(
        cfg,
        output_dir=out_dir,
        fps=4,
        max_animation_frames=4,
    )
    assert result.summary_md.is_file()
    assert result.mesh_velocity_snapshots.is_file()
    assert result.velocity_animation is not None
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["n_nodes"] == traj.n_nodes
    assert "mesh" in result.summary_md.read_text(encoding="utf-8").lower()


def test_velocity_magnitude_2d() -> None:
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    m = velocity_magnitude(v)
    assert m.shape == (2,)
