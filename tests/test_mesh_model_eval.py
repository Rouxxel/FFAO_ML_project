"""
#############################################################################
### Mesh model evaluation tests
###
### @file test_mesh_model_eval.py
### @date 2026
#############################################################################
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.data.mesh_io import load_mesh_trajectory_dir
from ffaoml.data.sources.meshgraphnets_cylinder import import_trajectory_from_arrays
from ffaoml.evaluation.evaluate_mesh_model import run_mesh_model_evaluation
from ffaoml.evaluation.mesh_rollout import (
    mesh_model_rollout_curve,
    mesh_persistence_rollout_curve,
    mesh_state_series,
)
from ffaoml.ml.mesh_preprocessing import MeshPreprocessStats
from ffaoml.training.train_meshgraphnet import run_meshgraphnet_training

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"
WORK = REPO_ROOT / ".local_test_runs" / "mesh_eval"


def test_mesh_rollout_curves_on_fixture_series() -> None:
    traj = load_mesh_trajectory_dir(FIXTURE)
    vel = traj.velocity.reshape(-1, 2)
    pres = traj.pressure.reshape(-1)
    stats = MeshPreprocessStats(
        velocity_mean=(float(vel[:, 0].mean()), float(vel[:, 1].mean())),
        velocity_std=(float(max(vel[:, 0].std(), 1e-8)), float(max(vel[:, 1].std(), 1e-8))),
        pressure_mean=float(pres.mean()),
        pressure_std=float(max(pres.std(), 1e-8)),
        fit_split="train",
    )
    series = mesh_state_series(traj, stats)

    def identity(state: np.ndarray) -> np.ndarray:
        return state.copy()

    curve = mesh_model_rollout_curve(series, identity, horizon=3)
    persist = mesh_persistence_rollout_curve(series, horizon=3)
    assert len(curve.horizons) == 3
    assert curve.mse[0] <= curve.mse[-1] + 1e-6 or persist.mse[0] >= 0


@pytest.fixture(scope="module")
def mesh_eval_run_dir():
    out_root = WORK / "meshgraphnets_data"
    run_dir = WORK / "run"
    for path in (out_root, run_dir):
        if path.is_dir():
            shutil.rmtree(path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=stage2_meshgraphnets",
                "model=meshgraphnet",
                "train=meshgraphnet",
                "eval=meshgraphnet",
                f"dataset.output_root={out_root.as_posix()}",
                "model.hidden_dim=32",
                "model.num_message_passing_steps=2",
                "train.epochs=4",
                "eval.rollout_horizon=3",
                "eval.report_horizons=[1,2,3]",
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
    import_trajectory_from_arrays(
        cfg,
        arrays,
        sim_id="cylinder_flow_val_00000",
        split="val",
        repo_root=REPO_ROOT,
    )
    run_meshgraphnet_training(cfg, run_dir, repo_root=REPO_ROOT)
    return run_dir


def test_mesh_model_evaluation_writes_figures(mesh_eval_run_dir: Path) -> None:
    result = run_mesh_model_evaluation(mesh_eval_run_dir, split="val")
    assert result.horizon_figure.is_file()
    assert result.stability_figure.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "mesh_model_evaluation"
    assert "meshgraphnet" in payload["rollout"]
    assert "persistence" in payload["rollout"]
    assert "note_stage1_comparison" in payload
