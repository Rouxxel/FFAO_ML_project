"""
#############################################################################
### MeshGraphNet model and training tests
###
### @file test_meshgraphnet_training.py
### @date 2026
#############################################################################
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.data.mesh_io import load_mesh_trajectory_dir
from ffaoml.data.sources.meshgraphnets_cylinder import import_trajectory_from_arrays
from ffaoml.models.meshgraphnet import build_meshgraphnet
from ffaoml.training.train_meshgraphnet import run_meshgraphnet_training

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"
WORK = REPO_ROOT / ".local_test_runs" / "meshgn_train"


def test_meshgraphnet_forward_shape() -> None:
    import torch

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
                "model.hidden_dim=32",
                "model.num_message_passing_steps=2",
            ],
        )
    GlobalHydra.instance().clear()
    model = build_meshgraphnet(cfg)
    traj = load_mesh_trajectory_dir(FIXTURE)
    n = traj.mesh_pos.shape[0]
    x = torch.randn(n, 3)
    pos = torch.from_numpy(traj.mesh_pos.astype("float32"))
    edge_index = torch.randint(0, n, (2, 40))
    node_type = torch.from_numpy(traj.node_type.astype("int64"))
    y = model(x, pos, edge_index, node_type)
    assert y.shape == (n, 3)


@pytest.fixture(scope="module")
def mesh_train_cfg():
    out_root = WORK / "meshgraphnets_data"
    if out_root.is_dir():
        shutil.rmtree(out_root)
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
                f"dataset.output_root={out_root.as_posix()}",
                "model.hidden_dim=32",
                "model.num_message_passing_steps=2",
                "train.epochs=6",
                "train.batch_size=1",
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
        cfg, arrays, sim_id="cylinder_flow_val_00000", split="val", repo_root=REPO_ROOT
    )
    return cfg


def test_meshgraphnet_training_writes_checkpoint_bundle(mesh_train_cfg) -> None:
    out = WORK / "run"
    if out.is_dir():
        shutil.rmtree(out)
    result = run_meshgraphnet_training(mesh_train_cfg, out, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
    assert (out / "config.yaml").is_file()
    assert (out / "preprocess_stats.json").is_file()
    assert (out / "bundle_manifest.json").is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary.get("model") == "meshgraphnet"
    assert summary.get("config_hash")
