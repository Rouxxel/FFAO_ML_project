"""
#############################################################################
### MeshGraphDataset tests
###
### @file test_mesh_dataset.py
### @date 2026
#############################################################################
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader

from ffaoml.config import config_dir
from ffaoml.data.mesh_io import load_mesh_trajectory_dir
from ffaoml.data.sources.meshgraphnets_cylinder import import_trajectory_from_arrays
from ffaoml.ml.mesh_dataset import (
    MeshGraphDataset,
    build_mesh_datasets,
    collate_mesh_graph_batch,
)
from ffaoml.ml.mesh_preprocessing import fit_mesh_preprocess_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"
SCRATCH = REPO_ROOT / ".local_test_runs" / "mesh_dataset"


@pytest.fixture(scope="module")
def stage2_mesh_cfg():
    out_root = SCRATCH / "meshgraphnets_data"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    GlobalHydra.instance().clear()
    out = out_root.as_posix()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=stage2_meshgraphnets",
                f"dataset.output_root={out}",
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


def test_mesh_dataloader_smoke(stage2_mesh_cfg) -> None:
    stats = fit_mesh_preprocess_stats(stage2_mesh_cfg)
    assert stats.fit_split == "train"
    datasets = build_mesh_datasets(stage2_mesh_cfg, stats)
    assert len(datasets["train"]) > 0
    assert len(datasets["val"]) > 0
    train_loader = DataLoader(
        datasets["train"],
        batch_size=1,
        shuffle=False,
        collate_fn=collate_mesh_graph_batch,
    )
    batch = next(iter(train_loader))
    assert batch["input"].shape[-1] == 3
    assert batch["edge_index"].shape[0] == 2
    assert batch["batch"].shape[0] == batch["input"].shape[0]


def test_train_stats_do_not_use_val_trajectory(stage2_mesh_cfg) -> None:
    stats = fit_mesh_preprocess_stats(stage2_mesh_cfg)
    train_ds = MeshGraphDataset(stage2_mesh_cfg, "train", stats)
    val_ds = MeshGraphDataset(stage2_mesh_cfg, "val", stats)
    assert train_ds[0]["sim_id"].startswith("cylinder_flow_train")
    assert val_ds[0]["sim_id"].startswith("cylinder_flow_val")
    raw = load_mesh_trajectory_dir(
        SCRATCH / "meshgraphnets_data" / "trajectories" / "cylinder_flow_train_00000"
    )
    manual_mean = raw.velocity.reshape(-1, 2).mean(axis=0)
    assert np.allclose(manual_mean[0], stats.velocity_mean[0], rtol=1e-5)
