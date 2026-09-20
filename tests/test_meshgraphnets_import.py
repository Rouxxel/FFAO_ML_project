"""
#############################################################################
### MeshGraphNets import tests
###
### @file test_meshgraphnets_import.py
### @date 2026
#############################################################################
"""

import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.data.mesh_io import load_mesh_trajectory_dir, validate_mesh_trajectory
from ffaoml.data.metadata import read_metadata_rows
from ffaoml.data.sources.meshgraphnets_cylinder import (
    import_trajectory_from_arrays,
    run_meshgraphnets_import,
    tfrecord_cache_path,
)
from ffaoml.manifests import dataset_manifest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "meshgraphnets_mini"
SCRATCH = REPO_ROOT / ".local_test_runs" / "meshgraphnets_import"


def _stage2_cfg(output_root: Path):
    GlobalHydra.instance().clear()
    out = output_root.as_posix()
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
    return cfg


def test_import_trajectory_from_arrays_writes_manifest_and_metadata() -> None:
    out_root = SCRATCH / "from_arrays" / "meshgraphnets_data"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    traj = load_mesh_trajectory_dir(FIXTURE)
    arrays = {
        "mesh_pos": traj.mesh_pos,
        "node_type": traj.node_type,
        "cells": traj.cells,
        "velocity": traj.velocity,
        "pressure": traj.pressure,
    }
    cfg = _stage2_cfg(out_root)
    dest = import_trajectory_from_arrays(
        cfg,
        arrays,
        sim_id="cylinder_flow_train_00000",
        split="train",
        repo_root=REPO_ROOT,
    )
    assert dest.is_dir()
    loaded = load_mesh_trajectory_dir(dest)
    validate_mesh_trajectory(loaded)
    assert dataset_manifest_path(out_root).is_file()
    rows = read_metadata_rows(out_root)
    assert len(rows) == 1
    assert rows[0]["sim_id"] == "cylinder_flow_train_00000"


@pytest.mark.slow
def test_import_one_trajectory_from_cached_tfrecord() -> None:
    cache = REPO_ROOT / ".cache" / "meshgraphnets_cylinder"
    train_shard = tfrecord_cache_path(cache, "train")
    if not train_shard.is_file():
        pytest.skip("no cached train.tfrecord; run download script locally first")
    out_root = SCRATCH / "from_tfrecord" / "meshgraphnets_data"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _stage2_cfg(out_root)
    try:
        result = run_meshgraphnets_import(
            cfg,
            splits=("train",),
            max_trajectories_per_split=1,
            repo_root=REPO_ROOT,
            force_download=False,
        )
    except ImportError:
        pytest.skip("tensorflow not installed")
    assert result.trajectories_imported >= 1
    assert dataset_manifest_path(out_root).is_file()
