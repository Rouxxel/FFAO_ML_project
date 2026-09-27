"""
#############################################################################
### Hydra configuration tests
###
### @file test_config.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Compose default and override configs; verify run-dir helpers and seeding.
"""

# Native imports
from pathlib import Path

# Third-party imports
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Project imports
from ffaoml.config import (
    config_dir,
    seed_from_config,
    set_global_seed,
    write_resolved_config,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""FIXTURES-----------------------------------------------------------"""


@pytest.fixture(scope="module")
def default_config():
    """Composed root ``config`` with default Hydra groups."""
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        yield compose(config_name="config")


"""TESTS-----------------------------------------------------------"""


def test_compose_loads_default_groups(default_config) -> None:
    cfg = default_config
    assert cfg.simulation.backend == "fd"
    assert cfg.model.name == "cnn"
    assert cfg.train.batch_size == 8
    assert cfg.paths.runs_root == "results/runs"
    assert cfg.eval.split == "val"
    assert cfg.eval.rollout_horizon == 50


def test_default_dataset_is_stage1_zenodo(default_config) -> None:
    cfg = default_config
    assert cfg.dataset.stage == 1
    assert cfg.dataset.source_id == "zenodo_re100"
    assert cfg.dataset.re == 100
    assert cfg.dataset.use_re_splits is False
    assert cfg.dataset.temporal_split.train == [0, 70]


def test_stage2_meshgraphnets_compose_without_data_on_disk() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["dataset=stage2_meshgraphnets", "model=meshgraphnet"],
        )
    GlobalHydra.instance().clear()
    assert cfg.dataset.stage == 2
    assert cfg.dataset.source_id == "meshgraphnets_cylinder_flow"
    assert cfg.dataset.use_trajectory_splits is True
    assert cfg.dataset["import"].n_steps == 600
    assert str(cfg.dataset.output_root).endswith("meshgraphnets_data")
    assert cfg.model.name == "meshgraphnet"
    assert cfg.model.num_message_passing_steps == 15


def test_re_splits_compose_when_selected() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=["dataset=splits"])
    GlobalHydra.instance().clear()
    assert cfg.dataset.stage == 3
    assert cfg.dataset.use_re_splits is True
    assert cfg.dataset.train_re == [50, 75, 100, 150, 200]


def test_stage3_cfdbench_compose_without_data_on_disk() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["dataset=stage3_cfdbench", "model=cnn_re"],
        )
    GlobalHydra.instance().clear()
    assert cfg.dataset.stage == 3
    assert cfg.dataset.source_id == "cfdbench_cylinder"
    assert cfg.dataset.use_re_splits is True
    assert cfg.dataset.train_re == [50, 75, 100, 150, 200]
    assert str(cfg.dataset.output_root).endswith("cfdbench_data")
    assert cfg.model.condition_on_re is True


def test_write_resolved_config_roundtrip(default_config) -> None:
    run_dir = REPO_ROOT / ".local_test_runs" / "config_roundtrip"
    path = write_resolved_config(default_config, run_dir)
    assert path.is_file()
    loaded = OmegaConf.load(path)
    assert loaded.dataset.source_id == default_config.dataset.source_id


def test_seed_from_config(default_config) -> None:
    assert seed_from_config(default_config) == 42


def test_set_global_seed_accepts_int() -> None:
    set_global_seed(123)
