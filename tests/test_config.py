"""Hydra configuration composition."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ffaoml.config import (
    config_dir,
    seed_from_config,
    set_global_seed,
    write_resolved_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def default_config():
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        yield compose(config_name="config")


def test_compose_loads_default_groups(default_config) -> None:
    cfg = default_config
    assert cfg.simulation.backend == "fd"
    assert cfg.model.name == "cnn"
    assert cfg.train.batch_size == 8
    assert cfg.paths.runs_root == "results/runs"


def test_re_splits_are_config_driven(default_config) -> None:
    cfg = default_config
    assert cfg.dataset.train_re == [50, 75, 100, 150, 200]
    assert cfg.dataset.val_re == [125, 175]
    assert cfg.dataset.test_re == [250, 300, 400]


def test_write_resolved_config_roundtrip(default_config) -> None:
    run_dir = REPO_ROOT / ".local_test_runs" / "config_roundtrip"
    path = write_resolved_config(default_config, run_dir)
    assert path.is_file()
    loaded = OmegaConf.load(path)
    assert loaded.dataset.train_re == default_config.dataset.train_re


def test_seed_from_config(default_config) -> None:
    assert seed_from_config(default_config) == 42


def test_set_global_seed_accepts_int() -> None:
    set_global_seed(123)
