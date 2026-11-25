"""
#############################################################################
### Stage 3 train E2E on cfdbench_mini (no HF download)
###
### @file test_stage3_cfdbench_training.py
### @date 2026
#############################################################################
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.training.train import run_cnn_training

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cfdbench_mini"
WORK = REPO_ROOT / ".local_test_runs" / "stage3_cfdbench_train"

STAGE3_TRAIN_OVERRIDES = [
    "dataset=stage3_cfdbench",
    "model=cnn_re",
    f"dataset.output_root={FIXTURE_ROOT.as_posix()}",
    "dataset.train_re=[50,100]",
    "dataset.val_re=[125]",
    "dataset.test_re=[250]",
    "dataset.temporal_split.train=[0,4]",
    "dataset.temporal_split.val=[0,4]",
    "dataset.temporal_split.test=[0,4]",
    "train.epochs=3",
    "train.batch_size=2",
]


def test_train_stage3_cfdbench_mini_fixture() -> None:
    if not (FIXTURE_ROOT / "metadata.csv").is_file():
        pytest.skip("missing cfdbench_mini fixture")
    run_dir = WORK / "stage3_cnn_re"
    if run_dir.is_dir():
        shutil.rmtree(run_dir)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=STAGE3_TRAIN_OVERRIDES)
    GlobalHydra.instance().clear()

    result = run_cnn_training(cfg, run_dir, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
    assert result.beats_persistence is not None
