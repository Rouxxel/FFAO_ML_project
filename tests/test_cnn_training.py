"""
#############################################################################
### CNN model and training tests
###
### @file test_cnn_training.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Forward pass and short training run (requires PyTorch + Stage 1 fixture).
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.training.train import run_cnn_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "cnn_train"


def test_flow_cnn_forward_shape() -> None:
    import torch
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    model = build_flow_cnn(cfg)
    x = torch.randn(2, 4, 10, 12)
    y = model(x)
    assert y.shape == x.shape


@pytest.fixture(scope="module")
def train_cfg():
    if not FIXTURE_H5.is_file():
        pytest.skip(f"missing fixture {FIXTURE_H5}")
    dataset_root = WORK / "dataset"
    if dataset_root.is_dir():
        import shutil

        shutil.rmtree(dataset_root)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.dataset_root={dataset_root.as_posix()}",
                "dataset.temporal_split.train=[0,4]",
                "dataset.temporal_split.val=[3,6]",
                "dataset.temporal_split.test=[5,6]",
                "train.epochs=8",
                "train.batch_size=2",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_cnn_training_writes_checkpoint_bundle(train_cfg) -> None:
    out = WORK / "run"
    if out.is_dir():
        import shutil

        shutil.rmtree(out)
    result = run_cnn_training(train_cfg, out, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
    assert (out / "config.yaml").is_file()
    assert (out / "preprocess_stats.json").is_file()
    assert (out / "bundle_manifest.json").is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary.get("seed") == train_cfg.seed
    assert summary.get("config_hash")
    assert summary["beats_persistence"] is True
    assert result.beats_persistence
