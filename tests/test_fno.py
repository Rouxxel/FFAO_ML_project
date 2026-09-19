"""
#############################################################################
### FNO and physics-informed loss tests
###
### @file test_fno.py
### @author Sebastian Russo
### @date 2026
#############################################################################

FlowFNO forward pass, divergence penalty, and short training (requires [ml]).
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import pytest

pytest.importorskip("torch")
pytest.importorskip("neuralop")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.models.fno import build_flow_fno
from ffaoml.physics.torch_ops import divergence_penalty
from ffaoml.training.losses import build_training_loss
from ffaoml.training.train import run_cnn_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "fno_train"


def test_flow_fno_forward_shape() -> None:
    import torch

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["model=fno", "model.n_modes=[4,4]", "model.n_layers=2"],
        )
    model = build_flow_fno(cfg)
    x = torch.randn(2, 4, 10, 12)
    y = model(x)
    assert y.shape == x.shape


def test_divergence_penalty_nonnegative() -> None:
    import torch

    pred = torch.randn(2, 4, 10, 12)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["train.loss.divergence_weight=0.1"],
        )
    loss_fn = build_training_loss(cfg)
    value = loss_fn(pred, pred)
    assert float(value.item()) >= 0.0
    assert float(divergence_penalty(pred, dx=0.1, dy=0.1).item()) >= 0.0


@pytest.fixture(scope="module")
def fno_train_cfg():
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
                "model=fno",
                "model.n_modes=[4,4]",
                "model.n_layers=2",
                "model.hidden_channels=16",
                "train.epochs=6",
                "train.batch_size=2",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_fno_training_writes_checkpoint(fno_train_cfg) -> None:
    out = WORK / "run"
    if out.is_dir():
        import shutil

        shutil.rmtree(out)
    result = run_cnn_training(fno_train_cfg, out, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model"] == "fno"
