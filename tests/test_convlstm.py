"""
#############################################################################
### ConvLSTM and multi-step comparison tests
###
### @file test_convlstm.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Forward pass, short training, and CNN vs ConvLSTM compare (requires PyTorch).
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
from ffaoml.evaluation.multistep_compare import run_multistep_comparison
from ffaoml.models.convlstm import build_flow_convlstm
from ffaoml.training.train import run_cnn_training
from ffaoml.training.train_convlstm import run_convlstm_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "convlstm"


def test_flow_convlstm_forward_unroll_shape() -> None:
    import torch

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=["model=convlstm"])
    model = build_flow_convlstm(cfg)
    x = torch.randn(2, 4, 10, 12)
    y, _ = model(x)
    assert y.shape == x.shape
    seq = model.forward_unroll(x, 3, teacher_forcing=False)
    assert seq.shape == (2, 3, 4, 10, 12)


@pytest.fixture(scope="module")
def stage1_cfg():
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
                "eval.split=val",
                "eval.rollout_horizon=2",
                "eval.report_horizons=[1,2]",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_convlstm_training_writes_checkpoint(stage1_cfg) -> None:
    out = WORK / "convlstm_run"
    if out.is_dir():
        import shutil

        shutil.rmtree(out)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.dataset_root={stage1_cfg.paths.dataset_root}",
                "dataset.temporal_split.train=[0,4]",
                "dataset.temporal_split.val=[3,6]",
                "train.epochs=8",
                "train.batch_size=2",
                "model=convlstm",
            ],
        )
    result = run_convlstm_training(cfg, out, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model"] == "convlstm"
    assert summary["beats_persistence"] is True


def test_multistep_compare_json(stage1_cfg) -> None:
    cnn_dir = WORK / "cnn_run"
    lstm_dir = WORK / "lstm_run"
    for path in (cnn_dir, lstm_dir):
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        base = [
            f"paths.dataset_root={stage1_cfg.paths.dataset_root}",
            "dataset.temporal_split.train=[0,4]",
            "dataset.temporal_split.val=[3,6]",
            "train.epochs=6",
            "train.batch_size=2",
            "eval.split=val",
            "eval.rollout_horizon=2",
            "eval.report_horizons=[1,2]",
        ]
        cnn_cfg = compose(config_name="config", overrides=base)
        lstm_cfg = compose(config_name="config", overrides=[*base, "model=convlstm"])
    run_cnn_training(cnn_cfg, cnn_dir, repo_root=REPO_ROOT)
    run_convlstm_training(lstm_cfg, lstm_dir, repo_root=REPO_ROOT)
    result = run_multistep_comparison(cnn_dir, lstm_dir, split="val")
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "cnn_recursive" in payload["rollout"]
    assert "convlstm" in payload["rollout"]
    assert payload["rollout"]["cnn_recursive"]["at_report_horizons"]["mse"]["1"] > 0
