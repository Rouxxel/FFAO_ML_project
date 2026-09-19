"""
#############################################################################
### Multi-Re conditioning and generalization tests
###
### @file test_re_generalization.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Stub multi-Re dataset, Re-conditioned CNN training, and heatmap eval (PyTorch).
"""

# Native imports
import importlib.util
import json
from pathlib import Path

# Third-party imports
import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.evaluation.re_generalization import run_re_generalization_evaluation
from ffaoml.ml.conditioning import ReScaling, append_re_channel
from ffaoml.ml.dataset import FlowMultiReDataset
from ffaoml.ml.preprocessing import fit_preprocess_stats
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.training.train import run_cnn_training
"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = REPO_ROOT / ".local_test_runs" / "re_generalization"


def _build_multi_re_stub_dataset(output_root: Path, **kwargs) -> Path:
    path = REPO_ROOT / "tests" / "fixtures" / "build_multi_re_stub_dataset.py"
    spec = importlib.util.spec_from_file_location("build_multi_re_stub", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_multi_re_stub_dataset(output_root, **kwargs)

SPLIT_OVERRIDES = [
    "dataset=splits",
    "model=cnn_re",
    "dataset.train_re=[50,100]",
    "dataset.val_re=[125]",
    "dataset.test_re=[250]",
    "dataset.temporal_split.train=[0,4]",
    "dataset.temporal_split.val=[0,4]",
    "dataset.temporal_split.test=[0,4]",
    "train.epochs=10",
    "train.batch_size=4",
]


@pytest.fixture(scope="module")
def multire_cfg():
    dataset_root = WORK / "dataset"
    if dataset_root.is_dir():
        import shutil

        shutil.rmtree(dataset_root)
    _build_multi_re_stub_dataset(dataset_root, n_steps=8)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.dataset_root={dataset_root.as_posix()}",
                f"dataset.output_root={dataset_root.as_posix()}",
                *SPLIT_OVERRIDES,
            ],
        )
    return cfg


def test_append_re_channel_increases_depth() -> None:
    import numpy as np

    frame = np.zeros((4, 6, 8), dtype=np.float32)
    out = append_re_channel(frame, ReScaling(50, 200).scale(100))
    assert out.shape == (5, 6, 8)


def test_flow_multire_dataset_length(multire_cfg) -> None:
    stats = fit_preprocess_stats(multire_cfg)
    train_ds = FlowMultiReDataset(multire_cfg, "train", stats)
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["input"].shape[0] == 5
    assert sample["target"].shape[0] == 4


def test_cnn_re_forward_shape(multire_cfg) -> None:
    import torch

    model = build_flow_cnn(multire_cfg)
    x = torch.randn(2, 5, 8, 10)
    y = model(x)
    assert y.shape == (2, 4, 8, 10)


def test_re_generalization_pipeline(multire_cfg) -> None:
    run_dir = WORK / "run"
    if run_dir.is_dir():
        import shutil

        shutil.rmtree(run_dir)
    run_cnn_training(multire_cfg, run_dir, repo_root=REPO_ROOT)
    result = run_re_generalization_evaluation(run_dir)
    assert result.heatmap_path.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "exp3_extrapolation" in payload["experiments"]
    assert payload["experiments"]["exp3_extrapolation"]
