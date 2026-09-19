"""
#############################################################################
### Task A reconstruction dataset and training tests
###
### @file test_reconstruction.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Velocity-only input → full-field reconstruction (requires PyTorch).
"""

# Native imports
from pathlib import Path

# Third-party imports
import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.ml.preprocessing import fit_preprocess_stats
from ffaoml.ml.reconstruction import (
    FlowReconstructionDataset,
)
from ffaoml.models.factory import build_flow_model
from ffaoml.training.train import run_cnn_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "reconstruct"


@pytest.fixture(scope="module")
def recon_cfg():
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
                "model=reconstruct",
                "train.epochs=6",
                "train.batch_size=2",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_reconstruction_dataset_shapes(recon_cfg) -> None:
    stats = fit_preprocess_stats(recon_cfg)
    ds = FlowReconstructionDataset(recon_cfg, "train", stats)
    sample = ds[0]
    assert sample["input"].shape[0] == 2
    assert sample["target"].shape[0] == 4


def test_reconstruction_training_run(recon_cfg) -> None:
    out = WORK / "run"
    if out.is_dir():
        import shutil

        shutil.rmtree(out)
    import torch

    model = build_flow_model(recon_cfg)
    x = torch.randn(1, 2, 10, 12)
    y = model(x)
    assert y.shape == (1, 4, 10, 12)
    result = run_cnn_training(recon_cfg, out, repo_root=REPO_ROOT)
    assert result.model_path.is_file()
