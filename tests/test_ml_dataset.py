"""
#############################################################################
### FlowDataset and DataLoader smoke tests
###
### @file test_ml_dataset.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Stage 1 ML Phase 0: normalization fit on train times only; one-step windows.
"""

# Native imports
from pathlib import Path

# Third-party imports
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.loading import load_split_tensor
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.ml.dataset import build_flow_datasets
from ffaoml.ml.preprocessing import fit_preprocess_stats, normalize_fields

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "ml_dataset"


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
                "dataset.temporal_split.val=[4,5]",
                "dataset.temporal_split.test=[5,6]",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_normalization_uses_train_stats_only(stage1_cfg) -> None:
    stats = fit_preprocess_stats(stage1_cfg)
    assert stats.fit_split == "train"
    train = normalize_fields(load_split_tensor(stage1_cfg, "train"), stats)
    train_mean = train.mean(axis=(0, 2, 3))
    assert np.allclose(train_mean, 0.0, atol=1e-5)


def test_flow_dataset_one_step_shapes(stage1_cfg) -> None:
    datasets = build_flow_datasets(stage1_cfg, delta_steps=1)
    train = datasets["train"]
    assert len(train) == 3
    sample = train[0]
    assert sample["input"].shape == sample["target"].shape == (4, 10, 12)
    assert sample["re"] == 100.0


def test_pytorch_dataloader_smoke(stage1_cfg) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    train = build_flow_datasets(stage1_cfg)["train"]
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input"].shape == (2, 4, 10, 12)
    assert isinstance(batch["input"], torch.Tensor)
