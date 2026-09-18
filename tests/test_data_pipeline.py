"""
#############################################################################
### Dataset I/O pipeline integration tests
###
### @file test_data_pipeline.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import the committed fixture, read temporal windows, and verify ML tensor shapes.
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
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.catalog import resolve_simulation_dir, temporal_index_range
from ffaoml.data.io import open_field_store, read_time_window, stack_fields_tensor
from ffaoml.data.loading import load_split_tensor, split_shape
from ffaoml.data.resample import downsample_spatial
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
PIPELINE_ROOT = REPO_ROOT / ".local_test_runs" / "data_pipeline"

"""FIXTURES-----------------------------------------------------------"""


@pytest.fixture(scope="module")
def imported_dataset() -> Path:
    """Import ``stage1_mini.h5`` once into a local dataset root."""
    if not FIXTURE_H5.is_file():
        pytest.skip(f"missing fixture {FIXTURE_H5}")
    dataset_root = PIPELINE_ROOT / "dataset"
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
    return dataset_root


"""TESTS-----------------------------------------------------------"""


def test_downsample_reduces_resolution() -> None:
    ny, nx = 20, 30
    fields = {
        "velocity_x": np.ones((3, ny, nx), dtype=np.float32),
        "velocity_y": np.zeros((3, ny, nx), dtype=np.float32),
        "pressure": np.zeros((3, ny, nx), dtype=np.float32),
        "vorticity": np.zeros((3, ny, nx), dtype=np.float32),
    }
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    out, xo, yo, meta = downsample_spatial(fields, x, y, target_nx=10, target_ny=8)
    assert out["velocity_x"].shape == (3, 8, 10)
    assert meta["downsampled"] is True
    assert meta["nx_original"] == nx


def test_fixture_import_and_ml_window(imported_dataset: Path) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.dataset_root={imported_dataset.as_posix()}",
                "dataset.temporal_split.train=[0,4]",
                "dataset.temporal_split.val=[4,5]",
                "dataset.temporal_split.test=[5,6]",
            ],
        )
    sim_dir = resolve_simulation_dir(cfg)
    ds = open_field_store(sim_dir)
    start, end = temporal_index_range(cfg, "train")
    window = read_time_window(ds, start, end)
    tensor = stack_fields_tensor(window)
    assert tensor.shape == (end - start, len(DEFAULT_FIELD_CHANNELS), 10, 12)
    assert tensor.dtype == np.float32

    loaded = load_split_tensor(cfg, "train")
    assert loaded.shape == split_shape(cfg, "train")
