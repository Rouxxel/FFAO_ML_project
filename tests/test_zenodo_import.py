"""
#############################################################################
### Stage 1 Zenodo import tests (offline)
###
### @file test_zenodo_import.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Parse synthetic upstream files and run the import pipeline without network access.
"""

# Native imports
from pathlib import Path

# Third-party imports
import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.io import open_field_store
from ffaoml.data.sources.zenodo_re100 import (
    import_stage1_from_config,
    parse_upstream_file,
)
from ffaoml.manifests import DatasetManifest, dataset_manifest_path

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / ".local_test_runs" / "zenodo_import"

"""FIXTURES-----------------------------------------------------------"""


@pytest.fixture
def synthetic_h5() -> Path:
    """Minimal HDF5 matching the Addiucci layout."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / "cylinder_re100_grid64_last100.h5"
    n_time, ny, nx = 12, 8, 8
    rng = np.random.default_rng(0)
    fields = rng.standard_normal((n_time, 3, ny, nx), dtype=np.float32)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("grid_x", data=x)
        handle.create_dataset("grid_y", data=y)
    return path


"""TESTS-----------------------------------------------------------"""


def test_parse_hdf5_shapes(synthetic_h5: Path) -> None:
    parsed = parse_upstream_file(synthetic_h5, dt=0.5)
    assert parsed.velocity_x.shape == (12, 8, 8)
    assert parsed.vorticity.shape == (12, 8, 8)
    assert parsed.time.shape == (12,)
    assert parsed.time[1] == pytest.approx(0.5)


def test_parse_hdf5_uv_separate_datasets() -> None:
    """Alternate HDF5 with u/v/vorticity keys (e.g. external LBM scripts)."""
    path = WORK_ROOT / "uv_layout.h5"
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    n_time, ny, nx = 5, 6, 7
    rng = np.random.default_rng(1)
    u = rng.standard_normal((n_time, ny, nx), dtype=np.float32)
    v = rng.standard_normal((n_time, ny, nx), dtype=np.float32)
    w = rng.standard_normal((n_time, ny, nx), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("vorticity", data=w)
    parsed = parse_upstream_file(path, dt=1.0)
    assert parsed.velocity_x.shape == (5, 6, 7)
    assert parsed.attrs.get("upstream_format") == "hdf5_uv"


def test_import_stage1_writes_zarr_and_manifest(synthetic_h5: Path) -> None:
    dataset_root = WORK_ROOT / "dataset_out"
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
            ],
        )
    result = import_stage1_from_config(
        cfg,
        local_upstream=synthetic_h5,
        repo_root=REPO_ROOT,
    )
    assert result.n_steps == 12
    assert result.store_path.is_dir()
    ds = open_field_store(result.simulation_dir)
    for channel in DEFAULT_FIELD_CHANNELS:
        assert channel in ds
    manifest = DatasetManifest.read_json(dataset_manifest_path(dataset_root))
    assert manifest.stage == 1
    assert manifest.source_id == "zenodo_re100"
