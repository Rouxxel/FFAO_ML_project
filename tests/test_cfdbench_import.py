"""
#############################################################################
### CFDBench import tests (no HF download in default CI)
###
### @file test_cfdbench_import.py
### @date 2026
#############################################################################
"""

import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.data.catalog import open_simulation_from_config
from ffaoml.data.loading import load_simulation_tensor
from ffaoml.data.metadata import read_metadata_rows
from ffaoml.data.sources.cfdbench_cylinder import (
    CfdbenchCaseRef,
    cache_dir_from_config,
    import_cfdbench_case,
    run_cfdbench_import,
    simulation_split_for_re,
    stage_upstream_case_layout,
)
from ffaoml.manifests import dataset_manifest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cfdbench_upstream_case"
SCRATCH = REPO_ROOT / ".local_test_runs" / "cfdbench_import"


def _stage3_cfg(output_root: Path, **extra_overrides: str):
    GlobalHydra.instance().clear()
    overrides = [
        "dataset=stage3_cfdbench",
        f"dataset.output_root={output_root.as_posix()}",
        "dataset.train_re=[8,10,12]",
        "dataset.val_re=[10]",
        "dataset.test_re=[20]",
        "dataset.temporal_split.train=[0,4]",
        "dataset.temporal_split.val=[0,4]",
        "dataset.temporal_split.test=[0,4]",
    ]
    overrides.extend(f"{k}={v}" for k, v in extra_overrides.items())
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def test_simulation_split_for_re_uses_config_lists() -> None:
    cfg = _stage3_cfg(SCRATCH / "split_only")
    assert simulation_split_for_re(10.0, cfg, tolerance=1.0) == "train"
    assert simulation_split_for_re(20.0, cfg, tolerance=1.0) == "test"
    assert simulation_split_for_re(999.0, cfg, tolerance=1.0) is None


def test_import_cfdbench_case_writes_zarr_and_metadata() -> None:
    if not UPSTREAM_FIXTURE.is_dir():
        pytest.skip("missing cfdbench_upstream_case fixture")
    out_root = SCRATCH / "single_case"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    cfg = _stage3_cfg(out_root)
    ref = CfdbenchCaseRef(
        case_dir=UPSTREAM_FIXTURE,
        subset="prop",
        case_name="case0000",
    )
    dest = import_cfdbench_case(cfg, ref, split="train", re=10.0, dt=0.1)
    assert dest.is_dir()
    rows = read_metadata_rows(out_root)
    assert len(rows) == 1
    store = open_simulation_from_config(cfg, sim_id=rows[0]["sim_id"])
    assert "velocity_x" in store


def test_run_cfdbench_import_from_staged_fixture() -> None:
    if not UPSTREAM_FIXTURE.is_dir():
        pytest.skip("missing cfdbench_upstream_case fixture")
    data_root = SCRATCH / "staged_data"
    out_root = SCRATCH / "from_fixture" / "cfdbench_data"
    if data_root.is_dir():
        shutil.rmtree(data_root)
    if out_root.is_dir():
        shutil.rmtree(out_root)
    stage_upstream_case_layout(UPSTREAM_FIXTURE, data_root)
    cfg = _stage3_cfg(out_root)
    result = run_cfdbench_import(
        cfg,
        repo_root=REPO_ROOT,
        local_data_root=data_root,
        re_filter=[10.0],
        max_cases=1,
    )
    assert result.cases_imported == 1
    assert dataset_manifest_path(out_root).is_file()
    rows = read_metadata_rows(out_root)
    assert len(rows) == 1
    tensor = load_simulation_tensor(cfg, rows[0]["sim_id"], "train")
    assert tensor.shape[0] == 4
    assert tensor.shape[1] == 4


@pytest.mark.slow
def test_import_one_case_from_user_cache() -> None:
    cfg = _stage3_cfg(SCRATCH / "from_cache" / "cfdbench_data")
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cache_cfg = compose(
            config_name="config",
            overrides=["dataset=stage3_cfdbench"],
        )
    cache = cache_dir_from_config(cache_cfg)
    if not (cache / "data" / "cylinder").is_dir() and not (cache / "cylinder").is_dir():
        pytest.skip("no .cache/cfdbench download; run download script locally first")
    try:
        result = run_cfdbench_import(
            cfg,
            repo_root=REPO_ROOT,
            force_download=False,
            max_cases=1,
        )
    except ImportError:
        pytest.skip("huggingface_hub not installed")
    assert result.cases_imported >= 0
