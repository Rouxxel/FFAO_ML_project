"""
#############################################################################
### Stage 1 validation script tests
###
### @file test_stage1_validation.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Run validation on the imported ``stage1_mini`` fixture (offline, headless).
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.validation.stage1_zenodo import (
    estimate_shedding_st,
    run_stage1_validation,
    wake_vorticity_proxy,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "stage1_validation"


def test_validation_writes_artifacts() -> None:
    if not FIXTURE_H5.is_file():
        pytest.skip(f"missing fixture {FIXTURE_H5}")
    dataset_root = WORK / "dataset"
    out_dir = WORK / "report"
    if dataset_root.is_dir():
        import shutil

        shutil.rmtree(dataset_root)
    if out_dir.is_dir():
        shutil.rmtree(out_dir)

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
    result = run_stage1_validation(
        cfg,
        output_dir=out_dir,
        fps=4,
        max_animation_frames=6,
    )
    assert result.summary_md.is_file()
    assert result.vorticity_snapshots.is_file()
    assert result.shedding_animation.is_file()
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["forces_available"] is False
    assert "Cd" in result.summary_md.read_text(encoding="utf-8")


def test_wake_proxy_shape_and_st_on_sine() -> None:
    import numpy as np

    n_time = 256
    dt = 0.05
    f0 = 2.0
    t = np.arange(n_time) * dt
    signal = np.sin(2 * np.pi * f0 * t)
    freq, st = estimate_shedding_st(signal, dt, 1.0, 0.1, min_frequency=0.5)
    assert abs(freq - f0) < 0.1
    assert abs(st - f0 * 0.1) < 0.02

    frame = signal[:, None, None] * np.ones((n_time, 4, 4), dtype=np.float32)
    proxy = wake_vorticity_proxy(frame)
    assert proxy.shape == (n_time,)
