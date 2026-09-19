"""
#############################################################################
### Baseline rollout and evaluation runner tests
###
### @file test_evaluation.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Rollout curves and ``metrics.json`` export on the Stage 1 mini fixture.
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.evaluation.baselines import predict_persistence
from ffaoml.evaluation.rollout import one_step_baseline_metrics, rollout_curve
from ffaoml.evaluation.runner import run_baseline_evaluation

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "eval_phase1"


def test_persistence_predictor_is_identity() -> None:
    x = np.ones((2, 3, 4), dtype=np.float32)
    assert np.allclose(predict_persistence(x), x)


def test_rollout_curve_on_synthetic_sine() -> None:
    n_time = 32
    t = np.arange(n_time)
    signal = np.sin(0.4 * t)[:, None, None, None]
    series = np.broadcast_to(signal, (n_time, 2, 4, 4)).astype(np.float32)
    curve = rollout_curve(series, "persistence", horizon=5)
    assert len(curve.horizons) == 5
    assert curve.mse[0] <= curve.mse[-1]


def test_rollout_linear_on_short_val_window() -> None:
    """Stage 1 val temporal split has 15 frames; horizon 50 must cap, not fail."""
    n_time = 15
    series = np.random.default_rng(0).standard_normal((n_time, 2, 4, 4)).astype(
        np.float32
    )
    curve = rollout_curve(series, "linear", horizon=50)
    assert curve.n_starts >= 1
    assert len(curve.horizons) == 13


@pytest.fixture(scope="module")
def eval_cfg():
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
                "eval.split=train",
                "eval.rollout_horizon=2",
                "eval.report_horizons=[1,2]",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    return cfg


def test_baseline_evaluation_writes_metrics_json(eval_cfg) -> None:
    out = WORK / "run"
    if out.is_dir():
        import shutil

        shutil.rmtree(out)
    result = run_baseline_evaluation(eval_cfg, out, repo_root=REPO_ROOT)
    assert result.metrics_path.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "persistence" in payload["baselines"]
    assert "linear" in payload["baselines"]
    assert (out / "config.yaml").is_file()
    assert (out / "preprocess_stats.json").is_file()
    one = one_step_baseline_metrics(
        np.random.randn(6, 2, 3, 3).astype(np.float32),
        "persistence",
    )
    assert "mse" in one
