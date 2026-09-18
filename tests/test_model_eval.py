"""
#############################################################################
### Model evaluation figure tests
###
### @file test_model_eval.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Phase 3 figures from a short CNN training run (requires PyTorch).
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
from ffaoml.evaluation.model_report import run_model_evaluation
from ffaoml.training.train import run_cnn_training

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO_ROOT / "tests" / "fixtures" / "stage1_mini.h5"
WORK = REPO_ROOT / ".local_test_runs" / "model_eval"


@pytest.fixture(scope="module")
def trained_run_dir():
    if not FIXTURE_H5.is_file():
        pytest.skip(f"missing fixture {FIXTURE_H5}")
    dataset_root = WORK / "dataset"
    run_dir = WORK / "run"
    for path in (dataset_root, run_dir):
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
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
                "eval.split=val",
                "eval.rollout_horizon=2",
                "train.epochs=6",
                "train.batch_size=2",
            ],
        )
    import_stage1_from_config(cfg, local_upstream=FIXTURE_H5, repo_root=REPO_ROOT)
    run_cnn_training(cfg, run_dir, repo_root=REPO_ROOT)
    return run_dir


def test_model_evaluation_writes_figures(trained_run_dir: Path) -> None:
    result = run_model_evaluation(trained_run_dir, split="val")
    assert result.vorticity_figure.is_file()
    assert result.horizon_figure.is_file()
    assert result.stability_figure.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "cnn" in payload["rollout"]
    assert "persistence" in payload["rollout"]
