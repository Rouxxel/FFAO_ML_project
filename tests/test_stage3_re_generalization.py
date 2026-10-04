"""
#############################################################################
### Stage 3 re_generalization on cfdbench_mini bundle
###
### @file test_stage3_re_generalization.py
### @date 2026
#############################################################################
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.contracts import STAGE3_CFDBENCH_SOURCE_ID
from ffaoml.evaluation.re_generalization import (
    run_re_generalization_evaluation,
    validate_re_generalization_run,
)
from ffaoml.training.train import run_cnn_training

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cfdbench_mini"
WORK = REPO_ROOT / ".local_test_runs" / "stage3_re_generalization"

STAGE3_OVERRIDES = [
    "dataset=stage3_cfdbench",
    "model=cnn_re",
    f"dataset.output_root={FIXTURE_ROOT.as_posix()}",
    "dataset.train_re=[50,100]",
    "dataset.val_re=[125]",
    "dataset.test_re=[250]",
    "dataset.temporal_split.train=[0,4]",
    "dataset.temporal_split.val=[0,4]",
    "dataset.temporal_split.test=[0,4]",
    "train.epochs=2",
    "train.batch_size=2",
    "eval.rollout_horizon=3",
    "eval.report_horizons=[1,2,3]",
]


def _compose_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config", overrides=STAGE3_OVERRIDES)
    GlobalHydra.instance().clear()
    return cfg


def test_stage3_cnn_re_generalization_with_rollout() -> None:
    if not (FIXTURE_ROOT / "metadata.csv").is_file():
        pytest.skip("missing cfdbench_mini fixture")
    run_dir = WORK / "stage3_cnn_re"
    if run_dir.is_dir():
        shutil.rmtree(run_dir)

    cfg = _compose_cfg()
    assert str(cfg.dataset.source_id) == STAGE3_CFDBENCH_SOURCE_ID
    run_cnn_training(cfg, run_dir, repo_root=REPO_ROOT)
    checks = validate_re_generalization_run(cfg, run_dir)
    assert checks["source_id"] == STAGE3_CFDBENCH_SOURCE_ID

    result = run_re_generalization_evaluation(
        run_dir,
        include_rollout=True,
        rollout_horizon=3,
    )
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["bundle_checks"]["source_id"] == STAGE3_CFDBENCH_SOURCE_ID
    assert payload["rollout"]["per_simulation"]
    assert result.rollout_horizon_path is not None
    assert result.rollout_horizon_path.is_file()
    assert "exp1_in_distribution" in payload["experiments"]
