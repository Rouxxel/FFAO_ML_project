"""
#############################################################################
### Stage 3 pipeline orchestration tests
###
### @file test_stage3_pipeline.py
### @date 2026
#############################################################################
"""

from pathlib import Path

from ffaoml.pipeline.stage3 import (
    STAGE3_PHASE_ORDER,
    Stage3PipelineOptions,
    _check_prerequisites,
    compose_stage3_config,
    hydra_overrides_select_stage3,
    run_stage3_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage3_pipeline_dry_run() -> None:
    opts = Stage3PipelineOptions(repo_root=REPO_ROOT, dry_run=True)
    results = run_stage3_pipeline(opts)
    assert results["status"] == "dry_run"
    assert results["plan"]["stage"] == 3
    assert [p.value for p in STAGE3_PHASE_ORDER] == results["plan"]["phases"]


def test_stage3_prerequisites_allow_import_before_train() -> None:
    opts = Stage3PipelineOptions(repo_root=REPO_ROOT)
    cfg = compose_stage3_config(REPO_ROOT)
    phases = list(STAGE3_PHASE_ORDER)
    _check_prerequisites(phases, opts, cfg)


def test_hydra_stage3_selector() -> None:
    assert hydra_overrides_select_stage3(["dataset=stage3_cfdbench"])
    assert not hydra_overrides_select_stage3(["train.epochs=3"])
