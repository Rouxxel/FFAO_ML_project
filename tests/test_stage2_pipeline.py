"""
#############################################################################
### Stage 2 pipeline orchestration tests
###
### @file test_stage2_pipeline.py
### @date 2026
#############################################################################
"""

from pathlib import Path

from ffaoml.pipeline.stage2 import (
    STAGE2_PHASE_ORDER,
    Stage2PipelineOptions,
    _check_prerequisites,
    compose_stage2_config,
    hydra_overrides_select_stage2,
    run_stage2_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage2_pipeline_dry_run() -> None:
    opts = Stage2PipelineOptions(repo_root=REPO_ROOT, dry_run=True)
    results = run_stage2_pipeline(opts)
    assert results["status"] == "dry_run"
    assert results["plan"]["stage"] == 2
    assert [p.value for p in STAGE2_PHASE_ORDER] == results["plan"]["phases"]


def test_stage2_prerequisites_allow_import_before_train() -> None:
    opts = Stage2PipelineOptions(repo_root=REPO_ROOT)
    cfg = compose_stage2_config(REPO_ROOT)
    phases = list(STAGE2_PHASE_ORDER)
    _check_prerequisites(phases, opts, cfg)


def test_hydra_stage2_selector() -> None:
    assert hydra_overrides_select_stage2(["dataset=stage2_meshgraphnets"])
    assert not hydra_overrides_select_stage2(["train.epochs=3"])
