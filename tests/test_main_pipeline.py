"""
#############################################################################
### Root main.py pipeline tests
###
### @file test_main_pipeline.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Dry-run plan for Stage 1 orchestrator (no training).
"""

# Native imports
from pathlib import Path

# Project imports
from ffaoml.pipeline.stage1 import (
    PHASE_ORDER,
    Stage1PipelineOptions,
    _check_prerequisites,
    compose_stage1_config,
    run_stage1_pipeline,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage1_pipeline_dry_run() -> None:
    opts = Stage1PipelineOptions(
        repo_root=REPO_ROOT,
        dry_run=True,
        with_convlstm=True,
    )
    results = run_stage1_pipeline(opts)
    assert results["status"] == "dry_run"
    assert "import" in results["plan"]["phases"]
    assert "stage1_import" in results["plan"]
    assert "train_cnn" in results["plan"]["phases"]
    assert "compare_multistep" in results["plan"]["phases"]


def test_prerequisites_allow_import_before_dataset_dependent_steps() -> None:
    """Fresh machines should not fail prereq check before IMPORT runs."""
    opts = Stage1PipelineOptions(repo_root=REPO_ROOT, with_convlstm=True)
    cfg = compose_stage1_config(REPO_ROOT)
    phases = list(PHASE_ORDER)
    _check_prerequisites(phases, opts, cfg)
