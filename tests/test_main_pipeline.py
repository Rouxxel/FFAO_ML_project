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
from ffaoml.pipeline.stage1 import Stage1PipelineOptions, run_stage1_pipeline

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
    assert "train_cnn" in results["plan"]["phases"]
    assert "compare_multistep" in results["plan"]["phases"]
