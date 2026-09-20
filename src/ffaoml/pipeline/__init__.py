"""
#############################################################################
### End-to-end pipeline orchestration
###
### @file pipeline/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Stage 1 (and future) multi-step runners invoked from repo-root ``main.py``.
"""

# Project imports
from ffaoml.pipeline.stage1 import PipelinePhase, run_stage1_pipeline
from ffaoml.pipeline.stage2 import (
    STAGE2_PHASE_ORDER,
    Stage2PipelineOptions,
    Stage2PipelinePhase,
    run_stage2_pipeline,
)

__all__ = [
    "PipelinePhase",
    "STAGE2_PHASE_ORDER",
    "Stage2PipelineOptions",
    "Stage2PipelinePhase",
    "run_stage1_pipeline",
    "run_stage2_pipeline",
]
