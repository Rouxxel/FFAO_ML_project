"""
#############################################################################
### Model evaluation and baselines
###
### @file evaluation/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Metrics, naive baselines, multi-step rollout, and evaluation runners.
"""

# Project imports
from ffaoml.evaluation.baselines import BaselineName, predict_next
from ffaoml.evaluation.metrics import mse, relative_l2
from ffaoml.evaluation.runner import run_baseline_evaluation

__all__ = [
    "BaselineName",
    "mse",
    "predict_next",
    "relative_l2",
    "run_baseline_evaluation",
]
