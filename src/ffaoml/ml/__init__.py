"""
#############################################################################
### Machine learning data and training helpers
###
### @file ml/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

PyTorch datasets, normalization, and split utilities (Stage 1 temporal windows).
"""

# Project imports
from ffaoml.ml.dataset import FlowDataset, build_flow_datasets
from ffaoml.ml.preprocessing import (
    PreprocessStats,
    fit_preprocess_stats,
    load_preprocess_stats,
    save_preprocess_stats,
)

__all__ = [
    "FlowDataset",
    "PreprocessStats",
    "build_flow_datasets",
    "fit_preprocess_stats",
    "load_preprocess_stats",
    "save_preprocess_stats",
]
