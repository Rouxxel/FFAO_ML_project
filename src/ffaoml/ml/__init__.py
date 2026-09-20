"""
#############################################################################
### Machine learning data and training helpers
###
### @file ml/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

PyTorch datasets, normalization, and split utilities (Stage 1 temporal windows;
Stage 2 mesh graphs).
"""

# Project imports
from ffaoml.ml.dataset import FlowDataset, build_flow_datasets
from ffaoml.ml.mesh_dataset import MeshGraphDataset, build_mesh_datasets
from ffaoml.ml.mesh_preprocessing import (
    MeshPreprocessStats,
    fit_mesh_preprocess_stats,
    load_mesh_preprocess_stats,
    save_mesh_preprocess_stats,
)
from ffaoml.ml.preprocessing import (
    PreprocessStats,
    fit_preprocess_stats,
    load_preprocess_stats,
    save_preprocess_stats,
)

__all__ = [
    "FlowDataset",
    "MeshGraphDataset",
    "MeshPreprocessStats",
    "PreprocessStats",
    "build_flow_datasets",
    "build_mesh_datasets",
    "fit_mesh_preprocess_stats",
    "fit_preprocess_stats",
    "load_mesh_preprocess_stats",
    "load_preprocess_stats",
    "save_mesh_preprocess_stats",
    "save_preprocess_stats",
]
