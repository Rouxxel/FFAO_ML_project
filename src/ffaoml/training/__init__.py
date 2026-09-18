"""
#############################################################################
### Training loops and losses
###
### @file training/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Stage 1 CNN training and checkpoint bundle export.
"""

# Project imports
from ffaoml.training.train import TrainingResult, run_cnn_training

__all__ = ["TrainingResult", "run_cnn_training"]
