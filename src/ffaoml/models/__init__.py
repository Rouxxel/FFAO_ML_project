"""
#############################################################################
### Neural network architectures
###
### @file models/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Model definitions (forward pass only; training lives in ``ffaoml.training``).
"""

# Project imports
from ffaoml.models.cnn import FlowCNN, build_flow_cnn

__all__ = ["FlowCNN", "build_flow_cnn"]
