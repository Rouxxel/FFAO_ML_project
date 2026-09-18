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
from ffaoml.models.convlstm import FlowConvLSTM, build_flow_convlstm

__all__ = [
    "FlowCNN",
    "FlowConvLSTM",
    "build_flow_cnn",
    "build_flow_convlstm",
]
