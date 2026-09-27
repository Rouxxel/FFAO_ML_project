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
from ffaoml.models.factory import build_flow_model
from ffaoml.models.fno import FlowFNO, build_flow_fno
from ffaoml.models.meshgraphnet import MeshGraphNet, build_meshgraphnet

__all__ = [
    "FlowCNN",
    "FlowConvLSTM",
    "FlowFNO",
    "MeshGraphNet",
    "build_flow_cnn",
    "build_flow_convlstm",
    "build_flow_fno",
    "build_flow_model",
    "build_meshgraphnet",
]
