"""
#############################################################################
### Model factory
###
### @file factory.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Build predictors from Hydra ``model.name``.
"""

# Native imports
from __future__ import annotations

# Third-party imports
from omegaconf import DictConfig

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover
    nn = None  # type: ignore[assignment]

# Project imports
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.models.convlstm import build_flow_convlstm
from ffaoml.models.fno import build_flow_fno
from ffaoml.models.meshgraphnet import build_meshgraphnet

"""FACTORY-----------------------------------------------------------"""


def build_flow_model(cfg: DictConfig) -> nn.Module:
    """
    Instantiate a flow predictor from ``cfg.model.name``.

    Parameters:
        cfg (DictConfig): Composed Hydra config.

    Returns:
        nn.Module: ``FlowCNN``, ``FlowFNO``, or ``FlowConvLSTM``.

    Raises:
        ValueError: For unknown model names.
    """
    name = str(cfg.model.name)
    if name == "cnn":
        return build_flow_cnn(cfg)
    if name == "fno":
        return build_flow_fno(cfg)
    if name == "convlstm":
        return build_flow_convlstm(cfg)
    if name == "meshgraphnet":
        return build_meshgraphnet(cfg)
    raise ValueError(f"unknown model.name: {name}")
