"""
#############################################################################
### Checkpoint load helpers
###
### @file checkpointing.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Normalize PyTorch checkpoints (e.g. strip ``neuralop`` ``_metadata`` keys).
"""

# Native imports
from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover
    nn = None  # type: ignore[assignment]

"""LOADERS-----------------------------------------------------------"""


def extract_state_dict(payload: Any) -> dict[str, Any]:
    """
    Pull a weight dict from a training checkpoint file payload.

    Parameters:
        payload (Any): ``torch.load`` result or raw state dict.

    Returns:
        dict[str, Any]: Model weights without private keys.
    """
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dict")
    return {k: v for k, v in state.items() if not str(k).startswith("_")}


def load_model_weights(model: nn.Module, payload: Any) -> None:
    """
    Load weights into ``model``, ignoring vendor-specific metadata keys.

    Parameters:
        model (nn.Module): Target module.
        payload (Any): Checkpoint object from ``torch.load``.

    Returns:
        None
    """
    model.load_state_dict(extract_state_dict(payload))
