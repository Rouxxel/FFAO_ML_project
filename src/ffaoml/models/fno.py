"""
#############################################################################
### Fourier Neural Operator (FNO) flow predictor
###
### @file fno.py
### @author Sebastian Russo
### @date 2026
#############################################################################

One-step grid predictor using ``neuraloperator`` (stretch model, PRD §10).
"""

# Native imports
from __future__ import annotations

from typing import Any

# Third-party imports
from omegaconf import DictConfig

try:
    import torch
    from torch import nn

    try:
        from neuralop.models import FNO as NeuralOpFNO
    except ImportError:  # pragma: no cover
        NeuralOpFNO = None  # type: ignore[assignment,misc]

    class FlowFNO(nn.Module):
        """
        Wrapper around ``neuralop.models.FNO`` for CONTRACT channel layouts.

        Input/output: ``(batch, channels, height, width)``.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_modes: tuple[int, int],
            hidden_channels: int = 64,
            n_layers: int = 4,
        ) -> None:
            super().__init__()
            if NeuralOpFNO is None:
                raise ImportError(
                    "neuraloperator is required for FlowFNO; "
                    "install with pip install -e '.[ml]'"
                )
            self.net = NeuralOpFNO(
                n_modes=n_modes,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Predict the next flow state.

            Parameters:
                x (torch.Tensor): ``(B, C, H, W)``.

            Returns:
                torch.Tensor: Next-state prediction, same spatial shape.
            """
            return self.net(x)

except ImportError:  # pragma: no cover

    class FlowFNO:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "torch is required for FlowFNO; install with pip install -e '.[ml]'"
            )


def build_flow_fno(cfg: DictConfig) -> FlowFNO:
    """
    Construct an FNO from Hydra ``model`` config.

    Parameters:
        cfg (DictConfig): Composed config with ``model`` group.

    Returns:
        FlowFNO: Initialized module (not on device yet).
    """
    model = cfg.model
    modes = model.get("n_modes", [16, 16])
    n_modes = (int(modes[0]), int(modes[1]))
    in_channels = int(model.in_channels)
    if bool(model.get("condition_on_re", False)):
        in_channels += 1
    return FlowFNO(
        in_channels=in_channels,
        out_channels=int(model.out_channels),
        n_modes=n_modes,
        hidden_channels=int(model.hidden_channels),
        n_layers=int(model.n_layers),
    )
