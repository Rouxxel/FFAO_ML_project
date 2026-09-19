"""
#############################################################################
### One-step flow CNN
###
### @file cnn.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Convolutional mapper ``flow(t) → flow(t+Δt)`` on structured grids (Task B).
"""

# Native imports
from __future__ import annotations

# Third-party imports
from omegaconf import DictConfig

try:
    import torch
    from torch import nn

    class FlowCNN(nn.Module):
        """
        Shallow CNN for one-step field prediction.

        Input/output shape: ``(batch, channels, height, width)``.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int = 64,
            kernel_size: int = 3,
        ) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels, hidden_channels, kernel_size, padding=padding
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Predict the next flow state.

            Parameters:
                x (torch.Tensor): Batch ``(B, C, H, W)``.

            Returns:
                torch.Tensor: Predicted next state, same spatial shape.
            """
            return self.net(x)

except ImportError:  # pragma: no cover

    class FlowCNN:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "torch is required for FlowCNN; install with pip install -e '.[ml]'"
            )


def build_flow_cnn(cfg: DictConfig) -> FlowCNN:
    """
    Construct a CNN from Hydra ``model`` config.

    When ``model.condition_on_re`` is true, one extra input channel is allocated
    for the broadcast Reynolds map (see ``ffaoml.ml.conditioning``).

    Parameters:
        cfg (DictConfig): Composed config with ``model`` group.

    Returns:
        FlowCNN: Initialized module (not on device yet).
    """
    model = cfg.model
    in_channels = int(model.in_channels)
    if bool(model.get("condition_on_re", False)):
        in_channels += 1
    return FlowCNN(
        in_channels=in_channels,
        out_channels=int(model.out_channels),
        hidden_channels=int(model.hidden_channels),
        kernel_size=int(model.kernel_size),
    )
