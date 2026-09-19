"""
#############################################################################
### Multi-step flow ConvLSTM
###
### @file convlstm.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Convolutional LSTM for temporal field evolution (Task C). Supports optional
truncated training unroll via ``forward_unroll``.
"""

# Native imports
from __future__ import annotations

from typing import Any

# Third-party imports
from omegaconf import DictConfig

try:
    import torch
    from torch import nn

    class ConvLSTMCell(nn.Module):
        """
        Single ConvLSTM cell on a structured grid.

        Hidden and cell states share spatial resolution with the input field.
        """

        def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            kernel_size: int,
        ) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.hidden_channels = hidden_channels
            self.conv = nn.Conv2d(
                input_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size,
                padding=padding,
            )

        def forward(
            self,
            x: torch.Tensor,
            h: torch.Tensor,
            c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Advance the cell by one time step.

            Parameters:
                x (torch.Tensor): Input ``(B, C_in, H, W)``.
                h (torch.Tensor): Hidden state ``(B, C_h, H, W)``.
                c (torch.Tensor): Cell state ``(B, C_h, H, W)``.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: Updated ``(h, c)``.
            """
            gates = self.conv(torch.cat([x, h], dim=1))
            i, f, o, g = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            return h_next, c_next

    class FlowConvLSTM(nn.Module):
        """
        ConvLSTM predictor with one-step and multi-step unroll APIs.

        One-step ``forward`` returns ``(prediction, (h, c))``. Training can use
        ``forward_unroll`` to average loss over several autoregressive steps.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int = 64,
            kernel_size: int = 3,
            input_steps: int = 1,
        ) -> None:
            super().__init__()
            if input_steps < 1:
                raise ValueError("input_steps must be >= 1")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.hidden_channels = hidden_channels
            self.input_steps = input_steps
            self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
            self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        def init_state(
            self,
            batch_size: int,
            height: int,
            width: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Zero-initialize hidden and cell states.

            Parameters:
                batch_size (int): Batch dimension.
                height (int): Grid height.
                width (int): Grid width.
                device (torch.device): Target device.
                dtype (torch.dtype): Tensor dtype.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: ``(h0, c0)``.
            """
            shape = (batch_size, self.hidden_channels, height, width)
            z = torch.zeros(shape, device=device, dtype=dtype)
            return z, z.clone()

        def forward(
            self,
            x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """
            Predict the next flow state from the current frame.

            Parameters:
                x (torch.Tensor): ``(B, C, H, W)``.
                state (tuple | None): Optional ``(h, c)``; zeros if omitted.

            Returns:
                tuple[torch.Tensor, tuple]: Prediction and new ``(h, c)``.
            """
            if state is None:
                h, c = self.init_state(
                    x.size(0),
                    x.size(2),
                    x.size(3),
                    x.device,
                    x.dtype,
                )
            else:
                h, c = state
            h, c = self.cell(x, h, c)
            return self.readout(h), (h, c)

        def forward_unroll(
            self,
            x0: torch.Tensor,
            steps: int,
            *,
            teacher_forcing: bool = False,
            target_seq: torch.Tensor | None = None,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """
            Autoregressively predict ``steps`` frames from ``x0``.

            Parameters:
                x0 (torch.Tensor): Starting frame ``(B, C, H, W)``.
                steps (int): Number of steps to predict.
                teacher_forcing (bool): Feed ground-truth frames as inputs when
                    ``target_seq`` is provided.
                target_seq (torch.Tensor | None): ``(B, steps, C, H, W)`` truths.
                state (tuple | None): Initial LSTM state.

            Returns:
                torch.Tensor: Predictions ``(B, steps, C, H, W)``.
            """
            if steps < 1:
                raise ValueError("steps must be >= 1")
            if teacher_forcing and (target_seq is None or target_seq.size(1) < steps):
                raise ValueError(
                    "teacher_forcing requires target_seq with length steps"
                )

            preds: list[torch.Tensor] = []
            x = x0
            for step in range(steps):
                y, state = self.forward(x, state)
                preds.append(y)
                if teacher_forcing and target_seq is not None:
                    x = target_seq[:, step]
                else:
                    x = y
            return torch.stack(preds, dim=1)

        def warm_state(
            self,
            seq: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """
            Advance hidden state over ``input_steps`` frames (no readout).

            Parameters:
                seq (torch.Tensor): ``(B, T_in, C, H, W)``.
                state (tuple | None): Optional initial state.

            Returns:
                tuple: Last input frame and updated ``(h, c)``.
            """
            if seq.size(1) != self.input_steps:
                raise ValueError(
                    f"expected {self.input_steps} input frames, got {seq.size(1)}"
                )
            state_out = state
            for t in range(seq.size(1)):
                _, state_out = self.forward(seq[:, t], state_out)
            return seq[:, -1], state_out

except ImportError:  # pragma: no cover

    class FlowConvLSTM:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "torch is required for FlowConvLSTM; "
                "install with pip install -e '.[ml]'"
            )


def build_flow_convlstm(cfg: DictConfig) -> FlowConvLSTM:
    """
    Construct a ConvLSTM from Hydra ``model`` config.

    Parameters:
        cfg (DictConfig): Composed config with ``model`` group.

    Returns:
        FlowConvLSTM: Initialized module (not on device yet).
    """
    model = cfg.model
    return FlowConvLSTM(
        in_channels=int(model.in_channels),
        out_channels=int(model.out_channels),
        hidden_channels=int(model.hidden_channels),
        kernel_size=int(model.kernel_size),
        input_steps=int(model.get("input_steps", 1)),
    )
