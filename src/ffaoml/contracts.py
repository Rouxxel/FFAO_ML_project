"""Shared layout contracts between CFD export, datasets, and models.

Human-readable specification: documentation/CONTRACTS.md
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

# ML tensor layout: (batch, channels, height, width)
# Channel order is fixed unless a run config documents a subset.
DEFAULT_FIELD_CHANNELS: Final[tuple[str, ...]] = (
    "velocity_x",
    "velocity_y",
    "pressure",
    "vorticity",
)

# Short names used in some configs and plots (u_x, u_y, p, omega).
FIELD_CHANNEL_ALIASES: Final[dict[str, str]] = {
    "u_x": "velocity_x",
    "u_y": "velocity_y",
    "p": "pressure",
    "omega": "vorticity",
    "ω": "vorticity",
}


class FieldChannel(StrEnum):
    """Channel identifiers for stacked flow tensors."""

    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    PRESSURE = "pressure"
    VORTICITY = "vorticity"


# Cylinder mask semantics (boolean grid, same H x W as fields):
# - True  -> solid / interior of cylinder; velocity is zero; exclude from loss
#          unless a task explicitly models the body.
# - False -> fluid cell; participate in field losses and divergence penalties.
#
# Grid metadata (spacing dx, dy, origin) lives in simulation metadata, not in
# the mask tensor.

METADATA_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "sim_id",
    "re",
    "split",
    "path",
    "u_inlet",
    "nu",
    "diameter",
    "nx",
    "ny",
    "dt",
    "n_steps",
    "seed",
)


def channel_index(name: str, channels: Sequence[str] | None = None) -> int:
    """Return the channel index for ``name`` (full name or alias)."""
    order = tuple(channels) if channels is not None else DEFAULT_FIELD_CHANNELS
    resolved = FIELD_CHANNEL_ALIASES.get(name, name)
    try:
        return order.index(resolved)
    except ValueError as exc:
        raise KeyError(f"Unknown field channel: {name}") from exc
