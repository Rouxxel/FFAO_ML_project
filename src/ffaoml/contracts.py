"""
#############################################################################
### Shared layout contracts
###
### @file contracts.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Tensor channel order, metadata columns, and naming aliases shared between
CFD export, Zarr datasets, and ML models.

Human-readable specification: ``documentation/CONTRACTS.md``.
"""

# Native imports
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

"""CONSTANTS-----------------------------------------------------------"""
# ML tensor layout: (batch, channels, height, width)
DEFAULT_FIELD_CHANNELS: Final[tuple[str, ...]] = (
    "velocity_x",
    "velocity_y",
    "pressure",
    "vorticity",
)

# Short names used in configs and plots (u_x, u_y, p, omega).
FIELD_CHANNEL_ALIASES: Final[dict[str, str]] = {
    "u_x": "velocity_x",
    "u_y": "velocity_y",
    "p": "pressure",
    "omega": "vorticity",
    "ω": "vorticity",
}

# Cylinder mask semantics (boolean grid, same H x W as fields):
# - True  -> solid / interior of cylinder; velocity is zero; exclude from loss
#          unless a task explicitly models the body.
# - False -> fluid cell; participate in field losses and divergence penalties.
# Grid spacing (dx, dy, origin) lives in simulation metadata, not in the mask.

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

"""TYPES-----------------------------------------------------------"""


class FieldChannel(StrEnum):
    """Identifiers for stacked flow-field tensors."""

    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    PRESSURE = "pressure"
    VORTICITY = "vorticity"


"""HELPERS-----------------------------------------------------------"""


def channel_index(name: str, channels: Sequence[str] | None = None) -> int:
    """
    Return the channel index for a full name or alias.

    Parameters:
        name (str): Channel name (e.g. ``velocity_x`` or alias ``u_x``).
        channels (Sequence[str] | None): Custom channel order; defaults to
            ``DEFAULT_FIELD_CHANNELS``.

    Returns:
        int: Zero-based index in the channel order.

    Raises:
        KeyError: If ``name`` does not resolve to a known channel.
    """
    order = tuple(channels) if channels is not None else DEFAULT_FIELD_CHANNELS
    resolved = FIELD_CHANNEL_ALIASES.get(name, name)
    try:
        return order.index(resolved)
    except ValueError as exc:
        raise KeyError(f"Unknown field channel: {name}") from exc
