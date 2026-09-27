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

# Stage 3 dataset manifest ``source_id`` for CFDBench cylinder subset.
STAGE3_CFDBENCH_SOURCE_ID: Final[str] = "cfdbench_cylinder"

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

# Optional mesh-trajectory columns (Stage 2); appended in metadata.csv when present.
MESH_METADATA_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "n_nodes",
    "n_cells",
)

# Static mesh arrays stored per trajectory (Zarr group or equivalent).
MESH_STATIC_ARRAYS: Final[tuple[str, ...]] = (
    "mesh_pos",
    "node_type",
    "cells",
)

# Time-varying node fields (T, N_nodes, …) for MeshGraphNets-style imports.
MESH_DYNAMIC_NODE_FIELDS: Final[tuple[str, ...]] = (
    "velocity",
    "pressure",
)

# MeshGraphNets node_type integers (documentation/CONTRACTS.md).
MESH_NODE_TYPE_NORMAL: Final[int] = 0
MESH_NODE_TYPE_OBSTACLE: Final[int] = 1
MESH_NODE_TYPE_AIRFOIL: Final[int] = 2
MESH_NODE_TYPE_HANDLE: Final[int] = 3
MESH_NODE_TYPE_INFLOW: Final[int] = 4
MESH_NODE_TYPE_OUTFLOW: Final[int] = 5
MESH_NODE_TYPE_WALL: Final[int] = 6

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
