"""
#############################################################################
### CFD simulation types
###
### @file types.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Immutable case description and mutable run result types shared by all CFD
backends (avoids circular imports through ``solver.py``).
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class SimulationCase:
    """
    Resolved inputs for a single simulation export.

    Paths follow ``<output_root>/simulations/<sim_id>/``.
    """

    sim_id: str
    output_root: Path
    re: float
    u_inlet: float
    nu: float
    diameter: float
    nx: int
    ny: int
    dt: float
    n_steps: int
    seed: int = 0
    domain_width: float = 10.0
    domain_height: float = 6.0
    split: str = "train"

    @property
    def simulation_dir(self) -> Path:
        """Absolute directory for this case's ``fields.zarr`` and sidecars."""
        return self.output_root / "simulations" / self.sim_id

    @property
    def relative_path(self) -> str:
        """Path relative to dataset root for ``metadata.csv``."""
        return f"simulations/{self.sim_id}"


@dataclass
class SimulationResult:
    """Outputs produced by ``Solver.run``."""

    case: SimulationCase
    store_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    force_drag: list[float] | None = None
    force_lift: list[float] | None = None
