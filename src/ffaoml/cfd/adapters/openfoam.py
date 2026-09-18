"""
#############################################################################
### OpenFOAM adapter (placeholder)
###
### @file openfoam.py
### @author Sebastian Russo
### @date 2026
#############################################################################

External OpenFOAM case runner (WSL2/Docker). Implement when wiring
``simulation.backend=openfoam``.
"""

# Native imports
from __future__ import annotations

# Project imports
from ffaoml.cfd.types import SimulationCase, SimulationResult

"""BACKEND-----------------------------------------------------------"""


class OpenFoamSolver:
    """OpenFOAM case adapter (deferred)."""

    name = "openfoam"

    def run(self, case: SimulationCase) -> SimulationResult:
        """
        Run an OpenFOAM case and convert results to ``fields.zarr``.

        Parameters:
            case (SimulationCase): Case inputs and output paths.

        Returns:
            SimulationResult: Written fields and metadata.

        Raises:
            NotImplementedError: Until the adapter is implemented.
        """
        raise NotImplementedError(
            "OpenFOAM adapter is not implemented yet; use WSL2/Docker and "
            "adapters/openfoam.py when adding external CFD."
        )
