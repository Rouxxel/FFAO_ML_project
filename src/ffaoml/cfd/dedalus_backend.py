"""
#############################################################################
### Dedalus spectral CFD backend (placeholder)
###
### @file dedalus_backend.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Spectral PDE backend for Linux/WSL2 environments. Install Dedalus separately
when enabling ``simulation.backend=dedalus``.
"""

# Native imports
from __future__ import annotations

# Project imports
from ffaoml.cfd.types import SimulationCase, SimulationResult

"""BACKEND-----------------------------------------------------------"""


class DedalusSolver:
    """Dedalus-based flow solver (deferred)."""

    name = "dedalus"

    def run(self, case: SimulationCase) -> SimulationResult:
        """
        Run Dedalus CFD for one case.

        Parameters:
            case (SimulationCase): Case inputs and output paths.

        Returns:
            SimulationResult: Written fields and metadata.

        Raises:
            NotImplementedError: Until Dedalus integration is added.
        """
        raise NotImplementedError(
            "Dedalus backend is not implemented yet; install on Linux/WSL2 and "
            "implement dedalus_backend.py when enabling simulation.backend=dedalus."
        )
