"""
#############################################################################
### Finite-difference CFD backend (placeholder)
###
### @file fd_backend.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Custom in-repo Navier–Stokes solver (Track B). Not implemented yet; default
Hydra config may still name ``backend: fd`` for future work.
"""

# Native imports
from __future__ import annotations

# Project imports
from ffaoml.cfd.types import SimulationCase, SimulationResult

"""BACKEND-----------------------------------------------------------"""


class FiniteDifferenceSolver:
    """Structured-grid finite-difference flow solver (deferred)."""

    name = "fd"

    def run(self, case: SimulationCase) -> SimulationResult:
        """
        Run finite-difference CFD for one case.

        Parameters:
            case (SimulationCase): Case inputs and output paths.

        Returns:
            SimulationResult: Written fields and metadata.

        Raises:
            NotImplementedError: Until Track B is implemented.
        """
        raise NotImplementedError(
            "Finite-difference CFD backend is not implemented yet; use backend=stub "
            "for schema tests or complete Track B in the CFD plan."
        )
