"""Custom finite-difference Navier–Stokes backend (Phase B — not implemented)."""

from __future__ import annotations

from ffaoml.cfd.types import SimulationCase, SimulationResult


class FiniteDifferenceSolver:
    name = "fd"

    def run(self, case: SimulationCase) -> SimulationResult:
        raise NotImplementedError(
            "Finite-difference CFD backend is not implemented yet; use backend=stub "
            "for schema tests or complete Track B in the CFD plan."
        )
