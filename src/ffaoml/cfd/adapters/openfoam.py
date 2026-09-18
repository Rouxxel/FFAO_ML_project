"""OpenFOAM case adapter (deferred)."""

from __future__ import annotations

from ffaoml.cfd.types import SimulationCase, SimulationResult


class OpenFoamSolver:
    name = "openfoam"

    def run(self, case: SimulationCase) -> SimulationResult:
        raise NotImplementedError(
            "OpenFOAM adapter is not implemented yet; use WSL2/Docker and "
            "adapters/openfoam.py when adding external CFD."
        )
