"""Dedalus spectral CFD backend (deferred)."""

from __future__ import annotations

from ffaoml.cfd.types import SimulationCase, SimulationResult


class DedalusSolver:
    name = "dedalus"

    def run(self, case: SimulationCase) -> SimulationResult:
        raise NotImplementedError(
            "Dedalus backend is not implemented yet; install on Linux/WSL2 and "
            "implement dedalus_backend.py when enabling simulation.backend=dedalus."
        )
