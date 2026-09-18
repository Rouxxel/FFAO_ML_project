"""CFD solver adapters."""

from ffaoml.cfd.solver import case_from_config, get_solver, run_from_config
from ffaoml.cfd.types import SimulationCase, SimulationResult

__all__ = [
    "SimulationCase",
    "SimulationResult",
    "case_from_config",
    "get_solver",
    "run_from_config",
]
