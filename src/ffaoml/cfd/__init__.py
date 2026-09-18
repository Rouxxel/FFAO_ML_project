"""
#############################################################################
### CFD package exports
###
### @file cfd/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Public CFD helpers: case types, config-driven case build, and solver factory.
"""

# Project imports
from ffaoml.cfd.solver import case_from_config, get_solver, run_from_config
from ffaoml.cfd.types import SimulationCase, SimulationResult

__all__ = [
    "SimulationCase",
    "SimulationResult",
    "case_from_config",
    "get_solver",
    "run_from_config",
]
