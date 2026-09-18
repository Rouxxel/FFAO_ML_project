"""
#############################################################################
### Physics utilities package
###
### @file physics/__init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Dimensionless numbers, integral coefficients, and discrete field operators
(CFD-backend agnostic; PRD §8).
"""

# Project imports
from ffaoml.physics.coefficients import (
    dominant_frequency_from_signal,
    drag_coefficient,
    dynamic_pressure_scale,
    lift_coefficient,
    strouhal_number,
)
from ffaoml.physics.navier_stokes import divergence_2d
from ffaoml.physics.reynolds import reynolds_number

__all__ = [
    "dominant_frequency_from_signal",
    "drag_coefficient",
    "divergence_2d",
    "dynamic_pressure_scale",
    "lift_coefficient",
    "reynolds_number",
    "strouhal_number",
]
