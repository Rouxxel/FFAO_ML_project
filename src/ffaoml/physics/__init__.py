"""Dimensionless numbers and field operators (CFD-backend agnostic)."""

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
