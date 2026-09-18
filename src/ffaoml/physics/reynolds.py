"""
#############################################################################
### Reynolds number
###
### @file reynolds.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Kinematic Reynolds number and related scalings for cylinder flow (PRD §4, §8).
"""

# Native imports
from __future__ import annotations

"""SCALINGS-----------------------------------------------------------"""


def reynolds_number(u: float, d: float, nu: float) -> float:
    """
    Kinematic Reynolds number Re = U D / ν.

    Parameters:
        u (float): Characteristic flow speed (e.g. inlet velocity).
        d (float): Characteristic length (e.g. cylinder diameter).
        nu (float): Kinematic viscosity.

    Returns:
        float: Reynolds number.

    Raises:
        ValueError: If viscosity is non-positive.
    """
    if nu <= 0:
        raise ValueError("kinematic viscosity nu must be positive")
    return u * d / nu
