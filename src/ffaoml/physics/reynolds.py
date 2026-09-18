"""Reynolds number and related scalings (PRD §4, §8)."""

from __future__ import annotations


def reynolds_number(u: float, d: float, nu: float) -> float:
    """Kinematic Reynolds number Re = U D / ν.

    Args:
        u: Characteristic flow speed (e.g. inlet velocity).
        d: Characteristic length (e.g. cylinder diameter).
        nu: Kinematic viscosity.

    Raises:
        ValueError: If viscosity is non-positive.
    """
    if nu <= 0:
        raise ValueError("kinematic viscosity nu must be positive")
    return u * d / nu
