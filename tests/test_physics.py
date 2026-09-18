"""
#############################################################################
### Physics utility tests
###
### @file test_physics.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Unit tests for Reynolds number, divergence, and force/shedding coefficients
(PRD §8).
"""

# Native imports
import math

# Third-party imports
import numpy as np
import pytest

# Project imports
from ffaoml.physics.coefficients import (
    dominant_frequency_from_signal,
    drag_coefficient,
    lift_coefficient,
    strouhal_number,
)
from ffaoml.physics.navier_stokes import divergence_2d
from ffaoml.physics.reynolds import reynolds_number

"""TESTS-----------------------------------------------------------"""


def test_reynolds_number_known_value() -> None:
    assert reynolds_number(1.0, 1.0, 0.01) == 100.0


def test_reynolds_number_rejects_non_positive_nu() -> None:
    with pytest.raises(ValueError):
        reynolds_number(1.0, 1.0, 0.0)


def test_divergence_uniform_flow_near_zero_interior() -> None:
    ny, nx = 32, 48
    u = np.ones((ny, nx))
    v = np.zeros((ny, nx))
    div = divergence_2d(u, v, dx=0.1, dy=0.1)
    interior = div[1:-1, 1:-1]
    assert np.all(np.isfinite(interior))
    assert np.max(np.abs(interior)) < 1e-12


def test_divergence_zero_field() -> None:
    ny, nx = 16, 16
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    div = divergence_2d(u, v, dx=1.0, dy=1.0)
    interior = div[1:-1, 1:-1]
    assert np.allclose(interior, 0.0)


def test_divergence_solid_mask_nan() -> None:
    ny, nx = 10, 10
    u = np.ones((ny, nx))
    v = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[4:6, 4:6] = True
    div = divergence_2d(u, v, dx=0.05, dy=0.05, solid_mask=mask)
    assert np.all(np.isnan(div[mask]))


def test_drag_and_lift_coefficients() -> None:
    # Fd / (0.5 * 1 * 1^2 * 1) = 1.0 when Fd = 0.5
    assert drag_coefficient(0.5, rho=1.0, u=1.0, d=1.0) == 1.0
    assert lift_coefficient(1.0, rho=1.0, u=2.0, d=1.0) == 0.5


def test_strouhal_from_synthetic_shedding() -> None:
    u, d = 1.0, 0.1
    f_true = 2.5
    dt = 0.01
    n = 2048
    t = np.arange(n) * dt
    lift = np.sin(2 * math.pi * f_true * t)
    f_est = dominant_frequency_from_signal(lift, dt, min_frequency=0.5)
    assert abs(f_est - f_true) < 0.05
    st = strouhal_number(f_est, d, u)
    assert abs(st - f_true * d / u) < 0.01
