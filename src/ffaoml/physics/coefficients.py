"""
#############################################################################
### Force coefficients and Strouhal number
###
### @file coefficients.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Integral force coefficients, Strouhal number, and spectral helpers for
vortex-shedding analysis (PRD §8).
"""

# Native imports
from __future__ import annotations

# Third-party imports
import numpy as np
from scipy.fft import rfft, rfftfreq

"""TYPES-----------------------------------------------------------"""
Array1D = np.ndarray

"""REFERENCE SCALES-----------------------------------------------------------"""


def dynamic_pressure_scale(rho: float, u: float, d: float) -> float:
    """
    Reference scale ½ ρ U² D used for Cd and Cl.

    Parameters:
        rho (float): Fluid density.
        u (float): Reference speed.
        d (float): Reference length (diameter).

    Returns:
        float: Force scale in consistent units.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if rho <= 0 or u <= 0 or d <= 0:
        raise ValueError("rho, u, and d must be positive")
    return 0.5 * rho * u**2 * d


def drag_coefficient(
    force_drag: float,
    rho: float,
    u: float,
    d: float,
) -> float:
    """
    Drag coefficient Cd = Fd / (½ ρ U² D).

    Parameters:
        force_drag (float): Drag force.
        rho (float): Fluid density.
        u (float): Reference speed.
        d (float): Cylinder diameter.

    Returns:
        float: Dimensionless Cd.
    """
    return force_drag / dynamic_pressure_scale(rho, u, d)


def lift_coefficient(
    force_lift: float,
    rho: float,
    u: float,
    d: float,
) -> float:
    """
    Lift coefficient Cl = Fl / (½ ρ U² D).

    Parameters:
        force_lift (float): Lift force.
        rho (float): Fluid density.
        u (float): Reference speed.
        d (float): Cylinder diameter.

    Returns:
        float: Dimensionless Cl.
    """
    return force_lift / dynamic_pressure_scale(rho, u, d)


def strouhal_number(frequency: float, d: float, u: float) -> float:
    """
    Strouhal number St = f D / U.

    Parameters:
        frequency (float): Shedding frequency (Hz).
        d (float): Cylinder diameter.
        u (float): Free-stream speed.

    Returns:
        float: Strouhal number.

    Raises:
        ValueError: If speed or diameter is non-positive.
    """
    if u <= 0 or d <= 0:
        raise ValueError("u and d must be positive")
    return frequency * d / u


"""SIGNAL ANALYSIS-----------------------------------------------------------"""


def dominant_frequency_from_signal(
    signal: Array1D,
    dt: float,
    *,
    min_frequency: float = 0.0,
) -> float:
    """
    Estimate dominant frequency (Hz) from a uniformly sampled time series.

    Uses the magnitude of the real FFT and ignores the DC component. Bins below
    ``min_frequency`` are excluded when that threshold is positive.

    Parameters:
        signal (Array1D): 1D samples vs time.
        dt (float): Sample interval (seconds).
        min_frequency (float): Minimum frequency (Hz) to consider for the peak.

    Returns:
        float: Frequency of the largest remaining spectral peak.

    Raises:
        ValueError: If ``dt`` is non-positive or the signal is too short.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1 or samples.size < 4:
        raise ValueError("signal must be a 1D array with at least 4 samples")

    centered = samples - np.mean(samples)
    spectrum = np.abs(rfft(centered))
    freqs = rfftfreq(centered.size, d=dt)

    if spectrum.size <= 1:
        raise ValueError("signal too short for frequency analysis")

    spectrum = spectrum[1:]
    freqs = freqs[1:]
    if min_frequency > 0:
        keep = freqs >= min_frequency
        spectrum = spectrum[keep]
        freqs = freqs[keep]

    if freqs.size == 0:
        raise ValueError("no frequency bins above min_frequency")

    peak_index = int(np.argmax(spectrum))
    return float(freqs[peak_index])
