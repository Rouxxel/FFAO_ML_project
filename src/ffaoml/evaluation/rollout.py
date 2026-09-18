"""
#############################################################################
### Multi-step rollout evaluation
###
### @file rollout.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Autoregressive rollout of baseline predictors over normalized field trajectories.
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass

# Third-party imports
import numpy as np

# Project imports
from ffaoml.evaluation.baselines import BaselineName, predict_next
from ffaoml.evaluation.metrics import mse, relative_l2

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class RolloutCurve:
    """Per-horizon errors aggregated over rollout start indices."""

    horizons: tuple[int, ...]
    mse: tuple[float, ...]
    relative_l2: tuple[float, ...]
    n_starts: int


"""ONE-STEP-----------------------------------------------------------"""


def one_step_baseline_metrics(
    series: np.ndarray,
    baseline: BaselineName,
) -> dict[str, float]:
    """
    One-step errors on consecutive pairs in ``series``.

    Parameters:
        series (np.ndarray): ``(T, C, H, W)`` normalized trajectory.
        baseline (BaselineName): Baseline identifier.

    Returns:
        dict[str, float]: ``mse`` and ``relative_l2``.
    """
    if series.shape[0] < 2:
        raise ValueError("series needs at least 2 time steps")
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for t in range(1, series.shape[0]):
        current = series[t - 1]
        previous = series[t - 2] if t >= 2 else None
        if baseline == "linear" and previous is None:
            continue
        pred = predict_next(baseline, current, previous=previous)
        preds.append(pred)
        truths.append(series[t])
    if not preds:
        raise ValueError("not enough frames for linear baseline one-step metrics")
    pred_stack = np.stack(preds, axis=0)
    true_stack = np.stack(truths, axis=0)
    return {
        "mse": mse(pred_stack, true_stack),
        "relative_l2": relative_l2(pred_stack, true_stack),
    }


"""ROLLOUT-----------------------------------------------------------"""


def rollout_curve(
    series: np.ndarray,
    baseline: BaselineName,
    horizon: int,
) -> RolloutCurve:
    """
    Roll out a baseline for ``horizon`` steps from every valid start index.

    Parameters:
        series (np.ndarray): ``(T, C, H, W)`` trajectory.
        baseline (BaselineName): ``persistence`` or ``linear``.
        horizon (int): Maximum rollout length (capped by available time).

    Returns:
        RolloutCurve: Mean MSE and relative L² at each step 1..horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    t_total = series.shape[0]
    max_horizon = min(horizon, t_total - 1)
    if baseline == "linear" and t_total < 3:
        max_horizon = 0
    if max_horizon < 1:
        raise ValueError("series too short for requested rollout")

    mse_sums = np.zeros(max_horizon, dtype=np.float64)
    rel_sums = np.zeros(max_horizon, dtype=np.float64)
    n_starts = 0
    min_start = 1 if baseline == "linear" else 0
    for start in range(min_start, t_total - max_horizon):
        n_starts += 1
        if baseline == "persistence":
            x_hat = series[start].copy()
            for step in range(max_horizon):
                truth = series[start + step + 1]
                mse_sums[step] += mse(x_hat, truth)
                rel_sums[step] += relative_l2(x_hat, truth)
        else:
            x_prev = series[start - 1].copy()
            x_hat = series[start].copy()
            for step in range(max_horizon):
                truth = series[start + step + 1]
                mse_sums[step] += mse(x_hat, truth)
                rel_sums[step] += relative_l2(x_hat, truth)
                x_next = predict_next("linear", x_hat, previous=x_prev)
                x_prev, x_hat = x_hat, x_next

    if n_starts == 0:
        raise ValueError("no valid rollout start indices")

    horizons = tuple(range(1, max_horizon + 1))
    return RolloutCurve(
        horizons=horizons,
        mse=tuple(float(x / n_starts) for x in mse_sums),
        relative_l2=tuple(float(x / n_starts) for x in rel_sums),
        n_starts=n_starts,
    )


def sample_at_report_horizons(
    curve: RolloutCurve,
    report_horizons: list[int] | tuple[int, ...],
) -> dict[str, dict[int, float]]:
    """
    Pick metrics at selected horizon indices from a rollout curve.

    Parameters:
        curve (RolloutCurve): Full per-step curve.
        report_horizons (list[int] | tuple[int, ...]): Horizons to report.

    Returns:
        dict[str, dict[int, float]]: ``mse`` and ``relative_l2`` maps.
    """
    index = {h: h - 1 for h in curve.horizons}
    mse_map: dict[int, float] = {}
    rel_map: dict[int, float] = {}
    for h in report_horizons:
        if h not in index:
            continue
        idx = index[h]
        mse_map[h] = curve.mse[idx]
        rel_map[h] = curve.relative_l2[idx]
    return {"mse": mse_map, "relative_l2": rel_map}
