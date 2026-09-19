"""
#############################################################################
### Learned-model rollout evaluation
###
### @file model_rollout.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Autoregressive rollout and one-step inference for trained PyTorch predictors.
"""

# Native imports
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np

# Project imports
from ffaoml.evaluation.metrics import mse, relative_l2
from ffaoml.evaluation.rollout import RolloutCurve

if TYPE_CHECKING:
    import torch

"""INFERENCE-----------------------------------------------------------"""


def one_step_model_predictions(
    series: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step predictions for consecutive frames in ``series``.

    Parameters:
        series (np.ndarray): ``(T, C, H, W)`` normalized trajectory.
        predict_fn (Callable): Maps ``(C, H, W)`` → next state.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(preds, truths)`` each ``(T-1, C, H, W)``.
    """
    if series.shape[0] < 2:
        raise ValueError("series needs at least 2 time steps")
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for t in range(series.shape[0] - 1):
        preds.append(predict_fn(series[t]))
        truths.append(series[t + 1])
    return np.stack(preds, axis=0), np.stack(truths, axis=0)


def model_rollout_curve(
    series: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    horizon: int,
) -> RolloutCurve:
    """
    Multi-step rollout using a learned one-step predictor.

    Parameters:
        series (np.ndarray): ``(T, C, H, W)`` trajectory.
        predict_fn (Callable): Maps current state to predicted next state.
        horizon (int): Maximum rollout length.

    Returns:
        RolloutCurve: Mean errors at horizons 1..H.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    t_total = series.shape[0]
    max_horizon = min(horizon, t_total - 1)
    if max_horizon < 1:
        raise ValueError("series too short for requested rollout")

    mse_sums = np.zeros(max_horizon, dtype=np.float64)
    rel_sums = np.zeros(max_horizon, dtype=np.float64)
    n_starts = 0
    for start in range(0, t_total - max_horizon):
        n_starts += 1
        x_hat = series[start].copy()
        for step in range(max_horizon):
            truth = series[start + step + 1]
            mse_sums[step] += mse(x_hat, truth)
            rel_sums[step] += relative_l2(x_hat, truth)
            x_hat = predict_fn(x_hat)

    horizons = tuple(range(1, max_horizon + 1))
    return RolloutCurve(
        horizons=horizons,
        mse=tuple(float(x / n_starts) for x in mse_sums),
        relative_l2=tuple(float(x / n_starts) for x in rel_sums),
        n_starts=n_starts,
    )


def torch_predict_fn(model: torch.nn.Module, device: torch.device) -> Callable:
    """
    Build a numpy ``predict_fn`` from a PyTorch module.

    Parameters:
        model (torch.nn.Module): One-step predictor.
        device (torch.device): Inference device.

    Returns:
        Callable: ``(C,H,W) -> (C,H,W)`` in float32.
    """
    import torch

    def _predict(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch = torch.from_numpy(state[None, ...]).float().to(device)
            out = model(batch).cpu().numpy()[0]
        return out.astype(np.float32)

    return _predict


def convlstm_rollout_curve(
    series: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    horizon: int,
) -> RolloutCurve:
    """
    Multi-step rollout for a ConvLSTM with carried hidden state.

    Parameters:
        series (np.ndarray): ``(T, C, H, W)`` trajectory.
        model (torch.nn.Module): ``FlowConvLSTM`` in eval mode.
        device (torch.device): Inference device.
        horizon (int): Maximum rollout length.

    Returns:
        RolloutCurve: Mean errors at horizons 1..H.
    """
    import torch

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    t_total = series.shape[0]
    max_horizon = min(horizon, t_total - 1)
    if max_horizon < 1:
        raise ValueError("series too short for requested rollout")

    mse_sums = np.zeros(max_horizon, dtype=np.float64)
    rel_sums = np.zeros(max_horizon, dtype=np.float64)
    n_starts = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, t_total - max_horizon):
            n_starts += 1
            state = None
            x = torch.from_numpy(series[start][None, ...]).float().to(device)
            for step in range(max_horizon):
                truth = series[start + step + 1]
                pred, state = model(x, state)
                x_hat = pred.cpu().numpy()[0]
                mse_sums[step] += mse(x_hat, truth)
                rel_sums[step] += relative_l2(x_hat, truth)
                x = pred

    horizons = tuple(range(1, max_horizon + 1))
    return RolloutCurve(
        horizons=horizons,
        mse=tuple(float(x / n_starts) for x in mse_sums),
        relative_l2=tuple(float(x / n_starts) for x in rel_sums),
        n_starts=n_starts,
    )
