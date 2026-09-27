"""
#############################################################################
### Mesh autoregressive rollout (Stage 2)
###
### @file mesh_rollout.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Multi-step rollout on normalized mesh node trajectories ``(T, N, C)``.
"""

# Native imports
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np

# Project imports
from ffaoml.data.mesh_io import MeshTrajectory
from ffaoml.evaluation.metrics import mse, relative_l2
from ffaoml.evaluation.rollout import RolloutCurve
from ffaoml.ml.mesh_preprocessing import MeshPreprocessStats, normalize_mesh_state

if TYPE_CHECKING:
    import torch

"""SERIES-----------------------------------------------------------"""


def mesh_state_series(
    traj: MeshTrajectory,
    stats: MeshPreprocessStats,
) -> np.ndarray:
    """
    Build a normalized node-feature time series.

    Parameters:
        traj (MeshTrajectory): Loaded mesh trajectory.
        stats (MeshPreprocessStats): Train-fitted normalization.

    Returns:
        np.ndarray: ``(T, N, 3)`` float32.
    """
    frames: list[np.ndarray] = []
    for t in range(traj.n_steps):
        frames.append(normalize_mesh_state(traj.velocity[t], traj.pressure[t], stats))
    return np.stack(frames, axis=0)


"""ROLLOUT-----------------------------------------------------------"""


def mesh_persistence_rollout_curve(
    series: np.ndarray,
    horizon: int,
) -> RolloutCurve:
    """
    Persistence rollout: hold the start state fixed vs future truths.

    Parameters:
        series (np.ndarray): ``(T, N, C)`` normalized trajectory.
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

    horizons = tuple(range(1, max_horizon + 1))
    return RolloutCurve(
        horizons=horizons,
        mse=tuple(float(x / n_starts) for x in mse_sums),
        relative_l2=tuple(float(x / n_starts) for x in rel_sums),
        n_starts=n_starts,
    )


def mesh_model_rollout_curve(
    series: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    horizon: int,
) -> RolloutCurve:
    """
    Autoregressive rollout with a one-step mesh predictor.

    Parameters:
        series (np.ndarray): ``(T, N, C)`` normalized trajectory.
        predict_fn (Callable): Maps ``(N, C)`` → next ``(N, C)``.
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


def aggregate_mesh_rollout_curves(curves: list[RolloutCurve]) -> RolloutCurve:
    """Average per-horizon metrics across trajectories."""
    if not curves:
        raise ValueError("no rollout curves to aggregate")
    min_len = min(len(c.horizons) for c in curves)
    horizons = tuple(range(1, min_len + 1))
    mse_vals: list[float] = []
    rel_vals: list[float] = []
    for idx in range(min_len):
        mse_vals.append(float(np.mean([c.mse[idx] for c in curves])))
        rel_vals.append(float(np.mean([c.relative_l2[idx] for c in curves])))
    n_starts = int(np.mean([c.n_starts for c in curves]))
    return RolloutCurve(
        horizons=horizons,
        mse=tuple(mse_vals),
        relative_l2=tuple(rel_vals),
        n_starts=n_starts,
    )


def mesh_torch_predict_fn(
    model: torch.nn.Module,
    mesh_pos: np.ndarray,
    edge_index: np.ndarray,
    node_type: np.ndarray,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Numpy predict_fn for MeshGraphNet one-step inference.

    Parameters:
        model (torch.nn.Module): Trained ``MeshGraphNet``.
        mesh_pos (np.ndarray): ``(N, 2)`` positions.
        edge_index (np.ndarray): ``(2, E)`` COO edges.
        node_type (np.ndarray): ``(N,)`` int node types.
        device (torch.device): Inference device.

    Returns:
        Callable: ``(N, C)`` → ``(N, C)`` in float32.
    """
    import torch

    pos_t = torch.from_numpy(mesh_pos.astype(np.float32)).to(device)
    edge_t = torch.from_numpy(edge_index.astype(np.int64)).to(device)
    type_t = torch.from_numpy(node_type.astype(np.int64)).to(device)

    def _predict(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(state.astype(np.float32)).to(device)
            out = model(x, pos_t, edge_t, type_t).cpu().numpy()
        return out.astype(np.float32)

    return _predict
