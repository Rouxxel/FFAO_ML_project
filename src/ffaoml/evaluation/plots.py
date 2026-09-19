"""
#############################################################################
### ML evaluation figures
###
### @file plots.py
### @author Sebastian Russo
### @date 2026
#############################################################################

PRD §13 figures: predicted vs true vorticity and error vs prediction horizon.
"""

# Native imports
from __future__ import annotations

from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from ffaoml.evaluation.rollout import RolloutCurve

"""VORTICITY-----------------------------------------------------------"""


def plot_vorticity_panels(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    output_path: str | Path,
    time_labels: list[str] | None = None,
    max_panels: int = 4,
) -> Path:
    """
    Side-by-side true vs predicted vorticity (PRD §13 item 2 analogue for ML).

    Parameters:
        truth (np.ndarray): ``(n, H, W)`` true ω slices.
        pred (np.ndarray): ``(n, H, W)`` predicted ω slices.
        output_path (str | Path): PNG destination.
        time_labels (list[str] | None): Optional panel titles.
        max_panels (int): Number of time slices to show.

    Returns:
        Path: Written figure path.
    """
    n = min(truth.shape[0], max_panels)
    if n < 1:
        raise ValueError("need at least one frame to plot")
    indices = np.linspace(0, truth.shape[0] - 1, n, dtype=int)
    vmax = np.percentile(np.abs(truth[indices]), 99)

    fig, axes = plt.subplots(n, 2, figsize=(6, 2.6 * n), squeeze=False)
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(
            truth[idx],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 0].set_title(time_labels[row] if time_labels else f"t={idx}")
        axes[row, 0].set_ylabel("true ω")
        axes[row, 1].imshow(
            pred[idx],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 1].set_title("predicted ω")
        for col in range(2):
            axes[row, col].set_xticks([])
    fig.suptitle("Vorticity: ground truth vs one-step CNN")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


"""HORIZON-----------------------------------------------------------"""


def plot_error_vs_horizon(
    curves: dict[str, RolloutCurve],
    *,
    output_path: str | Path,
    metric: str = "mse",
) -> Path:
    """
    Plot rollout error vs horizon for one or more predictors (PRD §13 item 10).

    Parameters:
        curves (dict[str, RolloutCurve]): Named rollout curves.
        output_path (str | Path): PNG path.
        metric (str): ``mse`` or ``relative_l2``.

    Returns:
        Path: Written figure path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, curve in curves.items():
        values = curve.mse if metric == "mse" else curve.relative_l2
        ax.plot(curve.horizons, values, marker="o", label=label)
    ax.set_xlabel("Prediction horizon (steps)")
    ylabel = "MSE" if metric == "mse" else "Relative L²"
    ax.set_ylabel(ylabel)
    ax.set_title("Rollout error vs horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_rollout_stability(
    model_curve: RolloutCurve,
    baseline_curve: RolloutCurve,
    *,
    output_path: str | Path,
) -> Path:
    """
    Compare model vs persistence rollout MSE (stability/drift view).

    Parameters:
        model_curve (RolloutCurve): Learned predictor rollout.
        baseline_curve (RolloutCurve): Persistence rollout on the same split.
        output_path (str | Path): PNG destination.

    Returns:
        Path: Written figure path.
    """
    return plot_error_vs_horizon(
        {"model": model_curve, "persistence": baseline_curve},
        output_path=output_path,
        metric="mse",
    )


"""REYNOLDS-----------------------------------------------------------"""


def plot_re_generalization_heatmap(
    per_re: dict[str, Any],
    *,
    train_re: list[float],
    val_re: list[float],
    test_re: list[float],
    output_path: str | Path,
) -> Path:
    """
    Heatmap of one-step MSE vs Reynolds (PRD §13 item 12).

    Parameters:
        per_re (dict[str, Any]): Output of ``re_generalization`` per-Re metrics.
        train_re (list[float]): Training Reynolds list from config.
        val_re (list[float]): Validation (interpolation) Reynolds.
        test_re (list[float]): Test (extrapolation) Reynolds.
        output_path (str | Path): PNG destination.

    Returns:
        Path: Written figure path.
    """
    entries = sorted(per_re.values(), key=lambda item: float(item["re"]))
    if not entries:
        raise ValueError("per_re is empty")
    res = [float(e["re"]) for e in entries]
    errors = [float(e["metrics"]["mse"]) for e in entries]
    regimes = [str(e["regime"]) for e in entries]

    fig, ax = plt.subplots(figsize=(max(6, len(res) * 0.6), 3.5))
    data = np.array(errors, dtype=float)[None, :]
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([str(int(r)) if r == int(r) else str(r) for r in res])
    ax.set_yticks([0])
    ax.set_yticklabels(["one-step MSE"])
    for idx, (err, regime) in enumerate(zip(errors, regimes, strict=True)):
        ax.text(
            idx, 0, f"{err:.3g}\n({regime})", ha="center", va="center", color="white"
        )
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    ax.set_title(
        f"Generalization vs Re (train={train_re}, val={val_re}, test={test_re})"
    )
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
