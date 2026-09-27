"""
#############################################################################
### MeshGraphNet evaluation report (Stage 2)
###
### @file evaluate_mesh_model.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Load a mesh training bundle and write rollout metrics + horizon figures.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Project imports
from ffaoml.data.catalog import list_mesh_trajectories_for_split
from ffaoml.data.mesh_io import edges_from_cells, load_mesh_trajectory_dir
from ffaoml.evaluation.mesh_rollout import (
    aggregate_mesh_rollout_curves,
    mesh_model_rollout_curve,
    mesh_persistence_rollout_curve,
    mesh_state_series,
    mesh_torch_predict_fn,
)
from ffaoml.evaluation.model_report import load_training_run_config
from ffaoml.evaluation.plots import plot_error_vs_horizon, plot_rollout_stability
from ffaoml.evaluation.rollout import sample_at_report_horizons
from ffaoml.ml.mesh_preprocessing import load_mesh_preprocess_stats
from ffaoml.models.meshgraphnet import build_meshgraphnet
from ffaoml.training.checkpointing import load_model_weights
from ffaoml.training.train_meshgraphnet import MODEL_FILENAME

"""CONSTANTS-----------------------------------------------------------"""
FIGURES_DIRNAME = "figures"
MODEL_EVAL_FILENAME = "model_eval_metrics.json"
STAGE1_COMPARISON_NOTE = (
    "Stage 1 grid CNN/FNO metrics use a different discretization; compare "
    "qualitatively only (no interpolation-based numeric winner)."
)

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class MeshModelEvaluationResult:
    """Output paths from ``run_mesh_model_evaluation``."""

    run_dir: Path
    figures_dir: Path
    metrics_path: Path
    horizon_figure: Path
    stability_figure: Path


"""LOADERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def load_trained_meshgraphnet(
    cfg: DictConfig, run_dir: str | Path, device: torch.device
):
    """Restore MeshGraphNet weights from ``model.pt``."""
    _require_torch()
    model = build_meshgraphnet(cfg).to(device)
    ckpt_path = Path(run_dir) / MODEL_FILENAME
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_model_weights(model, payload)
    model.eval()
    return model


"""RUNNER-----------------------------------------------------------"""


def run_mesh_model_evaluation(
    run_dir: str | Path,
    *,
    split: str | None = None,
    repo_root: str | Path | None = None,
) -> MeshModelEvaluationResult:
    """
    Generate rollout figures and ``model_eval_metrics.json`` for a mesh run.

    Parameters:
        run_dir (str | Path): Training output with ``model.pt`` and ``config.yaml``.
        split (str | None): Eval split; defaults to ``cfg.eval.split``.
        repo_root (str | Path | None): Reserved for provenance hooks.

    Returns:
        MeshModelEvaluationResult: Figure and JSON paths.
    """
    _require_torch()
    run_path = Path(run_dir)
    cfg = load_training_run_config(run_path)
    eval_split = split or str(cfg.eval.split)
    device = torch.device(str(cfg.train.device))
    horizon = int(cfg.eval.rollout_horizon)
    report_horizons = list(cfg.eval.report_horizons)

    stats = load_mesh_preprocess_stats(run_path / "preprocess_stats.json")
    model = load_trained_meshgraphnet(cfg, run_path, device)

    model_curves: list = []
    persist_curves: list = []
    trajectory_ids: list[str] = []

    for sim_id, traj_dir in list_mesh_trajectories_for_split(cfg, eval_split):
        traj = load_mesh_trajectory_dir(traj_dir)
        series = mesh_state_series(traj, stats)
        edge_index = edges_from_cells(traj.cells)
        predict_fn = mesh_torch_predict_fn(
            model,
            traj.mesh_pos,
            edge_index,
            traj.node_type,
            device,
        )
        model_curves.append(mesh_model_rollout_curve(series, predict_fn, horizon))
        persist_curves.append(mesh_persistence_rollout_curve(series, horizon))
        trajectory_ids.append(sim_id)

    if not model_curves:
        raise ValueError(
            f"no mesh trajectories for split {eval_split!r}; import Stage 2 data first"
        )

    model_curve = aggregate_mesh_rollout_curves(model_curves)
    persist_curve = aggregate_mesh_rollout_curves(persist_curves)

    figures_dir = run_path / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    horizon_path = plot_error_vs_horizon(
        {"meshgraphnet": model_curve, "persistence": persist_curve},
        output_path=figures_dir / "error_vs_horizon.png",
    )
    stability_path = plot_rollout_stability(
        model_curve,
        persist_curve,
        output_path=figures_dir / "rollout_stability.png",
    )

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "kind": "mesh_model_evaluation",
        "split": eval_split,
        "run_dir": run_path.as_posix(),
        "n_trajectories": len(trajectory_ids),
        "trajectory_ids": trajectory_ids,
        "rollout_horizon": horizon,
        "report_horizons": report_horizons,
        "rollout": {
            "meshgraphnet": {
                "horizons": list(model_curve.horizons),
                "mse": list(model_curve.mse),
                "relative_l2": list(model_curve.relative_l2),
                "at_report_horizons": sample_at_report_horizons(
                    model_curve, report_horizons
                ),
            },
            "persistence": {
                "horizons": list(persist_curve.horizons),
                "mse": list(persist_curve.mse),
                "relative_l2": list(persist_curve.relative_l2),
                "at_report_horizons": sample_at_report_horizons(
                    persist_curve, report_horizons
                ),
            },
        },
        "note_stage1_comparison": STAGE1_COMPARISON_NOTE,
    }
    metrics_path = run_path / MODEL_EVAL_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return MeshModelEvaluationResult(
        run_dir=run_path,
        figures_dir=figures_dir,
        metrics_path=metrics_path,
        horizon_figure=horizon_path,
        stability_figure=stability_path,
    )
