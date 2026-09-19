"""
#############################################################################
### Trained-model evaluation report
###
### @file model_report.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Load a CNN checkpoint bundle and write Stage 1 ML evaluation figures.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig, OmegaConf

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Project imports
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.loading import load_split_tensor
from ffaoml.evaluation.model_rollout import (
    model_rollout_curve,
    one_step_model_predictions,
    torch_predict_fn,
)
from ffaoml.evaluation.plots import (
    plot_error_vs_horizon,
    plot_rollout_stability,
    plot_vorticity_panels,
)
from ffaoml.evaluation.rollout import rollout_curve, sample_at_report_horizons
from ffaoml.ml.preprocessing import load_preprocess_stats, normalize_fields
from ffaoml.models.factory import build_flow_model
from ffaoml.training.checkpointing import load_model_weights
from ffaoml.training.train import MODEL_FILENAME

"""CONSTANTS-----------------------------------------------------------"""
FIGURES_DIRNAME = "figures"
MODEL_EVAL_FILENAME = "model_eval_metrics.json"
VORTICITY_CHANNEL = DEFAULT_FIELD_CHANNELS.index("vorticity")

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Output paths from ``run_model_evaluation``."""

    run_dir: Path
    figures_dir: Path
    metrics_path: Path
    vorticity_figure: Path
    horizon_figure: Path
    stability_figure: Path


"""LOADERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def load_training_run_config(run_dir: str | Path) -> DictConfig:
    """
    Load resolved ``config.yaml`` from a training run directory.

    Parameters:
        run_dir (str | Path): Checkpoint bundle path.

    Returns:
        DictConfig: Composed training configuration.
    """
    path = Path(run_dir) / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config.yaml in {run_dir}")
    return OmegaConf.create(OmegaConf.load(path))


def load_trained_cnn(cfg: DictConfig, run_dir: str | Path, device: torch.device):
    """
    Restore ``FlowCNN`` weights from ``model.pt``.

    Parameters:
        cfg (DictConfig): Model hyperparameters.
        run_dir (str | Path): Run directory.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Eval-ready model.
    """
    _require_torch()
    model = build_flow_model(cfg).to(device)
    ckpt_path = Path(run_dir) / MODEL_FILENAME
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_model_weights(model, payload)
    model.eval()
    return model


"""RUNNER-----------------------------------------------------------"""


def run_model_evaluation(
    run_dir: str | Path,
    *,
    split: str | None = None,
    repo_root: str | Path | None = None,
) -> ModelEvaluationResult:
    """
    Generate ML Phase 3 figures for a trained CNN run.

    Parameters:
        run_dir (str | Path): Training output with ``model.pt`` and ``config.yaml``.
        split (str | None): Eval split; defaults to ``cfg.eval.split``.
        repo_root (str | Path | None): Unused placeholder for future provenance.

    Returns:
        ModelEvaluationResult: Figure and JSON paths.
    """
    _require_torch()
    run_path = Path(run_dir)
    cfg = load_training_run_config(run_path)
    eval_split = split or str(cfg.eval.split)
    device = torch.device(str(cfg.train.device))

    stats = load_preprocess_stats(run_path / "preprocess_stats.json")
    raw = load_split_tensor(cfg, eval_split)
    series = normalize_fields(raw, stats)

    model = load_trained_cnn(cfg, run_path, device)
    predict_fn = torch_predict_fn(model, device)

    preds, truths = one_step_model_predictions(series, predict_fn)
    omega_true = truths[:, VORTICITY_CHANNEL]
    omega_pred = preds[:, VORTICITY_CHANNEL]

    figures_dir = run_path / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    vort_path = plot_vorticity_panels(
        omega_true,
        omega_pred,
        output_path=figures_dir / "vorticity_pred_vs_true.png",
    )

    horizon = int(cfg.eval.rollout_horizon)
    model_curve = model_rollout_curve(series, predict_fn, horizon)
    persist_curve = rollout_curve(series, "persistence", horizon)
    horizon_path = plot_error_vs_horizon(
        {"cnn": model_curve, "persistence": persist_curve},
        output_path=figures_dir / "error_vs_horizon.png",
    )
    stability_path = plot_rollout_stability(
        model_curve,
        persist_curve,
        output_path=figures_dir / "rollout_stability.png",
    )

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "kind": "model_evaluation",
        "split": eval_split,
        "run_dir": run_path.as_posix(),
        "vorticity_channel": VORTICITY_CHANNEL,
        "rollout": {
            "cnn": {
                "horizons": list(model_curve.horizons),
                "mse": list(model_curve.mse),
                "relative_l2": list(model_curve.relative_l2),
                "at_report_horizons": sample_at_report_horizons(
                    model_curve, list(cfg.eval.report_horizons)
                ),
            },
            "persistence": {
                "horizons": list(persist_curve.horizons),
                "mse": list(persist_curve.mse),
                "relative_l2": list(persist_curve.relative_l2),
            },
        },
        "note_re_heatmaps": "Deferred until Stage 3 (ML_EVAL Phase 3).",
    }
    metrics_path = run_path / MODEL_EVAL_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return ModelEvaluationResult(
        run_dir=run_path,
        figures_dir=figures_dir,
        metrics_path=metrics_path,
        vorticity_figure=vort_path,
        horizon_figure=horizon_path,
        stability_figure=stability_path,
    )
