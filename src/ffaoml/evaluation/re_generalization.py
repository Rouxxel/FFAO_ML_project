"""
#############################################################################
### Reynolds generalization evaluation (PRD §14 experiments 1–3)
###
### @file re_generalization.py
### @author Sebastian Russo
### @date 2026
#############################################################################

One-step field error per Reynolds number and split regime (train / val / test).
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Project imports
from ffaoml.data.catalog import dataset_root_from_config, lookup_metadata_row
from ffaoml.data.loading import load_simulation_tensor
from ffaoml.evaluation.metrics import mse, relative_l2
from ffaoml.evaluation.model_rollout import one_step_model_predictions, torch_predict_fn
from ffaoml.evaluation.plots import plot_re_generalization_heatmap
from ffaoml.ml.conditioning import (
    ReScaling,
    append_re_channel,
    condition_on_re_enabled,
)
from ffaoml.ml.preprocessing import load_preprocess_stats, normalize_fields
from ffaoml.ml.splits import re_split_simulation_ids
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.training.train import MODEL_FILENAME

"""CONSTANTS-----------------------------------------------------------"""
RE_EVAL_FILENAME = "re_generalization_metrics.json"
FIGURES_DIRNAME = "figures"

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class ReGeneralizationResult:
    """Artifacts from ``run_re_generalization_evaluation``."""

    run_dir: Path
    metrics_path: Path
    heatmap_path: Path


"""HELPERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def _re_regime(cfg: DictConfig, re: float) -> str:
    train = {float(r) for r in cfg.dataset.train_re}
    val = {float(r) for r in cfg.dataset.val_re}
    test = {float(r) for r in cfg.dataset.test_re}
    if re in train:
        return "train"
    if re in val:
        return "val"
    if re in test:
        return "test"
    return "unknown"


def _experiment_id(regime: str) -> str:
    if regime == "train":
        return "exp1_in_distribution"
    if regime == "val":
        return "exp2_interpolation"
    if regime == "test":
        return "exp3_extrapolation"
    return "unknown"


def evaluate_one_step_per_simulation(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    stats,
    sim_id: str,
    *,
    eval_split: str = "train",
) -> dict[str, float]:
    """
    One-step MSE / relative L² for all consecutive frames in a simulation.

    Parameters:
        cfg (DictConfig): Dataset config.
        model (torch.nn.Module): Trained predictor.
        device (torch.device): Inference device.
        stats: ``PreprocessStats`` for normalization and Re scaling.
        sim_id (str): Simulation identifier.
        eval_split (str): Temporal window name.

    Returns:
        dict[str, float]: ``mse`` and ``relative_l2``.
    """
    row = lookup_metadata_row(dataset_root_from_config(cfg), sim_id)
    re_val = float(row["re"])
    raw = load_simulation_tensor(cfg, sim_id, eval_split)
    series = normalize_fields(raw, stats)
    predict_fn = torch_predict_fn(model, device)

    if condition_on_re_enabled(cfg):
        if stats.re_min is None or stats.re_max is None:
            raise ValueError("preprocess stats missing re_min/re_max")
        scaling = ReScaling(stats.re_min, stats.re_max)
        re_scaled = scaling.scale(re_val)

        def _predict(state: np.ndarray) -> np.ndarray:
            conditioned = append_re_channel(state, re_scaled)
            return predict_fn(conditioned)

        preds, truths = one_step_model_predictions(series, _predict)
    else:
        preds, truths = one_step_model_predictions(series, predict_fn)

    return {
        "mse": mse(preds, truths),
        "relative_l2": relative_l2(preds, truths),
    }


"""RUNNER-----------------------------------------------------------"""


def run_re_generalization_evaluation(
    run_dir: str | Path,
    *,
    eval_split: str | None = None,
) -> ReGeneralizationResult:
    """
    Evaluate explicit OOD Reynolds performance (PRD §14 experiments 1–3).

    Parameters:
        run_dir (str | Path): CNN training bundle with ``model.pt``.
        eval_split (str | None): Temporal window; defaults to ``train`` temporal split.

    Returns:
        ReGeneralizationResult: JSON and heatmap paths.
    """
    _require_torch()
    run_path = Path(run_dir)
    cfg_path = run_path / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = OmegaConf.create(OmegaConf.load(cfg_path))
    if not bool(cfg.dataset.get("use_re_splits", False)):
        raise ValueError("re_generalization requires dataset.use_re_splits: true")

    device = torch.device(str(cfg.train.device))
    stats = load_preprocess_stats(run_path / "preprocess_stats.json")
    model = build_flow_cnn(cfg).to(device)
    payload = torch.load(
        run_path / MODEL_FILENAME,
        map_location=device,
        weights_only=False,
    )
    state = payload["model_state_dict"] if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()

    temporal_split = eval_split or str(cfg.dataset.normalization.fit_split)
    all_ids = re_split_simulation_ids(cfg)
    sim_ids: list[str] = []
    for group in ("train", "val", "test"):
        sim_ids.extend(all_ids.get(group, []))

    per_re: dict[str, Any] = {}
    for sim_id in sim_ids:
        row = lookup_metadata_row(dataset_root_from_config(cfg), sim_id)
        re_val = float(row["re"])
        metrics = evaluate_one_step_per_simulation(
            cfg,
            model,
            device,
            stats,
            sim_id,
            eval_split=temporal_split,
        )
        regime = _re_regime(cfg, re_val)
        per_re[str(int(re_val) if re_val == int(re_val) else re_val)] = {
            "sim_id": sim_id,
            "re": re_val,
            "regime": regime,
            "experiment": _experiment_id(regime),
            "metrics": metrics,
        }

    figures_dir = run_path / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = plot_re_generalization_heatmap(
        per_re,
        train_re=[float(r) for r in cfg.dataset.train_re],
        val_re=[float(r) for r in cfg.dataset.val_re],
        test_re=[float(r) for r in cfg.dataset.test_re],
        output_path=figures_dir / "re_generalization_heatmap.png",
    )

    report = {
        "schema_version": 1,
        "kind": "re_generalization",
        "eval_temporal_split": temporal_split,
        "run_dir": run_path.as_posix(),
        "per_re": per_re,
        "experiments": {
            "exp1_in_distribution": [
                k for k, v in per_re.items() if v["experiment"] == "exp1_in_distribution"
            ],
            "exp2_interpolation": [
                k for k, v in per_re.items() if v["experiment"] == "exp2_interpolation"
            ],
            "exp3_extrapolation": [
                k for k, v in per_re.items() if v["experiment"] == "exp3_extrapolation"
            ],
        },
    }
    metrics_path = run_path / RE_EVAL_FILENAME
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return ReGeneralizationResult(
        run_dir=run_path,
        metrics_path=metrics_path,
        heatmap_path=heatmap_path,
    )
