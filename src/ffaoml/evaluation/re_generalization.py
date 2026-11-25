"""
#############################################################################
### Reynolds generalization evaluation (PRD §14 experiments 1–3)
###
### @file re_generalization.py
### @author Sebastian Russo
### @date 2026
#############################################################################

One-step field error per Reynolds number and split regime (train / val / test),
plus optional multi-step rollout curves per simulation (Stage 3 ``stage3_cnn_re``).
"""

# Native imports
from __future__ import annotations

import json
from collections.abc import Callable
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
from ffaoml.contracts import STAGE3_CFDBENCH_SOURCE_ID
from ffaoml.data.catalog import dataset_root_from_config, lookup_metadata_row
from ffaoml.data.loading import load_simulation_tensor
from ffaoml.evaluation.metrics import mse, relative_l2
from ffaoml.evaluation.model_rollout import (
    model_rollout_curve,
    one_step_model_predictions,
    torch_predict_fn,
)
from ffaoml.evaluation.plots import (
    plot_error_vs_horizon,
    plot_re_generalization_heatmap,
)
from ffaoml.evaluation.rollout import RolloutCurve, sample_at_report_horizons
from ffaoml.manifests import RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME
from ffaoml.ml.conditioning import (
    ReScaling,
    append_re_channel,
    condition_on_re_enabled,
)
from ffaoml.ml.preprocessing import load_preprocess_stats, normalize_fields
from ffaoml.ml.splits import re_split_simulation_ids
from ffaoml.models.factory import build_flow_model
from ffaoml.training.checkpointing import load_model_weights
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
    rollout_horizon_path: Path | None


"""HELPERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def re_tolerance_from_config(cfg: DictConfig) -> float:
    import_cfg = cfg.dataset.get("import") or {}
    return float(import_cfg.get("re_tolerance", 2.0))


def _re_matches(re: float, target: float, tolerance: float) -> bool:
    return abs(re - target) <= tolerance


def _re_regime(cfg: DictConfig, re: float, *, tolerance: float | None = None) -> str:
    tol = tolerance if tolerance is not None else re_tolerance_from_config(cfg)
    for regime, key in (
        ("train", "train_re"),
        ("val", "val_re"),
        ("test", "test_re"),
    ):
        for target in cfg.dataset[key]:
            if _re_matches(re, float(target), tol):
                return regime
    return "unknown"


def validate_re_generalization_run(
    cfg: DictConfig,
    run_path: Path,
) -> dict[str, Any]:
    """
    Sanity-check a Stage 3 multi-Re training bundle before evaluation.

    Raises:
        FileNotFoundError: Missing ``model.pt`` or ``config.yaml``.
        ValueError: Config/bundle mismatch for Re-generalization eval.
    """
    model_path = run_path / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not bool(cfg.dataset.get("use_re_splits", False)):
        raise ValueError("re_generalization requires dataset.use_re_splits: true")
    if not condition_on_re_enabled(cfg):
        raise ValueError("Stage 3 Re eval expects model.condition_on_re: true")

    checks: dict[str, Any] = {
        "stage": int(cfg.dataset.get("stage", 0)),
        "source_id": str(cfg.dataset.get("source_id", "")),
        "dataset_output_root": str(cfg.dataset.output_root),
        "run_id": run_path.name,
    }
    source_id = checks["source_id"]
    if source_id == STAGE3_CFDBENCH_SOURCE_ID:
        root = dataset_root_from_config(cfg)
        if not (root / "metadata.csv").is_file():
            raise FileNotFoundError(
                f"dataset root {root} missing metadata.csv; import CFDBench first"
            )
    manifest_path = run_path / RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME
    if manifest_path.is_file():
        snap = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks["dataset_manifest_stage"] = snap.get("stage")
        checks["dataset_manifest_source_id"] = snap.get("source_id")
        if (
            source_id == STAGE3_CFDBENCH_SOURCE_ID
            and snap.get("source_id") != source_id
        ):
            raise ValueError(
                "run bundle dataset_manifest source_id does not match training config"
            )
    return checks


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
    predict_fn = _build_conditioned_predict_fn(cfg, model, device, stats, re_val)
    preds, truths = one_step_model_predictions(series, predict_fn)

    return {
        "mse": mse(preds, truths),
        "relative_l2": relative_l2(preds, truths),
    }


def _build_conditioned_predict_fn(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    stats,
    re_val: float,
) -> Callable[[np.ndarray], np.ndarray]:
    predict_fn = torch_predict_fn(model, device)
    if not condition_on_re_enabled(cfg):
        return predict_fn
    if stats.re_min is None or stats.re_max is None:
        raise ValueError("preprocess stats missing re_min/re_max")
    scaling = ReScaling(stats.re_min, stats.re_max)
    re_scaled = scaling.scale(re_val)

    def _predict(state: np.ndarray) -> np.ndarray:
        conditioned = append_re_channel(state, re_scaled)
        return predict_fn(conditioned)

    return _predict


def evaluate_rollout_per_simulation(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    stats,
    sim_id: str,
    *,
    eval_split: str,
    horizon: int,
) -> RolloutCurve:
    """Autoregressive rollout curve for one simulation at fixed Reynolds."""
    row = lookup_metadata_row(dataset_root_from_config(cfg), sim_id)
    re_val = float(row["re"])
    raw = load_simulation_tensor(cfg, sim_id, eval_split)
    series = normalize_fields(raw, stats)
    predict_fn = _build_conditioned_predict_fn(cfg, model, device, stats, re_val)
    return model_rollout_curve(series, predict_fn, horizon)


"""RUNNER-----------------------------------------------------------"""


def run_re_generalization_evaluation(
    run_dir: str | Path,
    *,
    eval_split: str | None = None,
    include_rollout: bool = True,
    rollout_horizon: int | None = None,
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
    bundle_checks = validate_re_generalization_run(cfg, run_path)

    device = torch.device(str(cfg.train.device))
    stats = load_preprocess_stats(run_path / "preprocess_stats.json")
    model = build_flow_model(cfg).to(device)
    payload = torch.load(
        run_path / MODEL_FILENAME,
        map_location=device,
        weights_only=False,
    )
    load_model_weights(model, payload)
    model.eval()

    temporal_split = eval_split or str(cfg.dataset.normalization.fit_split)
    all_ids = re_split_simulation_ids(cfg)
    sim_ids: list[str] = []
    for group in ("train", "val", "test"):
        sim_ids.extend(all_ids.get(group, []))

    per_re: dict[str, Any] = {}
    rollout_by_sim: dict[str, Any] = {}
    rollout_curves: dict[str, RolloutCurve] = {}
    horizon = (
        rollout_horizon
        if rollout_horizon is not None
        else int(cfg.eval.rollout_horizon)
    )
    report_horizons = [int(h) for h in cfg.eval.report_horizons]

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
        key = sim_id
        per_re[key] = {
            "sim_id": sim_id,
            "re": re_val,
            "regime": regime,
            "experiment": _experiment_id(regime),
            "metrics": metrics,
        }
        if include_rollout:
            try:
                curve = evaluate_rollout_per_simulation(
                    cfg,
                    model,
                    device,
                    stats,
                    sim_id,
                    eval_split=temporal_split,
                    horizon=horizon,
                )
            except ValueError:
                continue
            label = f"Re={int(re_val) if re_val == int(re_val) else re_val}"
            rollout_curves[label] = curve
            rollout_by_sim[sim_id] = {
                "re": re_val,
                "regime": regime,
                "horizons": list(curve.horizons),
                "mse": list(curve.mse),
                "relative_l2": list(curve.relative_l2),
                "n_starts": curve.n_starts,
                "at_report_horizons": sample_at_report_horizons(curve, report_horizons),
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

    rollout_horizon_path: Path | None = None
    if rollout_curves:
        rollout_horizon_path = plot_error_vs_horizon(
            rollout_curves,
            output_path=figures_dir / "error_vs_horizon_by_re.png",
        )

    report = {
        "schema_version": 1,
        "kind": "re_generalization",
        "eval_temporal_split": temporal_split,
        "run_dir": run_path.as_posix(),
        "bundle_checks": bundle_checks,
        "per_re": per_re,
        "rollout": {
            "horizon_requested": horizon,
            "report_horizons": report_horizons,
            "per_simulation": rollout_by_sim,
        },
        "experiments": {
            "exp1_in_distribution": [
                k
                for k, v in per_re.items()
                if v["experiment"] == "exp1_in_distribution"
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
        rollout_horizon_path=rollout_horizon_path,
    )
