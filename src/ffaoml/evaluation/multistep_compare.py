"""
#############################################################################
### CNN vs ConvLSTM multi-step comparison
###
### @file multistep_compare.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Compare recursive CNN rollout against ConvLSTM rollout at configured horizons
(PRD §12 long-horizon stability).
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
from ffaoml.data.loading import load_split_tensor
from ffaoml.evaluation.model_rollout import (
    convlstm_rollout_curve,
    model_rollout_curve,
    torch_predict_fn,
)
from ffaoml.evaluation.plots import plot_error_vs_horizon
from ffaoml.evaluation.rollout import rollout_curve, sample_at_report_horizons
from ffaoml.ml.preprocessing import load_preprocess_stats, normalize_fields
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.models.convlstm import build_flow_convlstm
from ffaoml.training.train import MODEL_FILENAME

"""CONSTANTS-----------------------------------------------------------"""
COMPARE_FILENAME = "multistep_compare.json"
FIGURES_DIRNAME = "figures"

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class MultistepCompareResult:
    """Paths written by ``run_multistep_comparison``."""

    output_dir: Path
    metrics_path: Path
    horizon_figure: Path | None


"""LOADERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def _load_run_config(run_dir: Path) -> DictConfig:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config.yaml in {run_dir}")
    return OmegaConf.create(OmegaConf.load(path))


def _load_model_weights(
    cfg: DictConfig,
    run_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    name = str(cfg.model.name)
    if name == "cnn":
        model = build_flow_cnn(cfg).to(device)
    elif name == "convlstm":
        model = build_flow_convlstm(cfg).to(device)
    else:
        raise ValueError(f"unsupported model for multistep compare: {name}")
    ckpt_path = run_dir / MODEL_FILENAME
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


"""RUNNER-----------------------------------------------------------"""


def run_multistep_comparison(
    cnn_run_dir: str | Path,
    convlstm_run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    split: str | None = None,
) -> MultistepCompareResult:
    """
    Report rollout MSE at ``eval.report_horizons`` for CNN vs ConvLSTM.

    Parameters:
        cnn_run_dir (str | Path): One-step CNN training bundle.
        convlstm_run_dir (str | Path): ConvLSTM training bundle.
        output_dir (str | Path | None): Defaults to ``convlstm_run_dir``.
        split (str | None): Eval split; uses ``cfg.eval.split`` from ConvLSTM run.

    Returns:
        MultistepCompareResult: JSON and optional figure paths.
    """
    _require_torch()
    cnn_path = Path(cnn_run_dir)
    lstm_path = Path(convlstm_run_dir)
    out = Path(output_dir or lstm_path)
    out.mkdir(parents=True, exist_ok=True)

    lstm_cfg = _load_run_config(lstm_path)
    cnn_cfg = _load_run_config(cnn_path)
    eval_split = split or str(lstm_cfg.eval.split)
    horizon = int(lstm_cfg.eval.rollout_horizon)
    report_horizons = list(lstm_cfg.eval.report_horizons)
    device = torch.device(str(lstm_cfg.train.device))

    stats = load_preprocess_stats(lstm_path / "preprocess_stats.json")
    raw = load_split_tensor(lstm_cfg, eval_split)
    series = normalize_fields(raw, stats)

    cnn_model = _load_model_weights(cnn_cfg, cnn_path, device)
    lstm_model = _load_model_weights(lstm_cfg, lstm_path, device)

    cnn_curve = model_rollout_curve(
        series, torch_predict_fn(cnn_model, device), horizon
    )
    lstm_curve = convlstm_rollout_curve(series, lstm_model, device, horizon)
    persist_curve = rollout_curve(series, "persistence", horizon)

    figures_dir = out / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    horizon_fig = plot_error_vs_horizon(
        {
            "cnn_recursive": cnn_curve,
            "convlstm": lstm_curve,
            "persistence": persist_curve,
        },
        output_path=figures_dir / "cnn_vs_convlstm_horizon.png",
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "multistep_compare",
        "split": eval_split,
        "report_horizons": report_horizons,
        "cnn_run_dir": cnn_path.as_posix(),
        "convlstm_run_dir": lstm_path.as_posix(),
        "rollout": {
            "cnn_recursive": {
                "at_report_horizons": sample_at_report_horizons(
                    cnn_curve, report_horizons
                ),
            },
            "convlstm": {
                "at_report_horizons": sample_at_report_horizons(
                    lstm_curve, report_horizons
                ),
            },
            "persistence": {
                "at_report_horizons": sample_at_report_horizons(
                    persist_curve, report_horizons
                ),
            },
        },
        "long_horizon_note": (
            "Recursive one-step CNN errors typically grow faster than ConvLSTM "
            "when hidden state is carried; see experiments/stage1_multistep_drift.md."
        ),
    }
    metrics_path = out / COMPARE_FILENAME
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return MultistepCompareResult(
        output_dir=out,
        metrics_path=metrics_path,
        horizon_figure=horizon_fig,
    )
