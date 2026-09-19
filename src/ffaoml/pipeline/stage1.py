"""
#############################################################################
### Stage 1 full ML pipeline
###
### @file stage1.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import Zenodo data → CFD validation → baselines → train → evaluate (optional
ConvLSTM / FNO). Mirrors the skip-if-done and phase controls of a ``run_all``
orchestrator.
"""

# Native imports
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Third-party imports
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Project imports
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.data.catalog import dataset_root_from_config, resolve_simulation_dir
from ffaoml.data.io import FIELD_STORE_NAME
from ffaoml.data.sources.zenodo_re100 import import_stage1_from_config
from ffaoml.evaluation.model_report import run_model_evaluation
from ffaoml.evaluation.multistep_compare import run_multistep_comparison
from ffaoml.evaluation.runner import run_baseline_evaluation
from ffaoml.manifests import dataset_manifest_path
from ffaoml.training.train import run_cnn_training
from ffaoml.training.train_convlstm import run_convlstm_training
from ffaoml.app_logging import log_handler
from ffaoml.validation.stage1_zenodo import run_stage1_validation

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_CFD_VALIDATION_DIR = Path("results/cfd_validation/stage1_zenodo")
DEFAULT_RUNS_ROOT = Path("results/runs")

"""TYPES-----------------------------------------------------------"""


class PipelinePhase(str, Enum):
    """Ordered Stage 1 pipeline steps."""

    IMPORT = "import"
    CFD_VALIDATION = "cfd_validation"
    BASELINE_EVAL = "baseline_eval"
    TRAIN_CNN = "train_cnn"
    EVAL_CNN = "eval_cnn"
    TRAIN_CONVLSTM = "train_convlstm"
    COMPARE_MULTISTEP = "compare_multistep"
    TRAIN_FNO = "train_fno"
    EVAL_FNO = "eval_fno"


PHASE_ORDER: tuple[PipelinePhase, ...] = (
    PipelinePhase.IMPORT,
    PipelinePhase.CFD_VALIDATION,
    PipelinePhase.BASELINE_EVAL,
    PipelinePhase.TRAIN_CNN,
    PipelinePhase.EVAL_CNN,
    PipelinePhase.TRAIN_CONVLSTM,
    PipelinePhase.COMPARE_MULTISTEP,
    PipelinePhase.TRAIN_FNO,
    PipelinePhase.EVAL_FNO,
)


@dataclass
class Stage1PipelineOptions:
    """User-facing knobs for ``run_stage1_pipeline``."""

    repo_root: Path
    local_upstream: Path | None = None
    dataset_root: Path | None = None
    runs_root: Path = DEFAULT_RUNS_ROOT
    baseline_run_id: str = "stage1_baseline"
    cnn_run_id: str = "stage1_cnn"
    convlstm_run_id: str = "stage1_convlstm"
    fno_run_id: str = "stage1_fno"
    train_epochs: int = 50
    fno_epochs: int = 50
    skip_cfd_validation: bool = False
    with_convlstm: bool = False
    with_fno: bool = False
    hydra_overrides: list[str] = field(default_factory=list)
    force: bool = False
    dry_run: bool = False


class PrerequisiteError(RuntimeError):
    """Raised when a phase cannot run because upstream artifacts are missing."""


"""CONFIG-----------------------------------------------------------"""


def compose_stage1_config(
    repo_root: Path,
    *,
    dataset_root: Path | None = None,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load the default Hydra config with optional path overrides.

    Parameters:
        repo_root (Path): Repository root.
        dataset_root (Path | None): Override ``paths.dataset_root``.
        extra_overrides (list[str] | None): Additional Hydra overrides.

    Returns:
        DictConfig: Composed configuration.
    """
    overrides: list[str] = list(extra_overrides or [])
    if dataset_root is not None:
        root_posix = dataset_root.as_posix()
        overrides.append(f"paths.dataset_root={root_posix}")
        overrides.append(f"dataset.output_root={root_posix}")
    with initialize_config_dir(
        config_dir=str(config_dir(repo_root)),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=overrides)


"""MARKERS-----------------------------------------------------------"""


def _dataset_import_done(cfg: DictConfig) -> bool:
    root = dataset_root_from_config(cfg)
    manifest = dataset_manifest_path(root)
    zarr_dir = resolve_simulation_dir(cfg) / FIELD_STORE_NAME
    return manifest.is_file() and zarr_dir.is_dir()


def _cfd_validation_done(repo_root: Path) -> bool:
    out = repo_root / DEFAULT_CFD_VALIDATION_DIR
    return (out / "metrics.json").is_file() or (out / "SUMMARY.md").is_file()


def _run_artifact(run_dir: Path, name: str) -> bool:
    return (run_dir / name).is_file()


def _phase_done(phase: PipelinePhase, opts: Stage1PipelineOptions, cfg: DictConfig) -> bool:
    runs = opts.repo_root / opts.runs_root
    if phase == PipelinePhase.IMPORT:
        return _dataset_import_done(cfg)
    if phase == PipelinePhase.CFD_VALIDATION:
        return _cfd_validation_done(opts.repo_root)
    if phase == PipelinePhase.BASELINE_EVAL:
        return _run_artifact(runs / opts.baseline_run_id, "metrics.json")
    if phase == PipelinePhase.TRAIN_CNN:
        return _run_artifact(runs / opts.cnn_run_id, "model.pt")
    if phase == PipelinePhase.EVAL_CNN:
        return _run_artifact(runs / opts.cnn_run_id, "model_eval_metrics.json")
    if phase == PipelinePhase.TRAIN_CONVLSTM:
        return _run_artifact(runs / opts.convlstm_run_id, "model.pt")
    if phase == PipelinePhase.COMPARE_MULTISTEP:
        return _run_artifact(runs / opts.convlstm_run_id, "multistep_compare.json")
    if phase == PipelinePhase.TRAIN_FNO:
        return _run_artifact(runs / opts.fno_run_id, "model.pt")
    if phase == PipelinePhase.EVAL_FNO:
        return _run_artifact(runs / opts.fno_run_id, "model_eval_metrics.json")
    return False


"""PHASE PLAN-----------------------------------------------------------"""


def _phases_to_run(
    opts: Stage1PipelineOptions,
    *,
    only: PipelinePhase | None = None,
    start_from: PipelinePhase | None = None,
) -> list[PipelinePhase]:
    phases = list(PHASE_ORDER)
    if opts.skip_cfd_validation:
        phases = [p for p in phases if p != PipelinePhase.CFD_VALIDATION]
    if not opts.with_convlstm:
        phases = [
            p
            for p in phases
            if p not in (PipelinePhase.TRAIN_CONVLSTM, PipelinePhase.COMPARE_MULTISTEP)
        ]
    if not opts.with_fno:
        phases = [p for p in phases if p not in (PipelinePhase.TRAIN_FNO, PipelinePhase.EVAL_FNO)]
    if only is not None:
        return [only]
    if start_from is None:
        return phases
    idx = phases.index(start_from)
    return phases[idx:]


def _check_prerequisites(
    phases: list[PipelinePhase],
    opts: Stage1PipelineOptions,
    cfg: DictConfig,
) -> None:
    runs = opts.repo_root / opts.runs_root
    missing: list[str] = []
    for phase in phases:
        if phase == PipelinePhase.CFD_VALIDATION and not _dataset_import_done(cfg):
            missing.append("dataset import (manifest + fields.zarr)")
        if phase in (
            PipelinePhase.BASELINE_EVAL,
            PipelinePhase.TRAIN_CNN,
            PipelinePhase.EVAL_CNN,
            PipelinePhase.TRAIN_CONVLSTM,
            PipelinePhase.TRAIN_FNO,
        ) and not _dataset_import_done(cfg):
            missing.append("dataset import")
        if phase == PipelinePhase.EVAL_CNN and not _run_artifact(runs / opts.cnn_run_id, "model.pt"):
            missing.append(f"CNN run {opts.cnn_run_id}/model.pt")
        if phase == PipelinePhase.COMPARE_MULTISTEP:
            if not _run_artifact(runs / opts.cnn_run_id, "model.pt"):
                missing.append(f"CNN run {opts.cnn_run_id}/model.pt")
            if not _run_artifact(runs / opts.convlstm_run_id, "model.pt"):
                missing.append(f"ConvLSTM run {opts.convlstm_run_id}/model.pt")
        if phase == PipelinePhase.EVAL_FNO and not _run_artifact(runs / opts.fno_run_id, "model.pt"):
            missing.append(f"FNO run {opts.fno_run_id}/model.pt")
    if missing:
        lines = "\n".join(f"  - {item}" for item in missing)
        raise PrerequisiteError(f"Missing prerequisite(s):\n{lines}")


def _confirm_rerun(
    phase: PipelinePhase,
    *,
    force: bool,
    confirm_fn: Callable[[str], str] | None,
) -> bool:
    if force:
        return True
    message = f"Step '{phase.value}' already completed. Run again? [Y/N]: "
    if confirm_fn is not None:
        answer = confirm_fn(message)
    elif not sys.stdin.isatty():
        log_handler.info(
            "[SKIP] %s (outputs exist; use --force to re-run).",
            phase.value,
        )
        return False
    else:
        answer = input(message)
    normalized = answer.strip().lower()
    return normalized in {"y", "yes"}


"""RUNNERS-----------------------------------------------------------"""


def _run_phase(
    phase: PipelinePhase,
    cfg: DictConfig,
    opts: Stage1PipelineOptions,
) -> Any:
    runs_root = opts.repo_root / opts.runs_root
    if phase == PipelinePhase.IMPORT:
        return import_stage1_from_config(
            cfg,
            local_upstream=opts.local_upstream,
            repo_root=opts.repo_root,
        )
    if phase == PipelinePhase.CFD_VALIDATION:
        out = opts.repo_root / DEFAULT_CFD_VALIDATION_DIR
        return run_stage1_validation(cfg, output_dir=out)
    if phase == PipelinePhase.BASELINE_EVAL:
        out = run_directory(runs_root, opts.baseline_run_id)
        seed_from_config(cfg)
        return run_baseline_evaluation(cfg, out, repo_root=opts.repo_root)
    if phase == PipelinePhase.TRAIN_CNN:
        out = run_directory(runs_root, opts.cnn_run_id)
        seed_from_config(cfg)
        return run_cnn_training(cfg, out, repo_root=opts.repo_root)
    if phase == PipelinePhase.EVAL_CNN:
        return run_model_evaluation(runs_root / opts.cnn_run_id)
    if phase == PipelinePhase.TRAIN_CONVLSTM:
        lstm_cfg = compose_stage1_config(
            opts.repo_root,
            dataset_root=opts.dataset_root,
            extra_overrides=[
                *opts.hydra_overrides,
                "model=convlstm",
                f"train.epochs={opts.train_epochs}",
            ],
        )
        out = run_directory(runs_root, opts.convlstm_run_id)
        seed_from_config(lstm_cfg)
        return run_convlstm_training(lstm_cfg, out, repo_root=opts.repo_root)
    if phase == PipelinePhase.COMPARE_MULTISTEP:
        return run_multistep_comparison(
            runs_root / opts.cnn_run_id,
            runs_root / opts.convlstm_run_id,
            output_dir=runs_root / opts.convlstm_run_id,
        )
    if phase == PipelinePhase.TRAIN_FNO:
        fno_cfg = compose_stage1_config(
            opts.repo_root,
            dataset_root=opts.dataset_root,
            extra_overrides=[
                *opts.hydra_overrides,
                "model=fno",
                f"train.epochs={opts.fno_epochs}",
            ],
        )
        out = run_directory(runs_root, opts.fno_run_id)
        seed_from_config(fno_cfg)
        return run_cnn_training(fno_cfg, out, repo_root=opts.repo_root)
    if phase == PipelinePhase.EVAL_FNO:
        return run_model_evaluation(runs_root / opts.fno_run_id)
    raise ValueError(f"Unhandled phase: {phase}")


"""PUBLIC API-----------------------------------------------------------"""


def run_stage1_pipeline(
    opts: Stage1PipelineOptions,
    *,
    only: PipelinePhase | None = None,
    start_from: PipelinePhase | None = None,
    confirm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Execute the Stage 1 pipeline and return a JSON-serializable summary.

    Parameters:
        opts (Stage1PipelineOptions): Paths, run ids, and feature flags.
        only (PipelinePhase | None): Run a single phase.
        start_from (PipelinePhase | None): Run this phase and all later ones.
        confirm_fn (Callable | None): Inject prompt for tests.

    Returns:
        dict[str, Any]: Plan, per-phase timings, and skipped steps.
    """
    cfg = compose_stage1_config(
        opts.repo_root,
        dataset_root=opts.dataset_root,
        extra_overrides=[
            *opts.hydra_overrides,
            f"train.epochs={opts.train_epochs}",
        ],
    )
    phases = _phases_to_run(opts, only=only, start_from=start_from)
    plan = {
        "phases": [p.value for p in phases],
        "dataset_root": str(dataset_root_from_config(cfg)),
        "runs_root": str(opts.repo_root / opts.runs_root),
        "cnn_run_id": opts.cnn_run_id,
        "with_convlstm": opts.with_convlstm,
        "with_fno": opts.with_fno,
    }
    already = {p.value: _phase_done(p, opts, cfg) for p in phases}

    if opts.dry_run:
        return {"status": "dry_run", "plan": plan, "already_ran": already}

    _check_prerequisites(phases, opts, cfg)

    results: dict[str, Any] = {
        "status": "ok",
        "plan": plan,
        "phases": {},
        "skipped_phases": [],
    }

    for phase in phases:
        if _phase_done(phase, opts, cfg) and not _confirm_rerun(
            phase, force=opts.force, confirm_fn=confirm_fn
        ):
            results["skipped_phases"].append(phase.value)
            log_handler.info("=== [%s] skipped ===", phase.value)
            continue

        log_handler.info("=== [%s] starting ===", phase.value)
        started = time.perf_counter()
        phase_result = _run_phase(phase, cfg, opts)
        elapsed = time.perf_counter() - started
        results["phases"][phase.value] = {
            "elapsed_seconds": round(elapsed, 1),
            "result": _summarize_result(phase_result),
        }
        log_handler.info("=== [%s] done (%.1fs) ===", phase.value, elapsed)

    return results


def _summarize_result(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "output_dir"):
        return {"output_dir": str(result.output_dir)}
    if hasattr(result, "run_dir"):
        return {"run_dir": str(result.run_dir)}
    if hasattr(result, "store_path"):
        return {
            "store_path": str(result.store_path),
            "manifest_path": str(getattr(result, "manifest_path", "")),
        }
    return str(result)


def print_summary(results: dict[str, Any]) -> None:
    """Log the pipeline summary JSON."""
    log_handler.info("%s", json.dumps(results, indent=2, default=str))
