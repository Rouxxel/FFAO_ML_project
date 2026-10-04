"""
#############################################################################
### Stage 3 CFDBench / multi-Re CNN pipeline
###
### @file stage3.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import CFDBench cylinder cases → grid validation figures → train Re-conditioned CNN
→ Reynolds generalization evaluation.
"""

# Native imports
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# Third-party imports
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ffaoml.app_logging import log_handler

# Project imports
from ffaoml.config import config_dir, run_directory, seed_from_config
from ffaoml.data.io import simulation_store_path
from ffaoml.data.metadata import read_metadata_rows
from ffaoml.data.sources.cfdbench_cylinder import run_cfdbench_import
from ffaoml.evaluation.re_generalization import (
    RE_EVAL_FILENAME,
    run_re_generalization_evaluation,
)
from ffaoml.manifests import dataset_manifest_path
from ffaoml.pipeline.stage1 import PrerequisiteError, print_summary
from ffaoml.training.train import run_cnn_training
from ffaoml.validation import stage3_cfdbench as grid_validation

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_RUNS_ROOT = Path("results/runs")
STAGE3_CFD_VALIDATION_DIR = grid_validation.DEFAULT_OUTPUT


def hydra_overrides_select_stage3(extra_overrides: list[str]) -> bool:
    """True when Hydra CLI overrides request the Stage 3 dataset group."""
    return any(
        item.startswith("dataset=stage3_cfdbench") for item in extra_overrides
    )


STAGE3_HYDRA_DEFAULTS: tuple[str, ...] = (
    "dataset=stage3_cfdbench",
    "model=cnn_re",
    "train=default",
    "eval=default",
)

"""TYPES-----------------------------------------------------------"""


class Stage3PipelinePhase(StrEnum):
    """Ordered Stage 3 pipeline steps."""

    IMPORT = "import"
    GRID_VALIDATION = "grid_validation"
    TRAIN_CNN_RE = "train_cnn_re"
    EVAL_RE = "eval_re"


STAGE3_PHASE_ORDER: tuple[Stage3PipelinePhase, ...] = (
    Stage3PipelinePhase.IMPORT,
    Stage3PipelinePhase.GRID_VALIDATION,
    Stage3PipelinePhase.TRAIN_CNN_RE,
    Stage3PipelinePhase.EVAL_RE,
)


@dataclass
class Stage3PipelineOptions:
    """User-facing knobs for ``run_stage3_pipeline``."""

    repo_root: Path
    runs_root: Path = DEFAULT_RUNS_ROOT
    cnn_run_id: str = "stage3_cnn_re"
    train_epochs: int = 50
    max_cases: int | None = None
    re_filter: list[float] | None = None
    local_data_root: Path | None = None
    force_download: bool = False
    skip_grid_validation: bool = False
    hydra_overrides: list[str] = field(default_factory=list)
    force: bool = False
    dry_run: bool = False


"""CONFIG-----------------------------------------------------------"""


def compose_stage3_config(
    repo_root: Path,
    *,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """Load Stage 3 Hydra config with CFDBench + cnn_re defaults."""
    overrides = [
        *STAGE3_HYDRA_DEFAULTS,
        *(extra_overrides or []),
    ]
    with initialize_config_dir(
        config_dir=str(config_dir(repo_root)),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=overrides)


def stage3_dataset_root(cfg: DictConfig) -> Path:
    return Path(str(cfg.dataset.output_root))


"""MARKERS-----------------------------------------------------------"""


def _cfdbench_import_done(cfg: DictConfig) -> bool:
    root = stage3_dataset_root(cfg)
    manifest = dataset_manifest_path(root)
    if not manifest.is_file():
        return False
    for row in read_metadata_rows(root):
        sim_dir = (root / str(row["path"])).resolve()
        if simulation_store_path(sim_dir).is_dir():
            return True
    return False


def _grid_validation_done(repo_root: Path) -> bool:
    out = repo_root / STAGE3_CFD_VALIDATION_DIR
    return (out / "metrics.json").is_file() or (out / "SUMMARY.md").is_file()


def _run_artifact(run_dir: Path, name: str) -> bool:
    return (run_dir / name).is_file()


def _phase_done(
    phase: Stage3PipelinePhase,
    opts: Stage3PipelineOptions,
    cfg: DictConfig,
) -> bool:
    runs = opts.repo_root / opts.runs_root
    if phase == Stage3PipelinePhase.IMPORT:
        return _cfdbench_import_done(cfg)
    if phase == Stage3PipelinePhase.GRID_VALIDATION:
        return _grid_validation_done(opts.repo_root)
    if phase == Stage3PipelinePhase.TRAIN_CNN_RE:
        return _run_artifact(runs / opts.cnn_run_id, "model.pt")
    if phase == Stage3PipelinePhase.EVAL_RE:
        return _run_artifact(runs / opts.cnn_run_id, RE_EVAL_FILENAME)
    return False


"""PHASE PLAN-----------------------------------------------------------"""


def _phases_to_run(
    opts: Stage3PipelineOptions,
    *,
    only: Stage3PipelinePhase | None = None,
    start_from: Stage3PipelinePhase | None = None,
) -> list[Stage3PipelinePhase]:
    phases = list(STAGE3_PHASE_ORDER)
    if opts.skip_grid_validation:
        phases = [p for p in phases if p != Stage3PipelinePhase.GRID_VALIDATION]
    if only is not None:
        return [only]
    if start_from is None:
        return phases
    idx = phases.index(start_from)
    return phases[idx:]


def _phase_runs_before(
    phases: list[Stage3PipelinePhase],
    earlier: Stage3PipelinePhase,
    later: Stage3PipelinePhase,
) -> bool:
    if earlier not in phases or later not in phases:
        return False
    return phases.index(earlier) < phases.index(later)


def _check_prerequisites(
    phases: list[Stage3PipelinePhase],
    opts: Stage3PipelineOptions,
    cfg: DictConfig,
) -> None:
    runs = opts.repo_root / opts.runs_root
    missing: list[str] = []
    dataset_phases = (
        Stage3PipelinePhase.GRID_VALIDATION,
        Stage3PipelinePhase.TRAIN_CNN_RE,
        Stage3PipelinePhase.EVAL_RE,
    )
    for phase in phases:
        if phase in dataset_phases and not _cfdbench_import_done(cfg):
            if not _phase_runs_before(phases, Stage3PipelinePhase.IMPORT, phase):
                missing.append("CFDBench import (manifest + Zarr simulations)")
        if phase == Stage3PipelinePhase.EVAL_RE and not _run_artifact(
            runs / opts.cnn_run_id, "model.pt"
        ):
            if not _phase_runs_before(
                phases, Stage3PipelinePhase.TRAIN_CNN_RE, phase
            ):
                missing.append(f"CNN run {opts.cnn_run_id}/model.pt")
    if missing:
        lines = "\n".join(f"  - {item}" for item in missing)
        raise PrerequisiteError(f"Missing prerequisite(s):\n{lines}")


def _confirm_rerun(
    phase: Stage3PipelinePhase,
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
    return answer.strip().lower() in {"y", "yes"}


"""RUNNERS-----------------------------------------------------------"""


def _run_phase(
    phase: Stage3PipelinePhase,
    cfg: DictConfig,
    opts: Stage3PipelineOptions,
) -> Any:
    runs_root = opts.repo_root / opts.runs_root
    if phase == Stage3PipelinePhase.IMPORT:
        log_handler.info(
            "Stage 3 import: max_cases=%s re_filter=%s local_data_root=%s",
            opts.max_cases if opts.max_cases is not None else "unlimited",
            opts.re_filter,
            opts.local_data_root,
        )
        return run_cfdbench_import(
            cfg,
            repo_root=opts.repo_root,
            local_data_root=opts.local_data_root,
            force_download=opts.force_download,
            re_filter=opts.re_filter,
            max_cases=opts.max_cases,
        )
    if phase == Stage3PipelinePhase.GRID_VALIDATION:
        out = opts.repo_root / STAGE3_CFD_VALIDATION_DIR
        return grid_validation.run_stage3_cfdbench_validation(cfg, output_dir=out)
    if phase == Stage3PipelinePhase.TRAIN_CNN_RE:
        out = run_directory(runs_root, opts.cnn_run_id)
        seed_from_config(cfg)
        return run_cnn_training(cfg, out, repo_root=opts.repo_root)
    if phase == Stage3PipelinePhase.EVAL_RE:
        return run_re_generalization_evaluation(runs_root / opts.cnn_run_id)
    raise ValueError(f"Unhandled phase: {phase}")


def _summarize_result(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "output_dir"):
        return {"output_dir": str(result.output_dir)}
    if hasattr(result, "run_dir"):
        return {"run_dir": str(result.run_dir)}
    if hasattr(result, "dataset_root"):
        return {
            "dataset_root": str(result.dataset_root),
            "manifest_path": str(getattr(result, "manifest_path", "")),
            "cases_imported": getattr(result, "cases_imported", None),
        }
    if hasattr(result, "metrics_path"):
        return {"metrics_path": str(result.metrics_path)}
    return str(result)


"""PUBLIC API-----------------------------------------------------------"""


def run_stage3_pipeline(
    opts: Stage3PipelineOptions,
    *,
    only: Stage3PipelinePhase | None = None,
    start_from: Stage3PipelinePhase | None = None,
    confirm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Execute the Stage 3 pipeline and return a JSON-serializable summary.

    Parameters:
        opts (Stage3PipelineOptions): Paths, run ids, and import caps.
        only (Stage3PipelinePhase | None): Run a single phase.
        start_from (Stage3PipelinePhase | None): Run this phase and all later ones.
        confirm_fn (Callable | None): Inject prompt for tests.

    Returns:
        dict[str, Any]: Plan, per-phase timings, and skipped steps.
    """
    cfg = compose_stage3_config(
        opts.repo_root,
        extra_overrides=[
            *opts.hydra_overrides,
            f"train.epochs={opts.train_epochs}",
        ],
    )
    phases = _phases_to_run(opts, only=only, start_from=start_from)
    plan = {
        "stage": 3,
        "phases": [p.value for p in phases],
        "dataset_root": str(stage3_dataset_root(cfg)),
        "runs_root": str(opts.repo_root / opts.runs_root),
        "cnn_run_id": opts.cnn_run_id,
        "max_cases": opts.max_cases,
        "re_filter": opts.re_filter,
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
        if phase == Stage3PipelinePhase.IMPORT:
            cfg = compose_stage3_config(
                opts.repo_root,
                extra_overrides=[
                    *opts.hydra_overrides,
                    f"train.epochs={opts.train_epochs}",
                ],
            )
            plan["dataset_root"] = str(stage3_dataset_root(cfg))
        elapsed = time.perf_counter() - started
        results["phases"][phase.value] = {
            "elapsed_seconds": round(elapsed, 1),
            "result": _summarize_result(phase_result),
        }
        log_handler.info("=== [%s] done (%.1fs) ===", phase.value, elapsed)

    return results


__all__ = [
    "STAGE3_PHASE_ORDER",
    "Stage3PipelineOptions",
    "Stage3PipelinePhase",
    "compose_stage3_config",
    "hydra_overrides_select_stage3",
    "print_summary",
    "run_stage3_pipeline",
]
