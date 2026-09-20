"""
#############################################################################
### Stage 2 MeshGraphNets pipeline
###
### @file stage2.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import cylinder_flow trajectories → mesh validation figures → train MeshGraphNet
→ rollout evaluation.
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
from ffaoml.data.catalog import list_mesh_trajectories_for_split
from ffaoml.data.sources.meshgraphnets_cylinder import run_meshgraphnets_import
from ffaoml.evaluation.evaluate_mesh_model import run_mesh_model_evaluation
from ffaoml.manifests import dataset_manifest_path
from ffaoml.pipeline.stage1 import PrerequisiteError, print_summary
from ffaoml.training.train_meshgraphnet import run_meshgraphnet_training
from ffaoml.validation import stage2_meshgraphnets as mesh_validation

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_RUNS_ROOT = Path("results/runs")
STAGE2_CFD_VALIDATION_DIR = mesh_validation.DEFAULT_OUTPUT


def hydra_overrides_select_stage2(extra_overrides: list[str]) -> bool:
    """True when Hydra CLI overrides request the Stage 2 dataset group."""
    return any(
        item.startswith("dataset=stage2_meshgraphnets") for item in extra_overrides
    )


STAGE2_HYDRA_DEFAULTS: tuple[str, ...] = (
    "dataset=stage2_meshgraphnets",
    "model=meshgraphnet",
    "train=meshgraphnet",
    "eval=meshgraphnet",
)

"""TYPES-----------------------------------------------------------"""


class Stage2PipelinePhase(StrEnum):
    """Ordered Stage 2 pipeline steps."""

    IMPORT = "import"
    MESH_VALIDATION = "mesh_validation"
    TRAIN_MESHGN = "train_meshgn"
    EVAL_MESHGN = "eval_meshgn"


STAGE2_PHASE_ORDER: tuple[Stage2PipelinePhase, ...] = (
    Stage2PipelinePhase.IMPORT,
    Stage2PipelinePhase.MESH_VALIDATION,
    Stage2PipelinePhase.TRAIN_MESHGN,
    Stage2PipelinePhase.EVAL_MESHGN,
)


@dataclass
class Stage2PipelineOptions:
    """User-facing knobs for ``run_stage2_pipeline``."""

    repo_root: Path
    runs_root: Path = DEFAULT_RUNS_ROOT
    mesh_run_id: str = "stage2_meshgn"
    train_epochs: int = 50
    import_splits: tuple[str, ...] = ("train", "val", "test")
    max_trajectories_per_split: int | None = None
    force_download: bool = False
    skip_mesh_validation: bool = False
    hydra_overrides: list[str] = field(default_factory=list)
    force: bool = False
    dry_run: bool = False


"""CONFIG-----------------------------------------------------------"""


def compose_stage2_config(
    repo_root: Path,
    *,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """Load Stage 2 Hydra config with mesh defaults."""
    overrides = [
        *STAGE2_HYDRA_DEFAULTS,
        *(extra_overrides or []),
    ]
    with initialize_config_dir(
        config_dir=str(config_dir(repo_root)),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=overrides)


def stage2_dataset_root(cfg: DictConfig) -> Path:
    return Path(str(cfg.dataset.output_root))


"""MARKERS-----------------------------------------------------------"""


def _mesh_import_done(cfg: DictConfig) -> bool:
    root = stage2_dataset_root(cfg)
    manifest = dataset_manifest_path(root)
    if not manifest.is_file():
        return False
    return bool(list_mesh_trajectories_for_split(cfg, "train"))


def _mesh_validation_done(repo_root: Path) -> bool:
    out = repo_root / STAGE2_CFD_VALIDATION_DIR
    return (out / "metrics.json").is_file() or (out / "SUMMARY.md").is_file()


def _run_artifact(run_dir: Path, name: str) -> bool:
    return (run_dir / name).is_file()


def _phase_done(
    phase: Stage2PipelinePhase,
    opts: Stage2PipelineOptions,
    cfg: DictConfig,
) -> bool:
    runs = opts.repo_root / opts.runs_root
    if phase == Stage2PipelinePhase.IMPORT:
        return _mesh_import_done(cfg)
    if phase == Stage2PipelinePhase.MESH_VALIDATION:
        return _mesh_validation_done(opts.repo_root)
    if phase == Stage2PipelinePhase.TRAIN_MESHGN:
        return _run_artifact(runs / opts.mesh_run_id, "model.pt")
    if phase == Stage2PipelinePhase.EVAL_MESHGN:
        return _run_artifact(runs / opts.mesh_run_id, "model_eval_metrics.json")
    return False


"""PHASE PLAN-----------------------------------------------------------"""


def _phases_to_run(
    opts: Stage2PipelineOptions,
    *,
    only: Stage2PipelinePhase | None = None,
    start_from: Stage2PipelinePhase | None = None,
) -> list[Stage2PipelinePhase]:
    phases = list(STAGE2_PHASE_ORDER)
    if opts.skip_mesh_validation:
        phases = [p for p in phases if p != Stage2PipelinePhase.MESH_VALIDATION]
    if only is not None:
        return [only]
    if start_from is None:
        return phases
    idx = phases.index(start_from)
    return phases[idx:]


def _phase_runs_before(
    phases: list[Stage2PipelinePhase],
    earlier: Stage2PipelinePhase,
    later: Stage2PipelinePhase,
) -> bool:
    if earlier not in phases or later not in phases:
        return False
    return phases.index(earlier) < phases.index(later)


def _check_prerequisites(
    phases: list[Stage2PipelinePhase],
    opts: Stage2PipelineOptions,
    cfg: DictConfig,
) -> None:
    runs = opts.repo_root / opts.runs_root
    missing: list[str] = []
    dataset_phases = (
        Stage2PipelinePhase.MESH_VALIDATION,
        Stage2PipelinePhase.TRAIN_MESHGN,
        Stage2PipelinePhase.EVAL_MESHGN,
    )
    for phase in phases:
        if phase in dataset_phases and not _mesh_import_done(cfg):
            if not _phase_runs_before(phases, Stage2PipelinePhase.IMPORT, phase):
                missing.append("mesh dataset import (manifest + trajectories)")
        if phase == Stage2PipelinePhase.EVAL_MESHGN and not _run_artifact(
            runs / opts.mesh_run_id, "model.pt"
        ):
            if not _phase_runs_before(phases, Stage2PipelinePhase.TRAIN_MESHGN, phase):
                missing.append(f"mesh run {opts.mesh_run_id}/model.pt")
    if missing:
        lines = "\n".join(f"  - {item}" for item in missing)
        raise PrerequisiteError(f"Missing prerequisite(s):\n{lines}")


def _confirm_rerun(
    phase: Stage2PipelinePhase,
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
    phase: Stage2PipelinePhase,
    cfg: DictConfig,
    opts: Stage2PipelineOptions,
) -> Any:
    runs_root = opts.repo_root / opts.runs_root
    if phase == Stage2PipelinePhase.IMPORT:
        return run_meshgraphnets_import(
            cfg,
            splits=opts.import_splits,
            max_trajectories_per_split=opts.max_trajectories_per_split,
            repo_root=opts.repo_root,
            force_download=opts.force_download,
        )
    if phase == Stage2PipelinePhase.MESH_VALIDATION:
        out = opts.repo_root / STAGE2_CFD_VALIDATION_DIR
        return mesh_validation.run_stage2_mesh_validation(cfg, output_dir=out)
    if phase == Stage2PipelinePhase.TRAIN_MESHGN:
        out = run_directory(runs_root, opts.mesh_run_id)
        seed_from_config(cfg)
        return run_meshgraphnet_training(cfg, out, repo_root=opts.repo_root)
    if phase == Stage2PipelinePhase.EVAL_MESHGN:
        return run_mesh_model_evaluation(runs_root / opts.mesh_run_id)
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
            "trajectories_imported": getattr(result, "trajectories_imported", None),
        }
    if hasattr(result, "metrics_path"):
        return {"metrics_path": str(result.metrics_path)}
    return str(result)


"""PUBLIC API-----------------------------------------------------------"""


def run_stage2_pipeline(
    opts: Stage2PipelineOptions,
    *,
    only: Stage2PipelinePhase | None = None,
    start_from: Stage2PipelinePhase | None = None,
    confirm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Execute the Stage 2 pipeline and return a JSON-serializable summary.

    Parameters:
        opts (Stage2PipelineOptions): Paths, run ids, and import caps.
        only (Stage2PipelinePhase | None): Run a single phase.
        start_from (Stage2PipelinePhase | None): Run this phase and all later ones.
        confirm_fn (Callable | None): Inject prompt for tests.

    Returns:
        dict[str, Any]: Plan, per-phase timings, and skipped steps.
    """
    cfg = compose_stage2_config(
        opts.repo_root,
        extra_overrides=[
            *opts.hydra_overrides,
            f"train.epochs={opts.train_epochs}",
        ],
    )
    phases = _phases_to_run(opts, only=only, start_from=start_from)
    plan = {
        "stage": 2,
        "phases": [p.value for p in phases],
        "dataset_root": str(stage2_dataset_root(cfg)),
        "runs_root": str(opts.repo_root / opts.runs_root),
        "mesh_run_id": opts.mesh_run_id,
        "import_splits": list(opts.import_splits),
        "max_trajectories_per_split": opts.max_trajectories_per_split,
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
        if phase == Stage2PipelinePhase.IMPORT:
            cfg = compose_stage2_config(
                opts.repo_root,
                extra_overrides=[
                    *opts.hydra_overrides,
                    f"train.epochs={opts.train_epochs}",
                ],
            )
            plan["dataset_root"] = str(stage2_dataset_root(cfg))
        elapsed = time.perf_counter() - started
        results["phases"][phase.value] = {
            "elapsed_seconds": round(elapsed, 1),
            "result": _summarize_result(phase_result),
        }
        log_handler.info("=== [%s] done (%.1fs) ===", phase.value, elapsed)

    return results


__all__ = [
    "STAGE2_PHASE_ORDER",
    "Stage2PipelineOptions",
    "Stage2PipelinePhase",
    "compose_stage2_config",
    "hydra_overrides_select_stage2",
    "print_summary",
    "run_stage2_pipeline",
]
