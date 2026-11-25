#!/usr/bin/env python3
"""
#############################################################################
### Clean local pipeline artifacts
###
### @file clean_pipeline_artifacts.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Remove generated data under the repo (datasets, training runs, cache, logs) so
you can re-run ``main.py``, ``main.py --stage2``, or ``main.py --stage3`` from a
clean slate. Pipelines
still support skip-if-done; delete only what you need.

Presets:

- ``pipeline``- **Stage 1 and Stage 2** on disk: entire ``dataset/`` (Zenodo,
  generated, **meshgraphnets**), entire ``.cache/`` (Zenodo/LBM **and** mesh
  TFRecords), all ``results/runs/``, all ``results/cfd_validation/`` (Stage 1
  **and** ``stage2_meshgraphnets/``), plus ``log/``.
- ``stage2``- Mesh-focused subset: ``dataset-meshgraphnets``, mesh cache,
  ``cfd-stage2``, and all runs (same ``runs`` target as ``pipeline``).
- ``stage3``- CFDBench subset: ``dataset-cfdbench``, CFDBench cache,
  ``cfd-stage3``, and all runs.
- ``models``- ``results/runs/`` only.
- ``all``- ``pipeline`` plus Hydra dirs and ``.local_test_runs/``.

Quick reference::

    # Inspect
    python scripts/clean_pipeline_artifacts.py --list
    python scripts/clean_pipeline_artifacts.py --preset pipeline --dry-run

    # Full reset for both stages (see preset list above)
    python scripts/clean_pipeline_artifacts.py --preset pipeline --yes

    # Stage 2 only (still wipes all run folders)
    python scripts/clean_pipeline_artifacts.py --preset stage2 --yes

    # Stage 3 only (CFDBench import + stage3 validation figures + all runs)
    python scripts/clean_pipeline_artifacts.py --preset stage3 --yes

    # Common partial cleanups (--target repeatable; names from --list)
    python scripts/clean_pipeline_artifacts.py --target dataset-generated --yes
    python scripts/clean_pipeline_artifacts.py --target dataset-zenodo --yes
    python scripts/clean_pipeline_artifacts.py --target dataset-meshgraphnets --yes
    python scripts/clean_pipeline_artifacts.py --preset models --yes
    python scripts/clean_pipeline_artifacts.py --run stage1_cnn --yes
    python scripts/clean_pipeline_artifacts.py --run stage2_meshgn --yes

    # Nuclear (adds Hydra outputs/ + .local_test_runs/)
    python scripts/clean_pipeline_artifacts.py --preset all --yes
"""

# Native imports
import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler, shutdown_logger
from ffaoml.data.stage1_layout import (
    DATASET_BASE_NAME,
    generated_dataset_root,
    zenodo_dataset_root,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

PRESET_PIPELINE = "pipeline"
PRESET_STAGE2 = "stage2"
PRESET_STAGE3 = "stage3"
PRESET_MODELS = "models"
PRESET_ALL = "all"

PRESET_TARGETS: dict[str, tuple[str, ...]] = {
    PRESET_PIPELINE: (
        "dataset",
        "cache",
        "runs",
        "cfd",
        "logs",
    ),
    PRESET_MODELS: ("runs",),
    PRESET_STAGE2: (
        "dataset-meshgraphnets",
        "cache-meshgraphnets",
        "cfd-stage2",
        "runs",
    ),
    PRESET_STAGE3: (
        "dataset-cfdbench",
        "cache-cfdbench",
        "cfd-stage3",
        "runs",
    ),
    PRESET_ALL: (
        "dataset",
        "cache",
        "runs",
        "cfd",
        "logs",
        "hydra",
        "test-runs",
    ),
}

TARGET_HELP: dict[str, str] = {
    "dataset": "Entire dataset/ tree (zenodo_data, generated_data, legacy layout)",
    "dataset-zenodo": "dataset/zenodo_data only",
    "dataset-generated": "dataset/generated_data only",
    "dataset-meshgraphnets": "dataset/meshgraphnets_data only",
    "dataset-cfdbench": "dataset/cfdbench_data only",
    "cache": ".cache/ (Zenodo/LBM upstream HDF5)",
    "cache-meshgraphnets": ".cache/meshgraphnets_cylinder only",
    "cache-cfdbench": ".cache/cfdbench only",
    "runs": "results/runs/ (baselines, CNN, ConvLSTM, FNO checkpoints)",
    "cfd": "results/cfd_validation/ (Stage 1 + Stage 2 validation figures)",
    "cfd-stage2": "results/cfd_validation/stage2_meshgraphnets only",
    "cfd-stage3": "results/cfd_validation/stage3_cfdbench only",
    "logs": "log/ (application log files)",
    "hydra": "outputs/, multirun/, .hydra/ (local Hydra outputs)",
    "test-runs": ".local_test_runs/ (pytest scratch)",
}


def _repo_relative(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved == root:
        raise ValueError(f"refusing to delete repository root: {root}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {resolved}") from exc
    return resolved


def artifact_paths(repo_root: Path, target: str) -> list[Path]:
    """Map a target name to absolute paths (may not exist)."""
    root = repo_root.resolve()
    if target == "dataset":
        return [root / DATASET_BASE_NAME]
    if target == "dataset-zenodo":
        return [zenodo_dataset_root(root)]
    if target == "dataset-generated":
        return [generated_dataset_root(root)]
    if target == "dataset-meshgraphnets":
        return [root / "dataset" / "meshgraphnets_data"]
    if target == "dataset-cfdbench":
        return [root / "dataset" / "cfdbench_data"]
    if target == "cache":
        return [root / ".cache"]
    if target == "cache-meshgraphnets":
        return [root / ".cache" / "meshgraphnets_cylinder"]
    if target == "cache-cfdbench":
        return [root / ".cache" / "cfdbench"]
    if target == "runs":
        return [root / "results" / "runs"]
    if target == "cfd":
        return [root / "results" / "cfd_validation"]
    if target == "cfd-stage2":
        return [root / "results" / "cfd_validation" / "stage2_meshgraphnets"]
    if target == "cfd-stage3":
        return [root / "results" / "cfd_validation" / "stage3_cfdbench"]
    if target == "logs":
        return [root / "log"]
    if target == "hydra":
        return [root / "outputs", root / "multirun", root / ".hydra"]
    if target == "test-runs":
        return [root / ".local_test_runs"]
    raise KeyError(target)


def run_subdirs(repo_root: Path, run_ids: Iterable[str]) -> list[Path]:
    runs_root = repo_root / "results" / "runs"
    return [runs_root / run_id for run_id in run_ids]


def expand_selection(
    repo_root: Path,
    *,
    targets: list[str],
    run_ids: list[str],
    preset: str | None,
) -> list[Path]:
    """Build a de-duplicated list of paths to remove."""
    names: list[str] = []
    if preset is not None:
        names.extend(PRESET_TARGETS[preset])
    names.extend(targets)
    paths: list[Path] = []
    for name in names:
        for path in artifact_paths(repo_root, name):
            paths.append(_repo_relative(path, repo_root))
    for run_dir in run_subdirs(repo_root, run_ids):
        paths.append(_repo_relative(run_dir, repo_root))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MiB"
    return f"{num_bytes / 1024**3:.2f} GiB"


def dir_size(path: Path) -> int:
    """Total file size under *path* (0 if missing)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def list_artifacts(repo_root: Path) -> None:
    """Log known artifact locations and whether they exist."""
    log_handler.info("Repository: %s", repo_root)
    all_names = sorted(TARGET_HELP)
    for name in all_names:
        for path in artifact_paths(repo_root, name):
            exists = path.is_dir() or path.is_file()
            size = _format_size(dir_size(path)) if exists else "—"
            state = "present" if exists else "missing"
            rel = path.relative_to(repo_root)
            log_handler.info("  [%s] %s (%s, %s)", name, rel, state, size)
    runs_root = repo_root / "results" / "runs"
    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir()):
            if child.is_dir():
                log_handler.info(
                    "  [run] %s (%s)",
                    child.relative_to(repo_root),
                    _format_size(dir_size(child)),
                )


def _is_repo_log_dir(path: Path) -> bool:
    """True when *path* is the repo's ``log/`` tree (file handler may be open)."""
    try:
        return path.resolve() == (REPO_ROOT / "log").resolve()
    except OSError:
        return False


def _delete_one_path(path: Path) -> None:
    """Remove a file or directory; release log handlers before deleting ``log/``."""
    if path.is_dir():
        if _is_repo_log_dir(path):
            shutdown_logger()
        shutil.rmtree(path)
    else:
        path.unlink()


def delete_paths(paths: list[Path], *, dry_run: bool) -> list[Path]:
    """Remove directories or files; return paths actually removed."""
    removed: list[Path] = []
    # Delete log/ last so earlier messages still go to the file handler on Windows.
    ordered = sorted(paths, key=lambda p: 1 if _is_repo_log_dir(p) else 0)
    for path in ordered:
        if not path.exists():
            log_handler.info("[skip] not found: %s", path)
            continue
        if dry_run:
            size = _format_size(dir_size(path))
            log_handler.info("[dry-run] would delete: %s (%s)", path, size)
            removed.append(path)
            continue
        logging_active = bool(log_handler.handlers)
        _delete_one_path(path)
        if logging_active and log_handler.handlers:
            log_handler.info("[deleted] %s", path)
        else:
            print(f"[deleted] {path}", flush=True)
        removed.append(path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete local FFAO ML pipeline artifacts (datasets, runs, cache).",
    )
    target_names = sorted(TARGET_HELP)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show artifact locations and sizes (no deletions).",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_TARGETS),
        help=(
            "pipeline: full dataset/ + .cache/ + all runs + all cfd_validation "
            "(Stage 1 and Stage 2) + logs; "
            "stage2: mesh dataset + mesh cache + cfd-stage2 + all runs; "
            "models: runs only; all: pipeline+hydra+test-runs"
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=target_names,
        dest="targets",
        metavar="TARGET",
        help="Repeatable; see --list for names.",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="run_ids",
        metavar="RUN_ID",
        help="Delete results/runs/<RUN_ID> (e.g. stage1_cnn).",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print what would be deleted.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Do not ask for confirmation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = REPO_ROOT

    if args.list:
        list_artifacts(repo_root)
        return 0

    targets = list(args.targets or [])
    run_ids = list(args.run_ids or [])
    if args.preset is None and not targets and not run_ids:
        parser.error(
            "Specify --preset, --target, and/or --run, or use --list. "
            "Example: --preset pipeline --dry-run"
        )

    paths = expand_selection(
        repo_root,
        targets=targets,
        run_ids=run_ids,
        preset=args.preset,
    )
    if not paths:
        log_handler.info("Nothing to delete.")
        return 0

    log_handler.info("Paths to process (%s):", "dry-run" if args.dry_run else "delete")
    for path in paths:
        rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
        log_handler.info("  - %s", rel)

    if not args.dry_run and not args.yes:
        answer = input("Delete these paths? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            log_handler.info("Cancelled.")
            return 1

    delete_paths(paths, dry_run=args.dry_run)
    if args.dry_run:
        log_handler.info("Dry run complete. Re-run with --yes to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
