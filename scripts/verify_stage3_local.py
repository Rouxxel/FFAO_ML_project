#!/usr/bin/env python3
"""
#############################################################################
### Stage 3 local release gate (imported CFDBench + optional trained run)
###
### @file verify_stage3_local.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Checks that imported data covers ``test_re`` from ``dataset=stage3_cfdbench`` and,
when ``results/runs/stage3_cnn_re/`` exists, that ``re_generalization_metrics.json``
includes extrapolation (PRD experiment 3) entries.

Not run in CI (requires local CFDBench import). See documentation/STAGE3_RELEASE.md.
"""

# Native imports
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler, shutdown_logger
from ffaoml.config import config_dir
from ffaoml.data.io import simulation_store_path
from ffaoml.data.metadata import read_metadata_rows
from ffaoml.evaluation.re_generalization import RE_EVAL_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "stage3_cnn_re"


def _compose_cfg(repo_root: Path) -> object:
    with initialize_config_dir(
        config_dir=str(config_dir(repo_root)),
        version_base="1.3",
    ):
        return compose(
            config_name="config",
            overrides=["dataset=stage3_cfdbench", "model=cnn_re"],
        )


def _imported_re_values(dataset_root: Path) -> set[float]:
    values: set[float] = set()
    for row in read_metadata_rows(dataset_root):
        sim_dir = (dataset_root / str(row["path"])).resolve()
        if simulation_store_path(sim_dir).is_dir():
            values.add(float(row["re"]))
    return values


def check_dataset(repo_root: Path) -> list[str]:
    cfg = _compose_cfg(repo_root)
    dataset_root = Path(str(cfg.dataset.output_root))
    errors: list[str] = []
    if not dataset_root.is_dir():
        errors.append(f"Missing dataset root: {dataset_root}")
        return errors

    imported = _imported_re_values(dataset_root)
    if not imported:
        errors.append(
            f"No Zarr simulations under {dataset_root} (run import first)."
        )
        return errors

    test_re = [float(x) for x in cfg.dataset.test_re]
    tolerance = float(cfg.dataset.import.re_tolerance)
    missing_test: list[float] = []
    for target in test_re:
        if not any(abs(target - r) <= tolerance for r in imported):
            missing_test.append(target)
    if missing_test:
        errors.append(
            "Imported Re values do not cover test_re "
            f"{test_re} within tolerance {tolerance}: missing {missing_test}. "
            f"On disk: {sorted(imported)}"
        )
    else:
        log_handler.info(
            "Dataset OK: %d Re values on disk; test_re covered.",
            len(imported),
        )
    return errors


def check_eval(repo_root: Path, run_id: str) -> list[str]:
    run_dir = repo_root / "results" / "runs" / run_id
    errors: list[str] = []
    metrics_path = run_dir / RE_EVAL_FILENAME
    if not (run_dir / "model.pt").is_file():
        log_handler.warning(
            "Skipping eval check: no model at %s (train first).",
            run_dir / "model.pt",
        )
        return errors
    if not metrics_path.is_file():
        errors.append(f"Missing {metrics_path} (run evaluate_re_generalization).")
        return errors

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_re = payload.get("per_re") or {}
    test_entries = [
        entry
        for entry in per_re.values()
        if isinstance(entry, dict)
        and str(entry.get("regime", "")).lower() == "test"
    ]
    exp3 = payload.get("experiments", {}).get("exp3_extrapolation") or []
    if not test_entries and exp3:
        test_entries = [{"sim_id": sid} for sid in exp3]
    if not test_entries:
        errors.append(
            f"{metrics_path} has no per_re entries with regime=test (experiment 3)."
        )
    else:
        log_handler.info(
            "Eval OK: %d test-regime simulation(s) in %s.",
            len(test_entries),
            metrics_path.name,
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 local release gate.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="CNN run folder under results/runs/.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only verify imported dataset covers test_re.",
    )
    args = parser.parse_args(argv)

    errors = check_dataset(args.repo_root)
    if not args.skip_eval:
        errors.extend(check_eval(args.repo_root, args.run_id))

    if errors:
        for line in errors:
            log_handler.error("%s", line)
        return 1
    log_handler.info("Stage 3 local gate passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutdown_logger()
