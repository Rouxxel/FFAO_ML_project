#!/usr/bin/env python3
"""
#############################################################################
### Validate a training run checkpoint bundle
###
### @file validate_run_bundle.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Check required files and print ``bundle_manifest.json`` fields for reproducibility.

Example (after ``python main.py --run``; default CNN id is ``stage1_cnn``)::

    python scripts/validate_run_bundle.py --run-dir results/runs/stage1_cnn
"""

# Native imports
import argparse
import json
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.manifests import (
    CHECKPOINT_BUNDLE_MANIFEST_FILENAME,
    validate_checkpoint_bundle,
)

"""CLI-----------------------------------------------------------"""


def main() -> None:
    """
    Report missing bundle files and summarize provenance hashes.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Validate ML training run bundle.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory (results/runs/<run_id>).",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        log_handler.error(
            "Run directory does not exist: %s\n"
            "After clean_pipeline_artifacts (or a fresh clone), train first:\n"
            "  python main.py --run\n"
            "Default CNN output: results/runs/stage1_cnn/ "
            "(not configs/config.yaml - that is the Hydra template in git).",
            run_dir,
        )
        raise SystemExit(1)
    if not any(run_dir.iterdir()):
        log_handler.error(
            "Run directory is empty: %s\n"
            "Re-train or point --run-dir at an existing run under results/runs/.",
            run_dir,
        )
        raise SystemExit(1)

    missing = validate_checkpoint_bundle(run_dir)
    legacy_manifest = run_dir / "manifest.json"
    if "dataset_manifest.json" in missing and legacy_manifest.is_file():
        log_handler.warning(
            "Found legacy run file manifest.json; rename or copy to "
            "dataset_manifest.json, or re-run train with --force."
        )
    if missing:
        log_handler.warning("Missing required files: %s", ", ".join(missing))
    else:
        log_handler.info("Required checkpoint bundle files present.")

    bundle_path = run_dir / CHECKPOINT_BUNDLE_MANIFEST_FILENAME
    if bundle_path.is_file():
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        log_handler.info("%s", json.dumps(payload, indent=2))
    else:
        log_handler.warning(
            "No %s (re-train with current code).",
            CHECKPOINT_BUNDLE_MANIFEST_FILENAME,
        )

    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        log_handler.info(
            "training_summary: seed=%s config_hash=%s dataset_manifest_hash=%s",
            summary.get("seed"),
            summary.get("config_hash"),
            summary.get("dataset_manifest_hash"),
        )


if __name__ == "__main__":
    main()
