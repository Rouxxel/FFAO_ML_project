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

Example::

    python scripts/validate_run_bundle.py --run-dir results/runs/cnn_stage1
"""

# Native imports
import argparse
import json
from pathlib import Path

# Project imports
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
    run_dir = args.run_dir
    missing = validate_checkpoint_bundle(run_dir)
    if missing:
        print(f"Missing required files: {', '.join(missing)}")
    else:
        print("Required checkpoint bundle files present.")

    bundle_path = run_dir / CHECKPOINT_BUNDLE_MANIFEST_FILENAME
    if bundle_path.is_file():
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))
    else:
        print(f"No {CHECKPOINT_BUNDLE_MANIFEST_FILENAME} (re-train with current code).")

    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            "training_summary:",
            f"seed={summary.get('seed')}",
            f"config_hash={summary.get('config_hash')}",
            f"dataset_manifest_hash={summary.get('dataset_manifest_hash')}",
        )


if __name__ == "__main__":
    main()
