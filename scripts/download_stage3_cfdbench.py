#!/usr/bin/env python3
"""
#############################################################################
### Download and import Stage 3 CFDBench data
###
### @file download_stage3_cfdbench.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fetch the interpolated CFDBench cylinder subset (when missing) and export cases to
``dataset/cfdbench_data/``.

Example::

    python scripts/download_stage3_cfdbench.py --max-cases 1 --local-data-root data
    python scripts/download_stage3_cfdbench.py --re 100,200 --max-cases 5
"""

# Native imports
import argparse
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir
from ffaoml.data.sources.cfdbench_cylinder import run_cfdbench_import

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_re_list(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import CFDBench cylinder cases into dataset/cfdbench_data/.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap new imports this run (smoke tests).",
    )
    parser.add_argument(
        "--re",
        type=str,
        default=None,
        help="Comma-separated Reynolds numbers to import (near-match tolerance).",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help="Use an existing CFDBench data/ tree instead of downloading.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download HF snapshot even when cache exists.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        choices=["prop", "bc", "geo"],
        help="Cylinder subset(s) to import (repeatable). Default: prop.",
    )
    args = parser.parse_args()
    subsets = tuple(args.subsets) if args.subsets else None

    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["dataset=stage3_cfdbench"],
        )

    try:
        result = run_cfdbench_import(
            cfg,
            repo_root=REPO_ROOT,
            local_data_root=args.local_data_root,
            force_download=args.force_download,
            re_filter=_parse_re_list(args.re),
            max_cases=args.max_cases,
            subsets=subsets,
        )
    except ImportError as exc:
        log_handler.error("%s", exc)
        return 1
    except FileNotFoundError as exc:
        log_handler.error("%s", exc)
        return 1

    log_handler.info(
        "Imported %d case(s) into %s (manifest %s)",
        result.cases_imported,
        result.dataset_root,
        result.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
