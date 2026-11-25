#!/usr/bin/env python3
"""
#############################################################################
### Inspect one CFDBench cylinder case
###
### @file inspect_cfdbench_sample.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Print velocity array shapes, grid size, and ``case.json`` Reynolds hints for one
interpolated cylinder case (local download only — not used in CI).

Example::

    python scripts/inspect_cfdbench_sample.py data/cylinder/prop/case0000
    python scripts/inspect_cfdbench_sample.py .cache/cfdbench/data
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.data.sources.cfdbench_probe import (
    describe_cfdbench_case,
    find_first_cylinder_case,
    is_cfdbench_case_dir,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize one CFDBench cylinder case (u.npy / v.npy / case.json).",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data",
        help="Case directory or CFDBench data root (default: ./data)",
    )
    args = parser.parse_args(argv)
    root = Path(args.path)
    if not root.exists():
        log_handler.error("Path does not exist: %s", root)
        return 1
    try:
        case_dir = (
            root if is_cfdbench_case_dir(root) else find_first_cylinder_case(root)
        )
    except FileNotFoundError as exc:
        log_handler.error("%s", exc)
        return 1
    log_handler.info("%s", describe_cfdbench_case(case_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
