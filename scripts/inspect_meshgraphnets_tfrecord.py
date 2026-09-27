#!/usr/bin/env python3
"""
#############################################################################
### Inspect MeshGraphNets TFRecord
###
### @file inspect_meshgraphnets_tfrecord.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Print feature keys and array shapes for the first example in a local ``.tfrecord``.

Example::

    python scripts/inspect_meshgraphnets_tfrecord.py path/to/train.tfrecord
    python scripts/inspect_meshgraphnets_tfrecord.py path/to/train.tfrecord \\
        --meta path/to/meta.json
"""

# Native imports
import argparse

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.data.sources.meshgraphnets_tfrecord import (
    describe_arrays,
    read_first_tfrecord_example,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the first MeshGraphNets TFRecord example.",
    )
    parser.add_argument(
        "tfrecord",
        help="Path to train.tfrecord / valid.tfrecord / test.tfrecord",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional meta.json (defaults to built-in cylinder_flow template).",
    )
    args = parser.parse_args(argv)
    try:
        arrays = read_first_tfrecord_example(args.tfrecord, meta_path=args.meta)
    except ImportError as exc:
        log_handler.error("%s", exc)
        return 1
    log_handler.info("First example in %s:", args.tfrecord)
    log_handler.info("%s", describe_arrays(arrays))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
