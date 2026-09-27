#!/usr/bin/env python3
"""
#############################################################################
### Generate Stage 1 benchmark HDF5 (local fallback)
###
### @file generate_stage1_cylinder_h5.py
### @date 2026
#############################################################################

Thin CLI around ``ffaoml.data.sources.stage1_lbm``. Prefer the pipeline::

    python main.py --run --only generate
    python main.py --run --generate-data
"""

# Native imports
import argparse
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.data.sources.stage1_lbm import DEFAULT_H5_NAME, generate_stage1_h5

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".cache" / "zenodo_stage1" / DEFAULT_H5_NAME


def main() -> None:
    """Generate HDF5 upstream file for Stage 1 import."""
    parser = argparse.ArgumentParser(description="Generate Stage 1 cylinder HDF5.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fast", action="store_true", help="Short LBM smoke run.")
    parser.add_argument("--preview", type=Path, default=None, help="Vorticity PNG.")
    args = parser.parse_args()

    generate_stage1_h5(args.output, fast=args.fast)
    log_handler.info("Wrote %s", args.output)

    if args.preview is not None:
        import h5py
        import matplotlib.pyplot as plt

        with h5py.File(args.output, "r") as handle:
            omega = handle["fields"][-1, 2]
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(omega, cmap="RdBu_r", origin="lower")
        plt.colorbar(label="vorticity")
        plt.title("LBM Re=100- last snapshot")
        plt.tight_layout()
        plt.savefig(args.preview, dpi=120)
        plt.close()


if __name__ == "__main__":
    main()
