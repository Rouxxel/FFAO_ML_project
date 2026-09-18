"""
#############################################################################
### Build stage1_mini.h5 fixture
###
### @file build_stage1_mini.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Regenerate ``tests/fixtures/stage1_mini.h5`` for offline import tests.
"""

# Native imports
from pathlib import Path

# Third-party imports
import h5py
import numpy as np

"""CONSTANTS-----------------------------------------------------------"""
OUTPUT = Path(__file__).resolve().parent / "stage1_mini.h5"

"""BUILD-----------------------------------------------------------"""


def main() -> None:
    """Write the miniature HDF5 upstream file."""
    n_time, ny, nx = 6, 10, 12
    rng = np.random.default_rng(7)
    fields = rng.standard_normal((n_time, 3, ny, nx), dtype=np.float32) * 0.1
    fields[:, 0] += 1.0
    x = np.linspace(0.0, 2.2, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.8, ny, dtype=np.float32)
    with h5py.File(OUTPUT, "w") as handle:
        handle.create_dataset("fields", data=fields, compression="gzip")
        handle.create_dataset("grid_x", data=x)
        handle.create_dataset("grid_y", data=y)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
