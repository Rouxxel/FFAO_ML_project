"""
#############################################################################
### Build CFDBench mini fixtures (Stage 3 CI)
###
### @file build_cfdbench_mini.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Write ``tests/fixtures/cfdbench_mini/`` (project Zarr dataset) and
``tests/fixtures/cfdbench_upstream_case/`` (upstream u/v layout for probe tests).
"""

# Native imports
import json
from pathlib import Path

# Third-party imports
import numpy as np

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.cfd.stub_backend import StubSolver
from ffaoml.cfd.types import SimulationCase
from ffaoml.contracts import (
    CFDBENCH_CASE_PARAMS_FILE,
    CFDBENCH_UPSTREAM_VELOCITY_X,
    CFDBENCH_UPSTREAM_VELOCITY_Y,
    METADATA_CSV_COLUMNS,
    STAGE3_CFDBENCH_SOURCE_ID,
)
from ffaoml.data.metadata import append_metadata_row, ensure_metadata_csv

"""DEFAULTS-----------------------------------------------------------"""
FIXTURES = Path(__file__).resolve().parent
DATASET_ROOT = FIXTURES / "cfdbench_mini"
UPSTREAM_CASE = FIXTURES / "cfdbench_upstream_case"

# (re, sim_id, metadata split column)
MINI_SIMS: tuple[tuple[int, str, str], ...] = (
    (50, "re_050_cfdbench", "train"),
    (100, "re_100_cfdbench", "train"),
    (125, "re_125_cfdbench", "val"),
)

DEFAULT_N_STEPS = 8
DEFAULT_NX = 16
DEFAULT_NY = 16


def build_cfdbench_upstream_case(
    output_dir: Path,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    ny: int = DEFAULT_NY,
    nx: int = DEFAULT_NX,
    seed: int = 42,
) -> Path:
    """Write synthetic ``u.npy``, ``v.npy``, and ``case.json`` (CFDBench layout)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_steps, ny, nx), dtype=np.float64) * 0.05 + 1.0
    v = rng.standard_normal((n_steps, ny, nx), dtype=np.float64) * 0.05
    np.save(out / CFDBENCH_UPSTREAM_VELOCITY_X, u)
    np.save(out / CFDBENCH_UPSTREAM_VELOCITY_Y, v)
    params = {
        "vel_in": 1.0,
        "density": 1.0,
        "viscosity": 0.01,
        "radius": 0.05,
        "height": 1.0,
        "width": 2.0,
    }
    (out / CFDBENCH_CASE_PARAMS_FILE).write_text(
        json.dumps(params, indent=2),
        encoding="utf-8",
    )
    return out


def build_cfdbench_mini_dataset(
    output_root: Path,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    sims: tuple[tuple[int, str, str], ...] = MINI_SIMS,
) -> Path:
    """Write Stage-3-shaped Zarr simulations under ``output_root``."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv(root)
    solver = StubSolver()
    for re, sim_id, split in sims:
        case = SimulationCase(
            sim_id=sim_id,
            output_root=root,
            re=float(re),
            u_inlet=1.0,
            nu=0.01,
            diameter=0.1,
            nx=nx,
            ny=ny,
            dt=0.1,
            n_steps=n_steps,
            seed=int(re),
            split=split,
        )
        result = solver.run(case)
        row = {col: result.metadata[col] for col in METADATA_CSV_COLUMNS}
        append_metadata_row(root, row)
    manifest = {
        "schema_version": 1,
        "stage": 3,
        "source_id": STAGE3_CFDBENCH_SOURCE_ID,
        "note": "synthetic mini fixture for CI — not from CFDBench download",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


def main() -> None:
    build_cfdbench_upstream_case(UPSTREAM_CASE)
    build_cfdbench_mini_dataset(DATASET_ROOT)
    log_handler.info("Wrote %s and %s", UPSTREAM_CASE, DATASET_ROOT)


if __name__ == "__main__":
    main()
