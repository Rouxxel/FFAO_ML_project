"""
#############################################################################
### Build multi-Re stub dataset for tests
###
### @file build_multi_re_stub_dataset.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Generate several stub CFD simulations under one dataset root (Stage 3 layout).
"""

# Native imports
from pathlib import Path

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.cfd.stub_backend import StubSolver
from ffaoml.cfd.types import SimulationCase
from ffaoml.contracts import METADATA_CSV_COLUMNS
from ffaoml.data.metadata import append_metadata_row, ensure_metadata_csv

"""DEFAULTS-----------------------------------------------------------"""
DEFAULT_SIMS: tuple[tuple[int, str], ...] = (
    (50, "re_050"),
    (100, "re_100"),
    (125, "re_125"),
    (250, "re_250"),
)


def build_multi_re_stub_dataset(
    output_root: str | Path,
    *,
    n_steps: int = 8,
    nx: int = 10,
    ny: int = 8,
    sims: tuple[tuple[int, str], ...] = DEFAULT_SIMS,
) -> Path:
    """
    Write ``fields.zarr`` and ``metadata.csv`` rows for each Reynolds case.

    Parameters:
        output_root (str | Path): Dataset root (``dataset/`` layout).
        n_steps (int): Time steps per simulation.
        nx (int): Grid width.
        ny (int): Grid height.
        sims (tuple): ``(re, sim_id)`` pairs.

    Returns:
        Path: Dataset root.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv(root)
    solver = StubSolver()
    for re, sim_id in sims:
        case = SimulationCase(
            sim_id=sim_id,
            output_root=root,
            re=float(re),
            u_inlet=1.0,
            nu=0.01,
            diameter=1.0,
            nx=nx,
            ny=ny,
            dt=0.1,
            n_steps=n_steps,
            seed=int(re),
            split="train",
        )
        result = solver.run(case)
        row = {col: result.metadata[col] for col in METADATA_CSV_COLUMNS}
        append_metadata_row(root, row)
    return root


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "multi_re_stub_dataset"
    build_multi_re_stub_dataset(out)
    log_handler.info("Wrote stub multi-Re dataset under %s", out)
