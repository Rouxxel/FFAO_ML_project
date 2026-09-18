"""CFD solver contract — stub backend."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.cfd.solver import case_from_config, get_solver
from ffaoml.config import config_dir
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS, METADATA_CSV_COLUMNS
from ffaoml.data.io import FIELD_STORE_NAME, open_field_store
from ffaoml.data.metadata import append_metadata_row, ensure_metadata_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stub_case() -> Path:
    root = REPO_ROOT / ".local_test_runs" / "cfd_stub_case"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_stub_solver_writes_zarr_and_metadata(stub_case: Path) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["simulation=stub_smoke"],
        )
    case = case_from_config(
        cfg,
        sim_id="re_100_stub",
        output_root=stub_case,
        split="train",
    )
    result = get_solver("stub").run(case)

    assert result.store_path.is_dir()
    assert (case.simulation_dir / FIELD_STORE_NAME).is_dir()

    ds = open_field_store(case.simulation_dir)
    for channel in DEFAULT_FIELD_CHANNELS:
        assert channel in ds
        assert ds[channel].shape == (case.n_steps, case.ny, case.nx)

    assert float(ds.attrs["re"]) == pytest.approx(100.0)
    assert "solid_mask" in ds

    ensure_metadata_csv(stub_case)
    append_metadata_row(stub_case, result.metadata)
    csv_path = stub_case / "metadata.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "re_100_stub" in text
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(METADATA_CSV_COLUMNS)


def test_fd_backend_not_implemented() -> None:
    from ffaoml.cfd.types import SimulationCase

    case = SimulationCase(
        sim_id="x",
        output_root=Path("."),
        re=100.0,
        u_inlet=1.0,
        nu=0.01,
        diameter=1.0,
        nx=8,
        ny=8,
        dt=0.01,
        n_steps=2,
    )
    with pytest.raises(NotImplementedError):
        get_solver("fd").run(case)
