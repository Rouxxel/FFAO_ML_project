"""
#############################################################################
### Stage 3 CFDBench fixture and probe tests (no download)
###
### @file test_stage3_cfdbench_fixture.py
### @date 2026
#############################################################################
"""

# Native imports
from pathlib import Path

# Third-party imports
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS
from ffaoml.data.loading import load_simulation_tensor
from ffaoml.data.sources.cfdbench_probe import (
    describe_cfdbench_case,
    load_cfdbench_case,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cfdbench_mini"
UPSTREAM_CASE = REPO_ROOT / "tests" / "fixtures" / "cfdbench_upstream_case"


@pytest.fixture(scope="module")
def cfdbench_mini_cfg():
    if not (FIXTURE_ROOT / "metadata.csv").is_file():
        pytest.skip(f"missing fixture {FIXTURE_ROOT}; run build_cfdbench_mini.py")
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=stage3_cfdbench",
                f"dataset.output_root={FIXTURE_ROOT.as_posix()}",
                "dataset.train_re=[50,100]",
                "dataset.val_re=[125]",
                "dataset.test_re=[250]",
                "dataset.temporal_split.train=[0,4]",
                "dataset.temporal_split.val=[0,4]",
                "dataset.temporal_split.test=[0,4]",
            ],
        )
    yield cfg
    GlobalHydra.instance().clear()


def test_cfdbench_upstream_probe_shapes() -> None:
    if not UPSTREAM_CASE.is_dir():
        pytest.skip(f"missing {UPSTREAM_CASE}")
    loaded = load_cfdbench_case(UPSTREAM_CASE)
    u, v = loaded["u"], loaded["v"]
    assert u.shape == v.shape
    assert u.ndim == 3
    assert loaded["re_estimate"] is not None
    summary = describe_cfdbench_case(UPSTREAM_CASE)
    assert "velocity_x" in summary


def test_cfdbench_mini_load_simulation_tensor(cfdbench_mini_cfg) -> None:
    tensor = load_simulation_tensor(cfdbench_mini_cfg, "re_050_cfdbench", "train")
    assert tensor.dtype.name == "float32"
    n_time = 4
    ny, nx = 16, 16
    assert tensor.shape == (n_time, len(DEFAULT_FIELD_CHANNELS), ny, nx)
