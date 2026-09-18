"""
#############################################################################
### Dataset split tests
###
### @file test_splits.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Temporal disjointness (Stage 1) and Reynolds metadata filtering (Stage 3 config).
"""

# Native imports
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Project imports
from ffaoml.config import config_dir
from ffaoml.ml.splits import (
    filter_metadata_by_re,
    re_split_simulation_ids,
    temporal_splits_disjoint,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""TESTS-----------------------------------------------------------"""


def test_stage1_temporal_splits_are_disjoint() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    assert temporal_splits_disjoint(cfg) is True


def test_filter_metadata_by_re() -> None:
    rows = [
        {"sim_id": "re_050", "re": "50"},
        {"sim_id": "re_100", "re": "100"},
        {"sim_id": "re_200", "re": "200"},
    ]
    picked = filter_metadata_by_re(rows, [50, 100])
    assert [r["sim_id"] for r in picked] == ["re_050", "re_100"]


def test_re_split_simulation_ids_from_metadata() -> None:
    GlobalHydra.instance().clear()
    root = REPO_ROOT / ".local_test_runs" / "re_split_meta"
    root.mkdir(parents=True, exist_ok=True)
    csv = root / "metadata.csv"
    csv.write_text(
        "sim_id,re,split,path,u_inlet,nu,diameter,nx,ny,dt,n_steps,seed\n"
        "re_050,50,train,simulations/re_050,1,0.01,1,8,8,0.01,10,0\n"
        "re_100,100,train,simulations/re_100,1,0.01,1,8,8,0.01,10,0\n"
        "re_250,250,test,simulations/re_250,1,0.01,1,8,8,0.01,10,0\n",
        encoding="utf-8",
    )
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=splits",
                f"dataset.output_root={root.as_posix()}",
            ],
        )
    ids = re_split_simulation_ids(cfg)
    assert "re_050" in ids["train"]
    assert "re_100" in ids["train"]
    assert "re_250" in ids["test"]
