"""
#############################################################################
### Stage 1 import plan tests
###
### @file test_stage1_import_plan.py
### @date 2026
#############################################################################
"""

from pathlib import Path

from hydra import compose, initialize_config_dir

from ffaoml.config import config_dir
from ffaoml.data.sources.stage1_import import (
    Stage1ImportMode,
    build_import_plan,
)
from ffaoml.data.stage1_layout import GENERATED_DATA_DIR, ZENODO_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_ROOT = REPO_ROOT / ".local_test_runs" / "import_plan_no_cache"


def test_auto_plan_uses_generated_when_zenodo_empty() -> None:
    ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"dataset.import.cache_dir={ISOLATED_ROOT.as_posix()}/cache"],
        )
    plan = build_import_plan(
        cfg, ISOLATED_ROOT, mode=Stage1ImportMode.AUTO, local_upstream=None
    )
    assert plan.upstream_strategy == "zenodo_unavailable_lbm_fallback"
    assert plan.dataset_root.name == GENERATED_DATA_DIR


def test_generated_plan_targets_generated_data() -> None:
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    plan = build_import_plan(
        cfg, REPO_ROOT, mode=Stage1ImportMode.GENERATED, local_upstream=None
    )
    assert plan.dataset_root.name == GENERATED_DATA_DIR


def test_local_file_plan_targets_zenodo_data() -> None:
    ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    h5 = ISOLATED_ROOT / "data.h5"
    h5.write_bytes(b"x")
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    plan = build_import_plan(
        cfg, ISOLATED_ROOT, mode=Stage1ImportMode.AUTO, local_upstream=h5
    )
    assert plan.dataset_root.name == ZENODO_DATA_DIR
    assert plan.upstream_strategy == "local_file"
