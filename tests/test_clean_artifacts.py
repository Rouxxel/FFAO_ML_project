"""
#############################################################################
### clean_pipeline_artifacts tests
###
### @file test_clean_artifacts.py
### @date 2026
#############################################################################
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "clean_pipeline_artifacts",
    REPO_ROOT / "scripts" / "clean_pipeline_artifacts.py",
)
_clean = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_clean)
delete_paths = _clean.delete_paths
expand_selection = _clean.expand_selection
SCRATCH = REPO_ROOT / ".local_test_runs" / "clean_artifacts"


def test_expand_preset_pipeline_includes_dataset_and_runs() -> None:
    paths = expand_selection(
        REPO_ROOT,
        targets=[],
        run_ids=[],
        preset="pipeline",
    )
    rel = {p.relative_to(REPO_ROOT).as_posix() for p in paths}
    assert "dataset" in rel
    assert "results/runs" in rel
    assert ".cache" in rel


def test_delete_paths_dry_run_leaves_tree() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    marker = SCRATCH / "keep_me" / "file.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("x", encoding="utf-8")
    delete_paths([SCRATCH / "keep_me"], dry_run=True)
    assert marker.is_file()


def test_run_id_target() -> None:
    paths = expand_selection(
        REPO_ROOT,
        targets=[],
        run_ids=["stage1_cnn"],
        preset=None,
    )
    assert paths == [(REPO_ROOT / "results" / "runs" / "stage1_cnn").resolve()]


def test_dataset_zenodo_target_in_preset_paths() -> None:
    paths = expand_selection(
        REPO_ROOT,
        targets=["dataset-zenodo"],
        run_ids=[],
        preset=None,
    )
    assert paths[0].name == "zenodo_data"
