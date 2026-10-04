"""
#############################################################################
### Stage 3 grid validation tests
###
### @file test_stage3_validation.py
### @date 2026
#############################################################################
"""

import json
import shutil
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ffaoml.config import config_dir
from ffaoml.validation.stage3_cfdbench import run_stage3_cfdbench_validation

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cfdbench_mini"
WORK = REPO_ROOT / ".local_test_runs" / "stage3_validation"


def test_stage3_validation_writes_artifacts() -> None:
    if not (FIXTURE_ROOT / "metadata.csv").is_file():
        import pytest

        pytest.skip("missing cfdbench_mini fixture")
    out_dir = WORK / "report"
    if out_dir.is_dir():
        shutil.rmtree(out_dir)

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
            ],
        )
    GlobalHydra.instance().clear()

    result = run_stage3_cfdbench_validation(
        cfg,
        output_dir=out_dir,
        fps=4,
        max_animation_frames=4,
    )
    assert result.summary_md.is_file()
    assert result.metrics_json.is_file()
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["n_simulations"] >= 1
    assert "reynolds_numbers" in metrics
    assert metrics["simulations"][0]["channels"]["vorticity"]["min"] is not None
    assert result.vorticity_animation is not None
    first_sid = metrics["simulations"][0]["sim_id"]
    assert "vorticity" in result.per_sim_figures[first_sid]
    assert "stage 3" in result.summary_md.read_text(encoding="utf-8").lower()
