"""
#############################################################################
### Stage 3 CFDBench / multi-Re grid validation figures
###
### @file stage3_cfdbench.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Per-Re velocity and vorticity snapshots (and optional GIF) for imported Stage 3
Zarr simulations — qualitative check before Re-conditioned CNN training.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib

matplotlib.use("Agg")
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import dataset_root_from_config
from ffaoml.data.io import open_field_store, simulation_store_path
from ffaoml.data.metadata import read_metadata_rows
from ffaoml.validation.stage1_zenodo import (
    plot_velocity_snapshots,
    plot_vorticity_snapshots,
    save_vorticity_animation,
)

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_OUTPUT = Path("results/cfd_validation/stage3_cfdbench")

"""TYPES-----------------------------------------------------------"""


@dataclass
class Stage3ValidationResult:
    """Paths written by ``run_stage3_cfdbench_validation``."""

    output_dir: Path
    summary_md: Path
    metrics_json: Path
    per_sim_figures: dict[str, dict[str, Path]]
    vorticity_animation: Path | None


"""HELPERS-----------------------------------------------------------"""


def _simulation_dir(dataset_root: Path, row: dict[str, Any]) -> Path:
    rel = str(row["path"])
    return (dataset_root / rel).resolve()


def _list_imported_simulations(cfg: DictConfig) -> list[dict[str, Any]]:
    root = dataset_root_from_config(cfg)
    rows: list[dict[str, Any]] = []
    for row in read_metadata_rows(root):
        sim_dir = _simulation_dir(root, row)
        if simulation_store_path(sim_dir).is_dir():
            rows.append(row)
    return rows


def _channel_stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
    }


def write_summary_markdown(
    path: Path,
    *,
    metrics: dict[str, Any],
    dataset_root: Path,
) -> Path:
    sims = metrics.get("simulations", [])
    re_list = sorted({float(s["re"]) for s in sims})
    lines = [
        "# Stage 3 CFDBench / multi-Re grid validation",
        "",
        f"Dataset root: `{dataset_root.as_posix()}`",
        "",
        "## Checks",
        "",
        "- Velocity magnitude and vorticity panels per simulation (see `figures/`).",
        "- Optional vorticity GIF for the first catalogued simulation.",
        "- Cd/Cl/St are **not** computed in this pass (grid QC only).",
        "",
        "## Catalog",
        "",
        f"- Simulations on disk: {metrics.get('n_simulations')}",
        f"- Reynolds values: {re_list}",
        f"- Source: `{metrics.get('source_id')}` (stage {metrics.get('stage')})",
        "",
        "See `metrics.json` for per-channel ranges, grid size, and `dt`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


"""PIPELINE-----------------------------------------------------------"""


def run_stage3_cfdbench_validation(
    cfg: DictConfig,
    output_dir: str | Path | None = None,
    *,
    sim_id: str | None = None,
    max_simulations: int | None = None,
    fps: int = 8,
    max_animation_frames: int = 60,
    write_animation: bool = True,
) -> Stage3ValidationResult:
    """
    Build per-Re snapshot figures and metrics for Stage 3 grid imports.

    Parameters:
        cfg (DictConfig): ``dataset=stage3_cfdbench`` or ``dataset=splits`` with data.
        output_dir (str | Path | None): Report folder.
        sim_id (str | None): Validate one simulation only.
        max_simulations (int | None): Cap how many rows to plot.
        fps (int): GIF frame rate.
        max_animation_frames (int): Cap animation length.
        write_animation (bool): When False, skip GIF generation.

    Returns:
        Stage3ValidationResult: Output paths.

    Raises:
        FileNotFoundError: When no importable simulations exist.
    """
    out = Path(output_dir or DEFAULT_OUTPUT)
    fig_root = out / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    root = dataset_root_from_config(cfg)

    rows = _list_imported_simulations(cfg)
    if sim_id is not None:
        rows = [r for r in rows if str(r["sim_id"]) == sim_id]
    if not rows:
        raise FileNotFoundError(
            f"no Stage 3 simulations under {root}; run download_stage3_cfdbench.py "
            "or populate stub/cfdbench data first"
        )
    if max_simulations is not None:
        rows = rows[: int(max_simulations)]

    per_sim_figures: dict[str, dict[str, Path]] = {}
    sim_metrics: list[dict[str, Any]] = []
    anim_path: Path | None = None

    for row in rows:
        sid = str(row["sim_id"])
        sim_dir = _simulation_dir(root, row)
        store = open_field_store(sim_dir)
        vx = np.asarray(store["velocity_x"].values, dtype=np.float32)
        vy = np.asarray(store["velocity_y"].values, dtype=np.float32)
        vort = np.asarray(store["vorticity"].values, dtype=np.float32)
        pressure = np.asarray(store["pressure"].values, dtype=np.float32)
        time_coord = np.asarray(store["time"].values, dtype=np.float64)
        dt = float(row.get("dt", store.attrs.get("dt", 0.1)))
        ny, nx = int(vx.shape[1]), int(vx.shape[2])

        sim_fig_dir = fig_root / sid
        sim_fig_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        vel_path = plot_velocity_snapshots(
            vx,
            vy,
            time=time_coord,
            output_path=sim_fig_dir / "velocity_magnitude.png",
        )
        if vel_path is not None:
            paths["velocity_magnitude"] = vel_path
        paths["vorticity"] = plot_vorticity_snapshots(
            vort,
            time=time_coord,
            output_path=sim_fig_dir / "vorticity.png",
        )
        per_sim_figures[sid] = paths

        if anim_path is None and write_animation and vort.shape[0] > 1:
            anim_path = save_vorticity_animation(
                vort,
                out / f"vorticity_{sid}.gif",
                fps=fps,
                max_frames=max_animation_frames,
            )

        sim_metrics.append(
            {
                "sim_id": sid,
                "re": float(row["re"]),
                "split": str(row["split"]),
                "nx": nx,
                "ny": ny,
                "n_steps": int(vx.shape[0]),
                "dt": dt,
                "channels": {
                    "velocity_x": _channel_stats(vx),
                    "velocity_y": _channel_stats(vy),
                    "pressure": _channel_stats(pressure),
                    "vorticity": _channel_stats(vort),
                },
            }
        )

    metrics: dict[str, Any] = {
        "stage": int(cfg.dataset.stage),
        "source_id": str(cfg.dataset.source_id),
        "dataset_root": root.as_posix(),
        "n_simulations": len(sim_metrics),
        "reynolds_numbers": sorted({m["re"] for m in sim_metrics}),
        "simulations": sim_metrics,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary_path = write_summary_markdown(
        out / "SUMMARY.md",
        metrics=metrics,
        dataset_root=root,
    )
    return Stage3ValidationResult(
        output_dir=out,
        summary_md=summary_path,
        metrics_json=metrics_path,
        per_sim_figures=per_sim_figures,
        vorticity_animation=anim_path,
    )
