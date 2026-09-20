"""
#############################################################################
### Stage 2 mesh dataset validation figures
###
### @file stage2_meshgraphnets.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Mesh layout and velocity-magnitude snapshots for imported MeshGraphNets
trajectories (qualitative data check before GNN training).
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import (
    dataset_root_from_config,
    list_mesh_trajectories_for_split,
)
from ffaoml.data.mesh_io import load_mesh_trajectory_dir

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_OUTPUT = Path("results/cfd_validation/stage2_meshgraphnets")

"""TYPES-----------------------------------------------------------"""


@dataclass
class Stage2ValidationResult:
    """Paths written by ``run_stage2_mesh_validation``."""

    output_dir: Path
    summary_md: Path
    metrics_json: Path
    mesh_velocity_snapshots: Path
    velocity_animation: Path | None


"""PLOTS-----------------------------------------------------------"""


def _snapshot_indices(n_steps: int, count: int = 4) -> list[int]:
    if n_steps <= count:
        return list(range(n_steps))
    return [int(round(i)) for i in np.linspace(0, n_steps - 1, count)]


def velocity_magnitude(velocity: np.ndarray) -> np.ndarray:
    """``|v|`` from ``(N, 2)`` or ``(T, N, 2)``."""
    if velocity.ndim == 2:
        return np.linalg.norm(velocity, axis=-1)
    return np.linalg.norm(velocity, axis=-1)


def plot_mesh_velocity_snapshots(
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    velocity: np.ndarray,
    *,
    output_path: Path,
    time_indices: list[int] | None = None,
) -> Path:
    """Tripcolor panels of ``|v|`` at selected time indices."""
    n_steps = velocity.shape[0]
    indices = time_indices or _snapshot_indices(n_steps)
    tri = Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells)
    n_panels = len(indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.2), squeeze=False)
    vmax = float(velocity_magnitude(velocity).max()) or 1.0
    for ax, t in zip(axes[0], indices, strict=True):
        speed = velocity_magnitude(velocity[t])
        tc = ax.tripcolor(
            tri, speed, shading="flat", vmin=0.0, vmax=vmax, cmap="viridis"
        )
        ax.set_aspect("equal")
        ax.set_title(f"t={t}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(tc, ax=axes[0].tolist(), fraction=0.046, pad=0.04, label="|v|")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_velocity_animation(
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    velocity: np.ndarray,
    output_path: Path,
    *,
    fps: int = 8,
    max_frames: int = 60,
) -> Path:
    """GIF of ``|v|`` on the mesh (frame cap for file size)."""
    n_steps = velocity.shape[0]
    step = max(1, n_steps // max_frames)
    frames = list(range(0, n_steps, step))
    tri = Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells)
    vmax = float(velocity_magnitude(velocity).max()) or 1.0

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    speed0 = velocity_magnitude(velocity[frames[0]])
    collection = ax.tripcolor(
        tri, speed0, shading="flat", vmin=0.0, vmax=vmax, cmap="viridis"
    )
    ax.set_aspect("equal")
    title = ax.set_title(f"t={frames[0]}")

    def _update(frame_idx: int):
        t = frames[frame_idx]
        collection.set_array(velocity_magnitude(velocity[t]))
        title.set_text(f"t={t}")
        return collection, title

    anim = FuncAnimation(fig, _update, frames=len(frames), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def write_summary_markdown(
    path: Path,
    *,
    metrics: dict[str, Any],
    trajectory_dir: Path,
) -> Path:
    lines = [
        "# Stage 2 mesh validation (MeshGraphNets cylinder_flow)",
        "",
        f"Trajectory: `{trajectory_dir.as_posix()}`",
        "",
        "## Checks",
        "",
        "- Mesh triangulation renders in the snapshot view.",
        "- Velocity magnitude is finite over the stored time range.",
        "- Cd/Cl/St are **not** computed on mesh node data in this pass.",
        "",
        "## Dataset",
        "",
        f"- Train trajectories in catalog: {metrics.get('n_train_trajectories')}",
        f"- Nodes: {metrics.get('n_nodes')}, cells: {metrics.get('n_cells')}",
        f"- Steps: {metrics.get('n_steps')}, dt: {metrics.get('dt')}",
        "",
        "See `metrics.json` for numeric summaries.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


"""PIPELINE-----------------------------------------------------------"""


def _default_trajectory_dir(cfg: DictConfig) -> tuple[str, Path]:
    pairs = list_mesh_trajectories_for_split(cfg, "train")
    if not pairs:
        pairs = list_mesh_trajectories_for_split(cfg, "val")
    if not pairs:
        raise FileNotFoundError(
            "no mesh trajectories on disk; run download_stage2_meshgraphnets.py first"
        )
    return pairs[0]


def run_stage2_mesh_validation(
    cfg: DictConfig,
    output_dir: str | Path | None = None,
    *,
    sim_id: str | None = None,
    fps: int = 8,
    max_animation_frames: int = 60,
    write_animation: bool = True,
) -> Stage2ValidationResult:
    """
    Build mesh snapshot figures and metrics for one imported trajectory.

    Parameters:
        cfg (DictConfig): ``dataset=stage2_meshgraphnets`` with data on disk.
        output_dir (str | Path | None): Report folder.
        sim_id (str | None): Override trajectory id; default first train row.
        fps (int): GIF frame rate.
        max_animation_frames (int): Cap animation length.
        write_animation (bool): When False, skip GIF generation.

    Returns:
        Stage2ValidationResult: Output paths.
    """
    out = Path(output_dir or DEFAULT_OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    root = dataset_root_from_config(cfg)

    if sim_id is not None:
        traj_dir = root / "trajectories" / sim_id
        if not traj_dir.is_dir():
            raise FileNotFoundError(traj_dir)
        chosen_id = sim_id
    else:
        chosen_id, traj_dir = _default_trajectory_dir(cfg)

    traj = load_mesh_trajectory_dir(traj_dir)
    speed = velocity_magnitude(traj.velocity)
    snap_path = plot_mesh_velocity_snapshots(
        traj.mesh_pos,
        traj.cells,
        traj.velocity,
        output_path=out / "mesh_velocity_snapshots.png",
    )
    anim_path: Path | None = None
    if write_animation and traj.n_steps > 1:
        anim_path = save_velocity_animation(
            traj.mesh_pos,
            traj.cells,
            traj.velocity,
            out / "velocity_animation.gif",
            fps=fps,
            max_frames=max_animation_frames,
        )

    n_train = len(list_mesh_trajectories_for_split(cfg, "train"))
    metrics: dict[str, Any] = {
        "sim_id": chosen_id,
        "trajectory_dir": traj_dir.as_posix(),
        "dataset_root": root.as_posix(),
        "n_nodes": traj.n_nodes,
        "n_cells": int(traj.cells.shape[0]),
        "n_steps": traj.n_steps,
        "dt": traj.dt,
        "velocity_mag_min": float(speed.min()),
        "velocity_mag_max": float(speed.max()),
        "velocity_mag_mean": float(speed.mean()),
        "pressure_min": float(traj.pressure.min()),
        "pressure_max": float(traj.pressure.max()),
        "n_train_trajectories": n_train,
        "stage": int(cfg.dataset.stage),
        "source_id": str(cfg.dataset.source_id),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary_path = write_summary_markdown(
        out / "SUMMARY.md",
        metrics=metrics,
        trajectory_dir=traj_dir,
    )
    return Stage2ValidationResult(
        output_dir=out,
        summary_md=summary_path,
        metrics_json=metrics_path,
        mesh_velocity_snapshots=snap_path,
        velocity_animation=anim_path,
    )
