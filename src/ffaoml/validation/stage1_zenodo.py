"""
#############################################################################
### Stage 1 Zenodo validation figures
###
### @file stage1_zenodo.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Vorticity snapshots, shedding animation, and qualitative Strouhal check for the
Re≈100 Zenodo trajectory (PRD §13 items 2 and 6). Documents that Cd/Cl time
series are deferred when force data are unavailable (PRD §8).
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
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import open_simulation_from_config, resolve_simulation_dir
from ffaoml.physics.coefficients import dominant_frequency_from_signal, strouhal_number

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_OUTPUT = Path("results/cfd_validation/stage1_zenodo")
# Literature reference for qualitative comparison at Re≈100 (cylinder wake).
LITERATURE_ST_RE100 = 0.164

"""TYPES-----------------------------------------------------------"""


@dataclass
class Stage1ValidationResult:
    """Paths and metrics written by ``run_stage1_validation``."""

    output_dir: Path
    summary_md: Path
    metrics_json: Path
    vorticity_snapshots: Path
    velocity_snapshots: Path | None
    shedding_animation: Path
    shedding_spectrum: Path
    dominant_frequency_hz: float | None
    strouhal_number: float | None


"""SIGNALS-----------------------------------------------------------"""


def wake_vorticity_proxy(
    vorticity: np.ndarray,
    *,
    downstream_fraction: float = 0.5,
) -> np.ndarray:
    """
    Build a scalar shedding proxy from mean |ω| in the downstream wake.

    Parameters:
        vorticity (np.ndarray): Shape ``(n_time, ny, nx)``.
        downstream_fraction (float): Fraction of the domain width kept (from mid-x).

    Returns:
        np.ndarray: 1D signal vs time.
    """
    if vorticity.ndim != 3:
        raise ValueError("vorticity must be 3D (time, y, x)")
    nx = vorticity.shape[2]
    x_start = int(nx * (1.0 - downstream_fraction))
    wake = vorticity[:, :, x_start:]
    return np.mean(np.abs(wake), axis=(1, 2))


def estimate_shedding_st(
    signal: np.ndarray,
    dt: float,
    u_inlet: float,
    diameter: float,
    *,
    min_frequency: float = 0.01,
) -> tuple[float, float]:
    """
    Dominant frequency and Strouhal number from a wake proxy time series.

    Parameters:
        signal (np.ndarray): Uniformly sampled scalar series.
        dt (float): Time step.
        u_inlet (float): Reference speed U.
        diameter (float): Cylinder diameter D.
        min_frequency (float): Ignore DC/low drift below this frequency (Hz).

    Returns:
        tuple[float, float]: ``(frequency_hz, strouhal)``.
    """
    freq = dominant_frequency_from_signal(signal, dt, min_frequency=min_frequency)
    st = strouhal_number(freq, diameter, u_inlet)
    return freq, st


"""PLOTS-----------------------------------------------------------"""


def _pick_snapshot_indices(n_time: int, count: int = 4) -> list[int]:
    if n_time <= count:
        return list(range(n_time))
    return [int(round(i)) for i in np.linspace(0, n_time - 1, count)]


def plot_vorticity_snapshots(
    vorticity: np.ndarray,
    *,
    time: np.ndarray | None,
    output_path: str | Path,
    indices: list[int] | None = None,
) -> Path:
    """
    Save a multi-panel vorticity figure (PRD §13 item 2).

    Parameters:
        vorticity (np.ndarray): ``(n_time, ny, nx)``.
        time (np.ndarray | None): Optional time coordinate for titles.
        output_path (str | Path): PNG destination.
        indices (list[int] | None): Time indices to plot.

    Returns:
        Path: Written PNG path.
    """
    n_time = vorticity.shape[0]
    picks = indices if indices is not None else _pick_snapshot_indices(n_time)
    cols = min(4, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    vmax = np.percentile(np.abs(vorticity), 99)
    for ax, idx in zip(axes_flat, picks, strict=False):
        im = ax.imshow(
            vorticity[idx],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        label = f"t={idx}"
        if time is not None and idx < time.size:
            label = f"t={time[idx]:.3g}"
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(picks) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.75, label="ω")
    fig.suptitle("Vorticity snapshots (Stage 1 Zenodo)")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_velocity_snapshots(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    *,
    time: np.ndarray | None,
    output_path: str | Path,
    indices: list[int] | None = None,
) -> Path | None:
    """
    Save speed magnitude panels when velocity is non-trivial (PRD §13 item 1).

    Returns:
        Path | None: PNG path, or ``None`` if velocity is all zeros.
    """
    if not np.any(velocity_x) and not np.any(velocity_y):
        return None
    speed = np.sqrt(velocity_x**2 + velocity_y**2)
    n_time = speed.shape[0]
    picks = indices if indices is not None else _pick_snapshot_indices(n_time)
    cols = min(4, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    vmax = np.percentile(speed, 99)
    for ax, idx in zip(axes_flat, picks, strict=False):
        im = ax.imshow(
            speed[idx],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
        )
        label = f"t={idx}"
        if time is not None and idx < time.size:
            label = f"t={time[idx]:.3g}"
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(picks) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.75, label="|u|")
    fig.suptitle("Velocity magnitude snapshots")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_shedding_spectrum(
    signal: np.ndarray,
    dt: float,
    output_path: str | Path,
    *,
    dominant_hz: float | None = None,
) -> Path:
    """
    Plot the magnitude spectrum of the wake proxy (PRD §13 item 9 analogue).

    Parameters:
        signal (np.ndarray): Wake proxy vs time.
        dt (float): Sample interval.
        output_path (str | Path): PNG path.
        dominant_hz (float | None): Mark peak frequency if known.

    Returns:
        Path: Written PNG.
    """
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    mask = freqs > 0
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(freqs[mask], spectrum[mask])
    if dominant_hz is not None:
        ax.axvline(
            dominant_hz, color="crimson", ls="--", label=f"f≈{dominant_hz:.3f} Hz"
        )
        ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Wake vorticity proxy spectrum")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def save_vorticity_animation(
    vorticity: np.ndarray,
    output_path: str | Path,
    *,
    fps: int = 8,
    max_frames: int = 120,
) -> Path:
    """
    Write a looping GIF of vorticity (PRD §13 item 6).

    Parameters:
        vorticity (np.ndarray): ``(n_time, ny, nx)``.
        output_path (str | Path): ``.gif`` destination.
        fps (int): Frames per second.
        max_frames (int): Subsample long series for file size.

    Returns:
        Path: Written GIF path.
    """
    n_time = vorticity.shape[0]
    if n_time > max_frames:
        idx = np.linspace(0, n_time - 1, max_frames, dtype=int)
        frames = vorticity[idx]
    else:
        frames = vorticity
    vmax = np.percentile(np.abs(frames), 99)

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(
        frames[0],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
        aspect="auto",
    )
    ax.set_title("Vorticity shedding")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame: int) -> list:
        im.set_array(frames[frame])
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=frames.shape[0],
        interval=1000 // max(fps, 1),
        blit=True,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    anim.save(destination, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return destination


"""REPORT-----------------------------------------------------------"""


def write_summary_markdown(
    path: str | Path,
    *,
    metrics: dict[str, Any],
    simulation_dir: Path,
    forces_available: bool,
) -> Path:
    """
    Write human-readable validation notes for Stage 1.

    Parameters:
        path (str | Path): ``SUMMARY.md`` destination.
        metrics (dict[str, Any]): Numeric results and metadata.
        simulation_dir (Path): Imported simulation folder.
        forces_available (bool): Whether Cd/Cl can be computed.

    Returns:
        Path: Written markdown path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 1 Zenodo validation summary",
        "",
        f"Simulation directory: `{simulation_dir.as_posix()}`",
        "",
        "## Figures",
        "",
        "- `vorticity_snapshots.png`- PRD §13 item 2",
        "- `velocity_snapshots.png`- PRD §13 item 1 (when velocity present)",
        "- `shedding_animation.gif`- PRD §13 item 6",
        "- `shedding_spectrum.png`- spectral content of wake proxy",
        "",
        "## Shedding metrics (qualitative)",
        "",
        f"- Dominant frequency (wake |ω| proxy): "
        f"{metrics.get('dominant_frequency_hz', 'n/a')} Hz",
        f"- Strouhal number St = f D / U: {metrics.get('strouhal_number', 'n/a')}",
        f"- Literature reference St ≈ {LITERATURE_ST_RE100} at Re≈100 "
        "(order-of-magnitude check only).",
        "",
        "## Force coefficients (PRD §8- partial for Stage 1)",
        "",
    ]
    if forces_available:
        lines.append(
            "Force time series were detected; extend this script to plot Cd/Cl vs time."
        )
    else:
        lines.append(
            "Cd, Cl, and lift/drag spectra (PRD §13 items 7–9) are **not** produced "
            "for this import: Zenodo Stage 1 fields do not include integrated forces "
            "or reliable pressure on the cylinder. Use own-CFD or datasets with force "
            "outputs for coefficient validation."
        )
    lines.extend(
        [
            "",
            "## Machine-readable metrics",
            "",
            "See `metrics.json` in this folder.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


"""PIPELINE-----------------------------------------------------------"""


def run_stage1_validation(
    cfg: DictConfig,
    output_dir: str | Path | None = None,
    *,
    fps: int = 8,
    max_animation_frames: int = 120,
) -> Stage1ValidationResult:
    """
    Generate Stage 1 validation artifacts from an imported Zarr store.

    Parameters:
        cfg (DictConfig): Composed Hydra config (dataset on disk).
        output_dir (str | Path | None): Defaults to ``results/cfd_validation/...``.
        fps (int): Animation frame rate.
        max_animation_frames (int): Cap GIF length.

    Returns:
        Stage1ValidationResult: Output paths and shedding metrics.
    """
    out = Path(output_dir or DEFAULT_OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    sim_dir = resolve_simulation_dir(cfg)
    store = open_simulation_from_config(cfg)

    vorticity = np.asarray(store["vorticity"].values, dtype=np.float32)
    velocity_x = np.asarray(store["velocity_x"].values, dtype=np.float32)
    velocity_y = np.asarray(store["velocity_y"].values, dtype=np.float32)
    time = np.asarray(store["time"].values, dtype=np.float64)

    attrs = dict(store.attrs)
    dt = float(attrs.get("dt", 1.0))
    u_inlet = float(attrs.get("u_inlet", 1.0))
    diameter = float(attrs.get("diameter", 1.0))

    snap_path = plot_vorticity_snapshots(
        vorticity,
        time=time,
        output_path=out / "vorticity_snapshots.png",
    )
    vel_path = plot_velocity_snapshots(
        velocity_x,
        velocity_y,
        time=time,
        output_path=out / "velocity_snapshots.png",
    )
    anim_path = save_vorticity_animation(
        vorticity,
        out / "shedding_animation.gif",
        fps=fps,
        max_frames=max_animation_frames,
    )

    proxy = wake_vorticity_proxy(vorticity)
    freq_hz: float | None = None
    st: float | None = None
    try:
        freq_hz, st = estimate_shedding_st(
            proxy,
            dt,
            u_inlet,
            diameter,
            min_frequency=0.01,
        )
    except ValueError:
        pass

    spec_path = plot_shedding_spectrum(
        proxy,
        dt,
        out / "shedding_spectrum.png",
        dominant_hz=freq_hz,
    )

    forces_available = bool(attrs.get("force_drag") or attrs.get("force_lift"))
    metrics: dict[str, Any] = {
        "simulation_dir": sim_dir.as_posix(),
        "n_steps": int(vorticity.shape[0]),
        "ny": int(vorticity.shape[1]),
        "nx": int(vorticity.shape[2]),
        "dt": dt,
        "u_inlet": u_inlet,
        "diameter": diameter,
        "re": float(attrs.get("re", cfg.dataset.re)),
        "dominant_frequency_hz": freq_hz,
        "strouhal_number": st,
        "literature_st_re100": LITERATURE_ST_RE100,
        "forces_available": forces_available,
        "pressure_source": attrs.get("pressure_source", "unknown"),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary_path = write_summary_markdown(
        out / "SUMMARY.md",
        metrics=metrics,
        simulation_dir=sim_dir,
        forces_available=forces_available,
    )

    return Stage1ValidationResult(
        output_dir=out,
        summary_md=summary_path,
        metrics_json=metrics_path,
        vorticity_snapshots=snap_path,
        velocity_snapshots=vel_path,
        shedding_animation=anim_path,
        shedding_spectrum=spec_path,
        dominant_frequency_hz=freq_hz,
        strouhal_number=st,
    )
