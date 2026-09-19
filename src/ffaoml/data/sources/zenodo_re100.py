"""
#############################################################################
### Stage 1 — Zenodo Re=100 cylinder import
###
### @file zenodo_re100.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Download upstream flow data from Zenodo record **18669296** (DOI 10.5281/zenodo.18669296;
CC BY 4.0, Copyright 2026 Luca Addiucci) or a local cache,
parse HDF5 / NumPy / MATLAB archives, and export ``dataset/simulations/re_100_zenodo/``
plus ``metadata.csv`` and ``manifest.json``.

Upstream layout (Addiucci PO-CAE release): ``cylinder_re100_grid64_last100.h5`` with
``fields`` array ``(n_time, n_channels, ny, nx)``, ``grid_x``, ``grid_y``. Velocity
uses channels 0–1; vorticity is taken from channel 2 when present or computed from
``u``, ``v``. Pressure is set to zero when absent (recorded in Zarr attrs).

See ``documentation/DATA_SOURCES.md`` and ``configs/dataset/stage1_zenodo.yaml``.
"""

# Native imports
from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Project imports
from ffaoml.contracts import METADATA_CSV_COLUMNS
from ffaoml.data.io import write_field_store
from ffaoml.data.metadata import ensure_metadata_csv
from ffaoml.data.resample import downsample_spatial
from ffaoml.manifests import (
    build_dataset_manifest,
    dataset_manifest_path,
    hash_config,
)
from ffaoml.physics.navier_stokes import divergence_2d

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_ZENODO_RECORD = "18669296"
DEFAULT_SOURCE_URL = f"https://zenodo.org/records/{DEFAULT_ZENODO_RECORD}"
DEFAULT_UPSTREAM_FILENAME = "cylinder_re100_grid64_last100.h5"

DATA_FILE_SUFFIXES: tuple[str, ...] = (
    ".h5",
    ".hdf5",
    ".mat",
    ".npz",
    ".npy",
    ".zip",
    ".tar.gz",
    ".tgz",
)

ZENODO_API_RECORD = "https://zenodo.org/api/records/{record_id}"
ZENODO_FILE_CONTENT = (
    "https://zenodo.org/api/records/{record_id}/files/{filename}/content"
)

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class ParsedTrajectory:
    """Normalized time series on a structured grid."""

    velocity_x: np.ndarray
    velocity_y: np.ndarray
    pressure: np.ndarray
    vorticity: np.ndarray
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    attrs: dict[str, Any]


@dataclass(frozen=True)
class ImportResult:
    """Summary after a successful Stage 1 import."""

    dataset_root: Path
    simulation_dir: Path
    store_path: Path
    manifest_path: Path
    upstream_path: Path
    n_steps: int
    ny: int
    nx: int


"""ZENODO API-----------------------------------------------------------"""


def zenodo_record_url(record_id: str) -> str:
    """Public HTML URL for a Zenodo record."""
    return f"https://zenodo.org/records/{record_id}"


def fetch_zenodo_record(record_id: str) -> dict[str, Any]:
    """
    Load Zenodo record metadata from the REST API.

    Parameters:
        record_id (str): Zenodo record id (e.g. ``18669296``).

    Returns:
        dict[str, Any]: Parsed JSON document.

    Raises:
        RuntimeError: On network or HTTP errors.
    """
    url = ZENODO_API_RECORD.format(record_id=record_id)
    try:
        with urlopen(url, timeout=120) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to fetch Zenodo record {record_id}: {exc}") from exc
    return json.loads(payload)


def list_zenodo_data_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return Zenodo file entries that look like simulation data (not ``.py``).

    Parameters:
        record (dict[str, Any]): Zenodo API record object.

    Returns:
        list[dict[str, Any]]: File metadata dicts with ``key`` and ``size``.
    """
    files = record.get("files") or []
    selected: list[dict[str, Any]] = []
    for entry in files:
        name = str(entry.get("key", ""))
        lower = name.lower()
        if lower.endswith(".py") or lower in {"readme.md", "requirements.txt"}:
            continue
        if any(lower.endswith(suffix) for suffix in DATA_FILE_SUFFIXES):
            selected.append(entry)
    return selected


def pick_zenodo_data_file(
    record: dict[str, Any],
    preferred_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Choose one data file from a Zenodo record.

    Parameters:
        record (dict[str, Any]): Zenodo API record.
        preferred_name (str | None): Exact filename to prefer when present.

    Returns:
        dict[str, Any] | None: File entry or ``None`` if no data file is listed.
    """
    candidates = list_zenodo_data_files(record)
    if not candidates:
        return None
    if preferred_name:
        for entry in candidates:
            if entry.get("key") == preferred_name:
                return entry
    for suffix in DATA_FILE_SUFFIXES:
        for entry in candidates:
            if str(entry.get("key", "")).lower().endswith(suffix):
                return entry
    return candidates[0]


def download_url_to_path(url: str, destination: Path) -> Path:
    """
    Stream-download a URL to ``destination``.

    Parameters:
        url (str): HTTP(S) URL.
        destination (Path): Output file path (parent dirs created).

    Returns:
        Path: ``destination``.

    Raises:
        RuntimeError: On network or HTTP errors.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=300) as response:
            data = response.read()
        tmp.write_bytes(data)
        tmp.replace(destination)
    except (HTTPError, URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    return destination


def resolve_upstream_file(
    *,
    record_id: str,
    cache_dir: str | Path,
    preferred_name: str,
    override_url: str | None = None,
    local_path: str | Path | None = None,
) -> Path:
    """
    Return a local path to upstream data, downloading when needed.

    Resolution order:
        1. ``local_path`` if provided and exists
        2. Cached file under ``cache_dir``
        3. Data file attached to the Zenodo record
        4. ``override_url`` when the record has no data file yet

    Parameters:
        record_id (str): Zenodo record id.
        cache_dir (str | Path): Download cache directory.
        preferred_name (str): Expected basename (e.g. HDF5 name from README).
        override_url (str | None): Direct download URL fallback.
        local_path (str | Path | None): User-supplied file (skips network).

    Returns:
        Path: Readable upstream file.

    Raises:
        FileNotFoundError: When no source can be resolved.
        RuntimeError: On failed downloads.
    """
    if local_path is not None:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"local upstream file not found: {path}")
        return path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / preferred_name
    if cached.is_file():
        return cached

    record = fetch_zenodo_record(record_id)
    picked = pick_zenodo_data_file(record, preferred_name=preferred_name)
    if picked is not None:
        filename = str(picked["key"])
        url = ZENODO_FILE_CONTENT.format(record_id=record_id, filename=filename)
        dest = cache_dir / filename
        return download_url_to_path(url, dest)

    if override_url:
        return download_url_to_path(override_url, cached)

    raise FileNotFoundError(
        f"Zenodo record {record_id} has no data file matching {preferred_name!r}. "
        f"Place the file at {cached} or set dataset.import.upstream_data_url in "
        "configs/dataset/stage1_zenodo.yaml."
    )


"""PARSING-----------------------------------------------------------"""


def _vorticity_from_uv(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute ω = ∂v/∂x − ∂u/∂y for a single time slice ``(ny, nx)``."""
    du_dy = np.gradient(velocity_x, dy, axis=0)
    dv_dx = np.gradient(velocity_y, dx, axis=1)
    return dv_dx - du_dy


def _time_series_vorticity(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Vorticity for all times, shape ``(n_time, ny, nx)``."""
    frames = [
        _vorticity_from_uv(velocity_x[t], velocity_y[t], dx, dy)
        for t in range(velocity_x.shape[0])
    ]
    return np.stack(frames, axis=0)


def _extract_archive(archive_path: Path, work_dir: Path) -> Path:
    """
    Extract ``.zip`` / ``.tar.gz`` and return the first inner data file.

    Parameters:
        archive_path (Path): Archive on disk.
        work_dir (Path): Extraction directory.

    Returns:
        Path: Inner data file.

    Raises:
        FileNotFoundError: When no supported file is found inside the archive.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(work_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(work_dir)
    else:
        return archive_path

    for path in sorted(work_dir.rglob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(lower.endswith(suffix) for suffix in DATA_FILE_SUFFIXES):
            return path
    raise FileNotFoundError(f"no data file found inside archive {archive_path}")


def parse_upstream_file(
    path: str | Path,
    *,
    dt: float,
) -> ParsedTrajectory:
    """
    Parse an upstream file into CONTRACT channel arrays.

    Parameters:
        path (str | Path): ``.h5``, ``.npz``, ``.npy``, or ``.mat`` file.
        dt (float): Time step between frames (seconds or simulation units).

    Returns:
        ParsedTrajectory: Arrays with shape ``(n_time, ny, nx)``.

    Raises:
        ValueError: On unknown layout or unsupported extension.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    work_dir = source.parent / f".extract_{source.stem}"
    try:
        if source.name.lower().endswith((".zip", ".tar.gz", ".tgz")):
            source = _extract_archive(source, work_dir)

        lower = source.name.lower()
        if lower.endswith((".h5", ".hdf5")):
            return _parse_hdf5(source, dt=dt)
        if lower.endswith(".npz"):
            return _parse_npz(source, dt=dt)
        if lower.endswith(".npy"):
            return _parse_npy(source, dt=dt)
        if lower.endswith(".mat"):
            return _parse_mat(source, dt=dt)
        raise ValueError(f"unsupported upstream extension: {source.name}")
    finally:
        if work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_hdf5(path: Path, *, dt: float) -> ParsedTrajectory:
    import h5py

    with h5py.File(path, "r") as handle:
        if "fields" not in handle:
            raise ValueError(f"{path.name}: expected dataset 'fields'")
        fields = np.asarray(handle["fields"][:], dtype=np.float32)
        if fields.ndim != 4:
            raise ValueError(
                f"'fields' must be 4D (time, channel, y, x); got {fields.shape}"
            )
        gx = np.asarray(handle["grid_x"][:], dtype=np.float64).squeeze()
        gy = np.asarray(handle["grid_y"][:], dtype=np.float64).squeeze()

    n_time, n_channels, ny, nx = fields.shape
    velocity_x = fields[:, 0, :, :]
    velocity_y = fields[:, 1, :, :]
    if n_channels >= 3:
        vorticity = fields[:, 2, :, :]
        vorticity_source = "upstream_channel_2"
    else:
        dx = float(np.median(np.diff(gx))) if gx.size > 1 else 1.0
        dy = float(np.median(np.diff(gy))) if gy.size > 1 else 1.0
        vorticity = _time_series_vorticity(velocity_x, velocity_y, dx, dy)
        vorticity_source = "computed_from_uv"

    pressure = np.zeros_like(velocity_x, dtype=np.float32)
    time = np.arange(n_time, dtype=np.float64) * float(dt)
    attrs = {
        "upstream_format": "hdf5",
        "upstream_file": path.name,
        "pressure_source": "absent",
        "vorticity_source": vorticity_source,
        "n_channels_upstream": int(n_channels),
    }
    return ParsedTrajectory(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        vorticity=vorticity.astype(np.float32),
        x=gx.astype(np.float64),
        y=gy.astype(np.float64),
        time=time,
        attrs=attrs,
    )


def _parse_npz(path: Path, *, dt: float) -> ParsedTrajectory:
    archive = np.load(path)
    keys = set(archive.files)
    if "vorticity" in keys or "omega" in keys:
        omega = np.asarray(archive["vorticity" if "vorticity" in keys else "omega"])
        if omega.ndim == 3:
            vorticity = omega.astype(np.float32)
        else:
            raise ValueError("vorticity/omega must be 3D (time, y, x)")
    else:
        raise ValueError(f"{path.name}: npz must include 'vorticity' or 'omega'")

    n_time, ny, nx = vorticity.shape
    velocity_x = (
        np.asarray(archive["velocity_x"], dtype=np.float32)
        if "velocity_x" in keys
        else np.zeros_like(vorticity)
    )
    velocity_y = (
        np.asarray(archive["velocity_y"], dtype=np.float32)
        if "velocity_y" in keys
        else np.zeros_like(vorticity)
    )
    pressure = (
        np.asarray(archive["pressure"], dtype=np.float32)
        if "pressure" in keys
        else np.zeros_like(vorticity)
    )
    x = np.asarray(archive["x"], dtype=np.float64) if "x" in keys else np.arange(nx)
    y = np.asarray(archive["y"], dtype=np.float64) if "y" in keys else np.arange(ny)
    time = np.arange(n_time, dtype=np.float64) * float(dt)
    attrs = {"upstream_format": "npz", "upstream_file": path.name}
    return ParsedTrajectory(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        vorticity=vorticity,
        x=x,
        y=y,
        time=time,
        attrs=attrs,
    )


def _parse_npy(path: Path, *, dt: float) -> ParsedTrajectory:
    array = np.load(path)
    if array.ndim != 3:
        raise ValueError("npy vorticity stack must be 3D (time, y, x)")
    n_time, ny, nx = array.shape
    vorticity = array.astype(np.float32)
    velocity_x = np.zeros_like(vorticity)
    velocity_y = np.zeros_like(vorticity)
    pressure = np.zeros_like(vorticity)
    time = np.arange(n_time, dtype=np.float64) * float(dt)
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    attrs = {
        "upstream_format": "npy",
        "upstream_file": path.name,
        "velocity_source": "absent",
        "pressure_source": "absent",
    }
    return ParsedTrajectory(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        vorticity=vorticity,
        x=x,
        y=y,
        time=time,
        attrs=attrs,
    )


def _parse_mat(path: Path, *, dt: float) -> ParsedTrajectory:
    from scipy.io import loadmat

    data = loadmat(path)
    for key in ("omega", "vorticity", "w"):
        if key in data:
            vorticity = np.asarray(data[key], dtype=np.float32)
            break
    else:
        raise ValueError(f"{path.name}: mat file missing vorticity variable")

    if vorticity.ndim != 3:
        raise ValueError("vorticity in .mat must be 3D (time, y, x)")
    n_time, ny, nx = vorticity.shape
    velocity_x = np.zeros_like(vorticity)
    velocity_y = np.zeros_like(vorticity)
    pressure = np.zeros_like(vorticity)
    time = np.arange(n_time, dtype=np.float64) * float(dt)
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    attrs = {"upstream_format": "mat", "upstream_file": path.name}
    return ParsedTrajectory(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        vorticity=vorticity,
        x=x,
        y=y,
        time=time,
        attrs=attrs,
    )


"""IMPORT PIPELINE-----------------------------------------------------------"""


def import_stage1_from_config(
    cfg: DictConfig,
    *,
    local_upstream: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> ImportResult:
    """
    Run the full Stage 1 import using Hydra ``dataset`` settings.

    Parameters:
        cfg (DictConfig): Composed config (``dataset=stage1_zenodo``).
        local_upstream (str | Path | None): Skip download; use this file.
        repo_root (str | Path | None): Repository root for git hash in manifest.

    Returns:
        ImportResult: Paths and grid shape summary.
    """
    dataset = cfg.dataset
    record_id = str(dataset.get("zenodo_record", DEFAULT_ZENODO_RECORD))
    source_url = str(dataset.get("source_url", zenodo_record_url(record_id)))
    import_cfg = dataset.get("import") or {}
    cache_dir = Path(str(import_cfg.get("cache_dir", ".cache/zenodo_stage1")))
    preferred = str(import_cfg.get("upstream_data_filename", DEFAULT_UPSTREAM_FILENAME))
    override_url = import_cfg.get("upstream_data_url")
    override_url = str(override_url) if override_url else None

    dt = float(import_cfg.get("dt", 1.0))
    u_inlet = float(import_cfg.get("u_inlet", 1.0))
    nu = float(import_cfg.get("nu", 0.01))
    diameter = float(import_cfg.get("diameter", 1.0))
    re = float(dataset.re)
    seed = int(cfg.get("seed", 0))
    time_chunk = int(import_cfg.get("time_chunk", 16))

    dataset_root = Path(str(dataset.output_root))
    sim_id = str(dataset.simulation_id)
    simulation_dir = dataset_root / "simulations" / sim_id
    relative_path = f"simulations/{sim_id}"

    upstream_path = resolve_upstream_file(
        record_id=record_id,
        cache_dir=cache_dir,
        preferred_name=preferred,
        override_url=override_url,
        local_path=local_upstream,
    )
    parsed = parse_upstream_file(upstream_path, dt=dt)

    field_map = {
        "velocity_x": parsed.velocity_x,
        "velocity_y": parsed.velocity_y,
        "pressure": parsed.pressure,
        "vorticity": parsed.vorticity,
    }
    x_grid = parsed.x
    y_grid = parsed.y
    down_cfg = import_cfg.get("downsample") or {}
    if bool(down_cfg.get("enabled", False)):
        field_map, x_grid, y_grid, down_meta = downsample_spatial(
            field_map,
            x_grid,
            y_grid,
            target_nx=int(down_cfg.get("nx", 64)),
            target_ny=int(down_cfg.get("ny", 64)),
            order=int(down_cfg.get("order", 1)),
        )
    else:
        down_meta = {"downsampled": False}

    n_time, ny, nx = field_map["vorticity"].shape
    dx = float(np.median(np.diff(x_grid))) if x_grid.size > 1 else 1.0
    dy = float(np.median(np.diff(y_grid))) if y_grid.size > 1 else 1.0

    zarr_attrs = {
        "solver": "import:zenodo_re100",
        "source_id": str(dataset.source_id),
        "source_url": source_url,
        "re": re,
        "u_inlet": u_inlet,
        "nu": nu,
        "diameter": diameter,
        "dt": dt,
        "n_steps": n_time,
        "dx": dx,
        "dy": dy,
        "seed": seed,
        **parsed.attrs,
        **down_meta,
    }
    # Cheap sanity: divergence should be finite in fluid interior when u,v exist.
    if np.any(parsed.velocity_x):
        sample_div = divergence_2d(
            field_map["velocity_x"][0],
            field_map["velocity_y"][0],
            dx,
            dy,
        )
        zarr_attrs["divergence_interior_finite"] = bool(
            np.any(np.isfinite(sample_div[1:-1, 1:-1]))
        )

    store_path = write_field_store(
        simulation_dir,
        time=parsed.time,
        y=y_grid,
        x=x_grid,
        fields=field_map,
        attrs=zarr_attrs,
        time_chunk=time_chunk,
    )

    metadata_row = {
        "sim_id": sim_id,
        "re": re,
        "split": str(import_cfg.get("metadata_split", "full")),
        "path": relative_path,
        "u_inlet": u_inlet,
        "nu": nu,
        "diameter": diameter,
        "nx": int(nx),
        "ny": int(ny),
        "dt": dt,
        "n_steps": int(n_time),
        "seed": seed,
    }
    import csv

    csv_path = ensure_metadata_csv(dataset_root)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA_CSV_COLUMNS))
        writer.writeheader()
        writer.writerow({col: metadata_row[col] for col in METADATA_CSV_COLUMNS})

    doi = dataset.get("source_doi")
    doi_note = f" Upstream DOI: {doi}." if doi else ""
    manifest = build_dataset_manifest(
        stage=int(dataset.stage),
        source_id=str(dataset.source_id),
        source_url=source_url,
        config_hash=hash_config(cfg),
        repo_root=repo_root,
        notes=(
            f"Imported from {upstream_path.name}; "
            f"~{upstream_path.stat().st_size} bytes upstream.{doi_note} "
            "Temporal ML splits come from dataset.temporal_split in config."
        ),
    )
    manifest_path = dataset_manifest_path(dataset_root)
    manifest.write_json(manifest_path)

    return ImportResult(
        dataset_root=dataset_root,
        simulation_dir=simulation_dir,
        store_path=store_path,
        manifest_path=manifest_path,
        upstream_path=upstream_path,
        n_steps=n_time,
        ny=ny,
        nx=nx,
    )
