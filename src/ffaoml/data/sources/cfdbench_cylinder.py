"""
#############################################################################
### Stage 3 — CFDBench cylinder import
###
### @file cfdbench_cylinder.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Download interpolated CFDBench cylinder cases (optional) and export Zarr simulations
under ``dataset/cfdbench_data/`` with ``metadata.csv`` and ``manifest.json``.
"""

# Native imports
from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np
from omegaconf import DictConfig

from ffaoml.app_logging import log_handler

# Project imports
from ffaoml.contracts import METADATA_CSV_COLUMNS, STAGE3_CFDBENCH_SOURCE_ID
from ffaoml.data.io import write_field_store
from ffaoml.data.metadata import (
    append_metadata_row,
    ensure_metadata_csv,
    read_metadata_rows,
)
from ffaoml.data.sources.cfdbench_probe import (
    is_cfdbench_case_dir,
    load_cfdbench_case,
)
from ffaoml.data.sources.zenodo_re100 import _time_series_vorticity
from ffaoml.manifests import (
    build_dataset_manifest,
    dataset_manifest_path,
    hash_config,
)

"""CONSTANTS-----------------------------------------------------------"""
DEFAULT_HF_REPO = "luoyining/CFDBench"
DEFAULT_SUBSETS: tuple[str, ...] = ("prop",)


@dataclass(frozen=True)
class CfdbenchImportResult:
    """Summary after a Stage 3 CFDBench import run."""

    dataset_root: Path
    manifest_path: Path
    cases_imported: int
    data_root: Path


@dataclass(frozen=True)
class CfdbenchCaseRef:
    """Pointer to one upstream case directory."""

    case_dir: Path
    subset: str
    case_name: str


"""PATHS-----------------------------------------------------------"""


def cache_dir_from_config(cfg: DictConfig) -> Path:
    import_cfg = cfg.dataset.get("import") or {}
    return Path(str(import_cfg.get("cache_dir", ".cache/cfdbench")))


def resolve_cfdbench_data_root(cache_dir: Path) -> Path | None:
    """
    Return the directory containing ``cylinder/`` if present under cache.

    Supports ``<cache>/data/cylinder`` (HF layout) or ``<cache>/cylinder``.
    """
    cache_dir = Path(cache_dir)
    for candidate in (cache_dir / "data", cache_dir):
        if (candidate / "cylinder").is_dir():
            return candidate
    return None


"""DOWNLOAD-----------------------------------------------------------"""


def download_cfdbench_interpolated(
    cache_dir: Path,
    *,
    repo_id: str = DEFAULT_HF_REPO,
    force: bool = False,
) -> Path:
    """
    Fetch interpolated CFDBench into ``cache_dir`` (resumable via Hugging Face).

    Returns:
        Path: Cache root (cases under ``data/cylinder`` or ``cylinder``).

    Raises:
        ImportError: When ``huggingface_hub`` is not installed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = resolve_cfdbench_data_root(cache_dir)
    if existing is not None and not force:
        log_handler.info(
            "CFDBench cache hit under %s (use force_download to re-fetch)",
            existing,
        )
        return cache_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download CFDBench; "
            "pip install huggingface_hub"
        ) from exc
    log_handler.info(
        "Downloading CFDBench interpolated subset → %s (repo=%s)",
        cache_dir,
        repo_id,
    )
    log_handler.info(
        "This is ~13 GB; progress is logged by huggingface_hub. "
        "Interrupt and re-run to resume."
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir),
        allow_patterns=["**/cylinder/**", "data/cylinder/**", "cylinder/**"],
    )
    log_handler.info("CFDBench download finished under %s", cache_dir)
    return cache_dir


"""DISCOVERY-----------------------------------------------------------"""


def iter_cylinder_cases(
    data_root: Path,
    *,
    subsets: tuple[str, ...] = DEFAULT_SUBSETS,
) -> Iterator[CfdbenchCaseRef]:
    """Yield case directories under ``data_root/cylinder/<subset>/case*``."""
    root = Path(data_root)
    cylinder = root / "cylinder" if (root / "cylinder").is_dir() else root
    if cylinder.name != "cylinder" and not (cylinder / "prop").is_dir():
        raise FileNotFoundError(f"No cylinder/ tree under {data_root}")
    for subset in subsets:
        subset_dir = cylinder / subset
        if not subset_dir.is_dir():
            log_handler.warning("Skipping missing subset %s", subset_dir)
            continue
        for case_dir in sorted(subset_dir.glob("case*")):
            if is_cfdbench_case_dir(case_dir):
                yield CfdbenchCaseRef(
                    case_dir=case_dir,
                    subset=subset,
                    case_name=case_dir.name,
                )


"""SPLITS-----------------------------------------------------------"""


def _re_matches_target(re: float, target: float, tolerance: float) -> bool:
    return abs(re - target) <= tolerance


def simulation_split_for_re(
    re: float,
    cfg: DictConfig,
    *,
    tolerance: float,
) -> str | None:
    """Map Reynolds number to ``train`` / ``val`` / ``test`` using config lists."""
    for split_name, key in (
        ("train", "train_re"),
        ("val", "val_re"),
        ("test", "test_re"),
    ):
        for target in cfg.dataset[key]:
            if _re_matches_target(re, float(target), tolerance):
                return split_name
    return None


def allowed_re_values(
    cfg: DictConfig,
    *,
    re_filter: list[float] | None,
) -> list[float]:
    if re_filter:
        return re_filter
    values: list[float] = []
    for key in ("train_re", "val_re", "test_re"):
        values.extend(float(v) for v in cfg.dataset[key])
    return values


def re_allowed_for_import(
    re: float,
    cfg: DictConfig,
    *,
    re_filter: list[float] | None,
    tolerance: float,
) -> bool:
    targets = allowed_re_values(cfg, re_filter=re_filter)
    return any(_re_matches_target(re, target, tolerance) for target in targets)


"""IMPORT-----------------------------------------------------------"""


def make_sim_id(re: float, subset: str, case_name: str) -> str:
    re_tag = int(round(re))
    return f"re_{re_tag:03d}_cfdbench_{subset}_{case_name}"


def _case_seed(case_name: str) -> int:
    digest = hashlib.sha256(case_name.encode()).hexdigest()
    return int(digest[:8], 16)


def import_cfdbench_case(
    cfg: DictConfig,
    case: CfdbenchCaseRef,
    *,
    split: str,
    re: float,
    dt: float,
) -> Path:
    """
    Convert one CFDBench case folder to ``fields.zarr`` and append ``metadata.csv``.

    Returns:
        Path: Simulation directory under ``dataset.output_root``.
    """
    loaded = load_cfdbench_case(case.case_dir)
    u = np.asarray(loaded["u"], dtype=np.float32)
    v = np.asarray(loaded["v"], dtype=np.float32)
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"u/v shape mismatch in {case.case_dir}: {u.shape} {v.shape}")
    n_time, ny, nx = u.shape
    params = loaded["params"]
    width = float(params.get("width", 2.0))
    height = float(params.get("height", 1.0))
    x = np.linspace(0.0, width, nx, dtype=np.float32)
    y = np.linspace(0.0, height, ny, dtype=np.float32)
    dx = width / max(nx - 1, 1)
    dy = height / max(ny - 1, 1)
    time = np.arange(n_time, dtype=np.float32) * float(dt)

    velocity_x = u
    velocity_y = v
    pressure = np.zeros_like(velocity_x, dtype=np.float32)
    vorticity = _time_series_vorticity(velocity_x, velocity_y, dx, dy).astype(
        np.float32
    )

    vel_in = float(params.get("vel_in", params.get("u_in", 1.0)))
    viscosity = float(params.get("viscosity", 0.01))
    density = float(params.get("density", 1.0))
    nu = viscosity / density if density else viscosity
    radius = float(params.get("radius", 0.05))
    diameter = 2.0 * radius

    dataset_root = Path(str(cfg.dataset.output_root))
    sim_id = make_sim_id(re, case.subset, case.case_name)
    simulation_dir = dataset_root / "simulations" / sim_id
    relative_path = f"simulations/{sim_id}"

    attrs = {
        "solver": "import:cfdbench_cylinder",
        "source_id": str(cfg.dataset.source_id),
        "source_url": str(cfg.dataset.source_url),
        "cfdbench_subset": case.subset,
        "cfdbench_case": case.case_name,
        "re": float(re),
        "u_inlet": vel_in,
        "nu": nu,
        "diameter": diameter,
        "dt": float(dt),
        "n_steps": int(n_time),
        "dx": float(dx),
        "dy": float(dy),
        "seed": _case_seed(case.case_name),
        "pressure_source": "zeros_placeholder",
        "vorticity_source": "computed_from_uv",
    }
    write_field_store(
        simulation_dir,
        time=time,
        y=y,
        x=x,
        fields={
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "pressure": pressure,
            "vorticity": vorticity,
        },
        attrs=attrs,
        time_chunk=min(16, n_time),
    )

    row = {
        "sim_id": sim_id,
        "re": float(re),
        "split": split,
        "path": relative_path,
        "u_inlet": vel_in,
        "nu": nu,
        "diameter": diameter,
        "nx": int(nx),
        "ny": int(ny),
        "dt": float(dt),
        "n_steps": int(n_time),
        "seed": _case_seed(case.case_name),
    }
    append_metadata_row(dataset_root, {col: row[col] for col in METADATA_CSV_COLUMNS})
    log_handler.info(
        "Imported %s → %s (Re≈%.2f, split=%s, T=%d, %dx%d)",
        case.case_name,
        sim_id,
        re,
        split,
        n_time,
        ny,
        nx,
    )
    return simulation_dir


def write_stage3_manifest(
    cfg: DictConfig,
    dataset_root: Path,
    *,
    repo_root: Path | None,
    notes: str,
) -> Path:
    manifest = build_dataset_manifest(
        stage=int(cfg.dataset.stage),
        source_id=str(cfg.dataset.source_id),
        source_url=str(cfg.dataset.source_url),
        config_hash=hash_config(cfg),
        repo_root=repo_root,
        notes=notes,
    )
    path = dataset_manifest_path(dataset_root)
    manifest.write_json(path)
    return path


def run_cfdbench_import(
    cfg: DictConfig,
    *,
    repo_root: Path | None = None,
    local_data_root: Path | None = None,
    force_download: bool = False,
    re_filter: list[float] | None = None,
    max_cases: int | None = None,
    subsets: tuple[str, ...] | None = None,
) -> CfdbenchImportResult:
    """
    Download (optional) and import CFDBench cylinder cases into ``dataset.output_root``.

    Parameters:
        cfg (DictConfig): ``dataset=stage3_cfdbench``.
        repo_root (Path | None): Git metadata for manifest.
        local_data_root (Path | None): Skip download; use this ``data/`` or cache root.
        force_download (bool): Re-fetch HF snapshot.
        re_filter (list[float] | None): Only import Reynolds near these values.
        max_cases (int | None): Cap number of new imports this run.
        subsets (tuple[str, ...] | None): CFDBench subsets (default ``prop``).
    """
    import_cfg = cfg.dataset.get("import") or {}
    cache_dir = cache_dir_from_config(cfg)
    dt = float(import_cfg.get("dt", 0.1))
    tolerance = float(import_cfg.get("re_tolerance", 2.0))
    repo_id = str(import_cfg.get("huggingface_repo", DEFAULT_HF_REPO))
    cap = max_cases
    if cap is None and import_cfg.get("max_cases") is not None:
        cap = int(import_cfg["max_cases"])
    subset_list = subsets or tuple(import_cfg.get("subsets", list(DEFAULT_SUBSETS)))

    dataset_root = Path(str(cfg.dataset.output_root))
    dataset_root.mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv(dataset_root)
    existing_ids = {row["sim_id"] for row in read_metadata_rows(dataset_root)}

    if local_data_root is not None:
        local = Path(local_data_root)
        data_root = resolve_cfdbench_data_root(local) or local
        if not (data_root / "cylinder").is_dir():
            raise FileNotFoundError(f"No CFDBench cylinder tree under {local}")
    else:
        download_cfdbench_interpolated(
            cache_dir,
            repo_id=repo_id,
            force=force_download,
        )
        resolved = resolve_cfdbench_data_root(cache_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"Download finished but no cylinder/ under {cache_dir}"
            )
        data_root = resolved

    log_handler.info(
        "CFDBench import → %s | data_root=%s | subsets=%s | max_cases=%s",
        dataset_root,
        data_root,
        ",".join(subset_list),
        cap if cap is not None else "ALL matching",
    )

    imported = 0
    for ref in iter_cylinder_cases(data_root, subsets=tuple(subset_list)):
        loaded = load_cfdbench_case(ref.case_dir)
        re_est = loaded["re_estimate"]
        if re_est is None:
            log_handler.warning("Skipping %s: cannot estimate Re", ref.case_dir)
            continue
        re_val = float(re_est)
        if not re_allowed_for_import(
            re_val,
            cfg,
            re_filter=re_filter,
            tolerance=tolerance,
        ):
            continue
        split = simulation_split_for_re(re_val, cfg, tolerance=tolerance)
        if split is None:
            log_handler.warning(
                "Skipping Re≈%.2f (%s): not in train/val/test lists",
                re_val,
                ref.case_name,
            )
            continue
        sim_id = make_sim_id(re_val, ref.subset, ref.case_name)
        if sim_id in existing_ids:
            log_handler.info("Cache hit (metadata): %s", sim_id)
            continue
        import_cfdbench_case(cfg, ref, split=split, re=re_val, dt=dt)
        existing_ids.add(sim_id)
        imported += 1
        if cap is not None and imported >= cap:
            log_handler.info("Reached max_cases=%d", cap)
            break

    note = (
        f"Imported {imported} CFDBench cylinder case(s); "
        f"source_id={STAGE3_CFDBENCH_SOURCE_ID}; subsets={','.join(subset_list)}."
    )
    manifest_path = write_stage3_manifest(
        cfg,
        dataset_root,
        repo_root=repo_root,
        notes=note,
    )
    log_handler.info(
        "CFDBench import done: %d new case(s); manifest %s",
        imported,
        manifest_path,
    )
    return CfdbenchImportResult(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        cases_imported=imported,
        data_root=data_root,
    )


def stage_upstream_case_layout(
    upstream_case_dir: Path,
    dest_data_root: Path,
    *,
    subset: str = "prop",
    case_name: str = "case0000",
) -> Path:
    """
    Copy a single-case fixture into ``dest_data_root/cylinder/<subset>/<case_name>/``.

    Used by tests and local smoke imports without HF download.
    """
    case_dest = dest_data_root / "cylinder" / subset / case_name
    if case_dest.exists():
        shutil.rmtree(case_dest)
    shutil.copytree(upstream_case_dir, case_dest)
    return dest_data_root
