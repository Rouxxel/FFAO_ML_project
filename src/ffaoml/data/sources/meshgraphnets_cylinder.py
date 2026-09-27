"""
#############################################################################
### Stage 2 - MeshGraphNets cylinder_flow import
###
### @file meshgraphnets_cylinder.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Download TFRecord shards (optional) and export trajectories under
``dataset/meshgraphnets_data/`` with ``metadata.csv`` and ``manifest.json``.

See ``documentation/DATA_SOURCES.md`` and ``configs/dataset/stage2_meshgraphnets.yaml``.
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

# Third-party imports
import numpy as np
from omegaconf import DictConfig

from ffaoml.app_logging import log_handler

# Project imports
from ffaoml.data.metadata import (
    append_metadata_row,
    ensure_metadata_csv,
    read_metadata_rows,
)
from ffaoml.data.sources.meshgraphnets_tfrecord import (
    iter_tfrecord_examples,
    load_meta,
)
from ffaoml.manifests import (
    build_dataset_manifest,
    dataset_manifest_path,
    hash_config,
)

"""CONSTANTS-----------------------------------------------------------"""
SPLIT_TO_TFRECORD_KEY = {"train": "train", "val": "val", "test": "test"}
TFRECORD_FILENAMES = {
    "train": "train.tfrecord",
    "val": "valid.tfrecord",
    "test": "test.tfrecord",
}

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class MeshImportResult:
    """Summary after a Stage 2 mesh import run."""

    dataset_root: Path
    manifest_path: Path
    trajectories_imported: int
    splits: tuple[str, ...]


"""DOWNLOAD-----------------------------------------------------------"""

_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_DOWNLOAD_LOG_EVERY_BYTES = 32 * 1024 * 1024


def _size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def tfrecord_cache_path(cache_dir: Path, split: str) -> Path:
    """Cached shard path for a split name (train/val/test)."""
    name = TFRECORD_FILENAMES[split]
    return cache_dir / name


def download_tfrecord(url: str, dest: Path, *, force: bool = False) -> Path:
    """
    Download a TFRecord shard to ``dest`` (skip if present unless ``force``).

    Returns:
        Path: Written file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        log_handler.info(
            "TFRecord cache hit: %s (%.1f MiB)",
            dest,
            _size_mib(dest),
        )
        return dest
    log_handler.info("Downloading TFRecord → %s", dest)
    log_handler.info("Source URL: %s", url)
    downloaded = 0
    last_logged = 0
    with urlopen(url) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_logged >= _DOWNLOAD_LOG_EVERY_BYTES:
                mib = downloaded / (1024 * 1024)
                log_handler.info("Downloaded %.1f MiB so far…", mib)
                last_logged = downloaded
    log_handler.info("Download complete: %s (%.1f MiB)", dest, _size_mib(dest))
    return dest


def resolve_tfrecord_path(
    cfg: DictConfig,
    split: str,
    *,
    force_download: bool = False,
) -> Path:
    """Return local TFRecord path, downloading from config URL when missing."""
    import_cfg = cfg.dataset["import"]
    cache_dir = Path(str(import_cfg.cache_dir))
    urls = import_cfg.tfrecord_urls
    key = SPLIT_TO_TFRECORD_KEY[split]
    url = str(urls[key])
    dest = tfrecord_cache_path(cache_dir, split)
    return download_tfrecord(url, dest, force=force_download)


"""EXPORT-----------------------------------------------------------"""


def trajectory_dir(dataset_root: Path, sim_id: str) -> Path:
    return dataset_root / "trajectories" / sim_id


def write_trajectory_arrays(
    dest_dir: Path,
    arrays: dict[str, np.ndarray],
    *,
    dt: float,
    sim_id: str,
) -> Path:
    """
    Write numpy trajectory arrays and ``meta.json`` (mesh I/O layout).

    Returns:
        Path: Trajectory directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.save(dest_dir / "mesh_pos.npy", arrays["mesh_pos"])
    np.save(dest_dir / "node_type.npy", arrays["node_type"])
    np.save(dest_dir / "cells.npy", arrays["cells"])
    np.save(dest_dir / "velocity.npy", arrays["velocity"])
    np.save(dest_dir / "pressure.npy", arrays["pressure"])
    n_steps = int(arrays["velocity"].shape[0])
    meta = {"dt": dt, "n_steps": n_steps, "sim_id": sim_id}
    (dest_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest_dir


def metadata_row_for_trajectory(
    *,
    sim_id: str,
    split: str,
    relative_path: str,
    arrays: dict[str, np.ndarray],
    dt: float,
    seed: int,
) -> dict[str, Any]:
    n_nodes = int(arrays["mesh_pos"].shape[0])
    n_steps = int(arrays["velocity"].shape[0])
    return {
        "sim_id": sim_id,
        "re": 0,
        "split": split,
        "path": relative_path,
        "u_inlet": 1.0,
        "nu": 0.01,
        "diameter": 1.0,
        "nx": n_nodes,
        "ny": 1,
        "dt": dt,
        "n_steps": n_steps,
        "seed": seed,
    }


def _existing_sim_ids(dataset_root: Path) -> set[str]:
    return {str(r["sim_id"]) for r in read_metadata_rows(dataset_root)}


def import_tfrecord_shard(
    cfg: DictConfig,
    tfrecord_path: Path,
    *,
    split: str,
    max_trajectories: int | None,
    repo_root: Path | None,
    meta: dict[str, Any],
    start_index: int = 0,
) -> int:
    """Import up to ``max_trajectories`` examples from one shard; return count added."""
    dataset_root = Path(str(cfg.dataset.output_root))
    dt = float(cfg.dataset["import"].dt)
    existing = _existing_sim_ids(dataset_root)
    imported = 0
    cap_label = str(max_trajectories) if max_trajectories is not None else "all"
    log_handler.info(
        "Parsing TFRecord split=%s from %s (max trajectories=%s; "
        "first record can take a minute while TensorFlow starts)",
        split,
        tfrecord_path,
        cap_label,
    )
    for offset, arrays in enumerate(
        iter_tfrecord_examples(tfrecord_path, meta, max_examples=max_trajectories)
    ):
        index = start_index + offset
        sim_id = f"cylinder_flow_{split}_{index:05d}"
        if sim_id in existing:
            continue
        rel_path = f"trajectories/{sim_id}"
        dest = trajectory_dir(dataset_root, sim_id)
        write_trajectory_arrays(dest, arrays, dt=dt, sim_id=sim_id)
        row = metadata_row_for_trajectory(
            sim_id=sim_id,
            split=split,
            relative_path=rel_path,
            arrays=arrays,
            dt=dt,
            seed=index,
        )
        append_metadata_row(dataset_root, row)
        existing.add(sim_id)
        imported += 1
        n_nodes = int(arrays["mesh_pos"].shape[0])
        n_steps = int(arrays["velocity"].shape[0])
        log_handler.info(
            "Imported %s (%d nodes, %d steps) [%d in this shard run]",
            sim_id,
            n_nodes,
            n_steps,
            imported,
        )
    log_handler.info(
        "Finished split=%s: %d new trajectory(ies) from %s",
        split,
        imported,
        tfrecord_path.name,
    )
    return imported


def write_stage2_manifest(
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


def run_meshgraphnets_import(
    cfg: DictConfig,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    max_trajectories_per_split: int | None = None,
    repo_root: Path | None = None,
    force_download: bool = False,
    meta_path: Path | None = None,
) -> MeshImportResult:
    """
    Download (if needed) and import MeshGraphNets ``cylinder_flow`` TFRecords.

    Parameters:
        cfg (DictConfig): Composed config with ``dataset=stage2_meshgraphnets``.
        splits (tuple[str, ...]): Subset of ``train``, ``val``, ``test``.
        max_trajectories_per_split (int | None): Cap per shard; ``None`` uses config.
        repo_root (Path | None): For manifest git metadata.
        force_download (bool): Re-fetch TFRecords even when cached.
        meta_path (Path | None): Optional ``meta.json`` beside cache.

    Returns:
        MeshImportResult: Import summary.
    """
    dataset_root = Path(str(cfg.dataset.output_root))
    dataset_root.mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv(dataset_root)

    cap = max_trajectories_per_split
    if cap is None:
        cap = cfg.dataset["import"].get("max_trajectories_per_split")
    if cap is not None:
        cap = int(cap)

    cap_msg = (
        str(cap) if cap is not None else "ALL per shard (full TFRecords; hours + GB)"
    )
    log_handler.info(
        "MeshGraphNets import → %s | splits=%s | max_trajectories_per_split=%s",
        dataset_root,
        ",".join(splits),
        cap_msg,
    )
    if cap is None:
        log_handler.warning(
            "No trajectory cap: downloading/parsing entire shards. "
            "For smoke runs use --max-trajectories N on main.py or the download script."
        )

    meta = load_meta(meta_path)
    total = 0
    for split in splits:
        log_handler.info("=== Mesh import: split %s ===", split)
        shard = resolve_tfrecord_path(cfg, split, force_download=force_download)
        total += import_tfrecord_shard(
            cfg,
            shard,
            split=split,
            max_trajectories=cap,
            repo_root=repo_root,
            meta=meta,
        )

    note = (
        "Imported MeshGraphNets cylinder_flow TFRecords; "
        f"{total} trajectory(ies) in this run; splits={','.join(splits)}."
    )
    manifest_path = write_stage2_manifest(
        cfg,
        dataset_root,
        repo_root=repo_root,
        notes=note,
    )
    log_handler.info(
        "MeshGraphNets import done: %d trajectory(ies); manifest %s",
        total,
        manifest_path,
    )
    return MeshImportResult(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        trajectories_imported=total,
        splits=splits,
    )


def import_trajectory_from_arrays(
    cfg: DictConfig,
    arrays: dict[str, np.ndarray],
    *,
    sim_id: str,
    split: str,
    repo_root: Path | None = None,
) -> Path:
    """
    Import one trajectory from in-memory arrays (tests / fixtures).

    Returns:
        Path: Trajectory directory.
    """
    dataset_root = Path(str(cfg.dataset.output_root))
    ensure_metadata_csv(dataset_root)
    dt = float(cfg.dataset["import"].dt)
    rel_path = f"trajectories/{sim_id}"
    dest = write_trajectory_arrays(
        trajectory_dir(dataset_root, sim_id),
        arrays,
        dt=dt,
        sim_id=sim_id,
    )
    row = metadata_row_for_trajectory(
        sim_id=sim_id,
        split=split,
        relative_path=rel_path,
        arrays=arrays,
        dt=dt,
        seed=0,
    )
    append_metadata_row(dataset_root, row)
    write_stage2_manifest(
        cfg,
        dataset_root,
        repo_root=repo_root,
        notes=f"Imported trajectory {sim_id} from in-memory arrays.",
    )
    return dest
