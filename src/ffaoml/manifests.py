"""
#############################################################################
### Dataset and run manifests
###
### @file manifests.py
### @author Sebastian Russo
### @date 2026
#############################################################################

JSON schemas and helpers for dataset provenance and checkpoint bundles
(reproducibility; see ``documentation/REPRODUCIBILITY.md``).
"""

# Native imports
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig, OmegaConf

"""CONSTANTS-----------------------------------------------------------"""
DATASET_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_BUNDLE_SCHEMA_VERSION = 1

DATASET_MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"

# Required files under results/runs/<run_id>/ (ARCHITECTURE §7.3)
CHECKPOINT_BUNDLE_FILES: tuple[str, ...] = (
    "model.pt",
    "config.yaml",
    "preprocess_stats.json",
    "dataset_manifest.json",
)

"""TYPES-----------------------------------------------------------"""


@dataclass
class DatasetManifest:
    """Provenance record for ``dataset/manifest.json``."""

    schema_version: int
    stage: int
    source_id: str
    source_url: str
    created_at: str
    git_commit: str | None = None
    config_hash: str | None = None
    license_spdx: str = "CC-BY-4.0"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        """
        Build a manifest from parsed JSON, ignoring unknown keys.

        Parameters:
            data (dict[str, Any]): Parsed manifest object.

        Returns:
            DatasetManifest: Validated instance.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def write_json(self, path: str | Path) -> Path:
        """
        Write this manifest to disk.

        Parameters:
            path (str | Path): Destination ``manifest.json`` path.

        Returns:
            Path: Written file path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> DatasetManifest:
        """
        Load a manifest from disk.

        Parameters:
            path (str | Path): ``manifest.json`` path.

        Returns:
            DatasetManifest: Parsed instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class CheckpointBundleManifest:
    """Describes artifacts under ``results/runs/<run_id>/``."""

    schema_version: int
    created_at: str
    git_commit: str | None = None
    seed: int | None = None
    dataset_manifest_hash: str | None = None
    config_hash: str | None = None
    files: dict[str, str] = field(
        default_factory=lambda: {name: name for name in CHECKPOINT_BUNDLE_FILES}
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointBundleManifest:
        """
        Build a bundle manifest from parsed JSON, ignoring unknown keys.

        Parameters:
            data (dict[str, Any]): Parsed bundle object.

        Returns:
            CheckpointBundleManifest: Validated instance.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def write_json(self, path: str | Path) -> Path:
        """
        Write this bundle manifest to disk.

        Parameters:
            path (str | Path): Destination JSON path.

        Returns:
            Path: Written file path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> CheckpointBundleManifest:
        """
        Load a bundle manifest from disk.

        Parameters:
            path (str | Path): JSON file path.

        Returns:
            CheckpointBundleManifest: Parsed instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


"""HELPERS-----------------------------------------------------------"""


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 (second resolution)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_short_commit(repo_root: str | Path | None = None) -> str | None:
    """
    Return the short git SHA for the repository.

    Parameters:
        repo_root (str | Path | None): Git working tree root.

    Returns:
        str | None: Short commit hash, or ``None`` if git is unavailable.
    """
    root = Path(repo_root or Path.cwd())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def hash_config(cfg: DictConfig | dict[str, Any]) -> str:
    """
    Stable SHA-256 digest (first 16 hex chars) of a resolved config.

    Parameters:
        cfg (DictConfig | dict[str, Any]): Config to hash.

    Returns:
        str: Short hex digest.
    """
    if isinstance(cfg, DictConfig):
        payload = OmegaConf.to_yaml(cfg, resolve=True)
    else:
        payload = OmegaConf.to_yaml(OmegaConf.create(cfg))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def hash_file(path: str | Path) -> str:
    """
    SHA-256 digest (first 16 hex chars) of a file's contents.

    Parameters:
        path (str | Path): File to hash.

    Returns:
        str: Short hex digest.
    """
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def dataset_manifest_path(dataset_root: str | Path) -> Path:
    """Return ``<dataset_root>/manifest.json``."""
    return Path(dataset_root) / DATASET_MANIFEST_FILENAME


def write_checkpoint_bundle_manifest(
    run_dir: str | Path,
    cfg: DictConfig,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write ``bundle_manifest.json`` with config/dataset hashes and training seed.

    Parameters:
        run_dir (str | Path): ``results/runs/<run_id>/``.
        cfg (DictConfig): Resolved training configuration.
        repo_root (str | Path | None): Repo root for git metadata.

    Returns:
        Path: Written ``bundle_manifest.json``.
    """
    root = Path(run_dir)
    ds_path = root / "dataset_manifest.json"
    ds_hash = hash_file(ds_path) if ds_path.is_file() else None
    seed_val: int | None = None
    if "seed" in cfg:
        seed_val = int(cfg.seed)
    manifest = CheckpointBundleManifest(
        schema_version=CHECKPOINT_BUNDLE_SCHEMA_VERSION,
        created_at=utc_now_iso(),
        git_commit=git_short_commit(repo_root),
        seed=seed_val,
        dataset_manifest_hash=ds_hash,
        config_hash=hash_config(cfg),
    )
    return manifest.write_json(root / CHECKPOINT_BUNDLE_MANIFEST_FILENAME)


def validate_checkpoint_bundle(run_dir: str | Path) -> list[str]:
    """
    List required checkpoint files that are missing from a run directory.

    Parameters:
        run_dir (str | Path): ``results/runs/<run_id>/`` path.

    Returns:
        list[str]: Basenames of missing required files (empty if complete).
    """
    root = Path(run_dir)
    missing: list[str] = []
    for name in CHECKPOINT_BUNDLE_FILES:
        if not (root / name).is_file():
            missing.append(name)
    return missing


def build_dataset_manifest(
    *,
    stage: int,
    source_id: str,
    source_url: str,
    config_hash: str | None = None,
    git_commit: str | None = None,
    repo_root: str | Path | None = None,
    notes: str | None = None,
) -> DatasetManifest:
    """
    Factory for import or generation pipelines.

    Parameters:
        stage (int): Dataset stage (see ``documentation/DATA_SOURCES.md``).
        source_id (str): Stable source identifier.
        source_url (str): Download or citation URL.
        config_hash (str | None): Hash of import config.
        git_commit (str | None): Override git SHA; auto-detected when omitted.
        repo_root (str | Path | None): Repo root for git detection.
        notes (str | None): Free-form provenance notes.

    Returns:
        DatasetManifest: Ready to write under ``dataset/``.
    """
    commit = git_commit if git_commit is not None else git_short_commit(repo_root)
    return DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        stage=stage,
        source_id=source_id,
        source_url=source_url,
        created_at=utc_now_iso(),
        git_commit=commit,
        config_hash=config_hash,
        notes=notes,
    )
