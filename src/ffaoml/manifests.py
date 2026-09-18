"""Dataset and training-run manifest schemas (reproducibility)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

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


@dataclass
class DatasetManifest:
    """Provenance for ``dataset/manifest.json`` (see documentation/DATA_SOURCES.md)."""

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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> DatasetManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class CheckpointBundleManifest:
    """Describes a training run directory under ``results/runs/<run_id>/``."""

    schema_version: int
    created_at: str
    git_commit: str | None = None
    dataset_manifest_hash: str | None = None
    config_hash: str | None = None
    files: dict[str, str] = field(
        default_factory=lambda: {name: name for name in CHECKPOINT_BUNDLE_FILES}
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointBundleManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> CheckpointBundleManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_short_commit(repo_root: str | Path | None = None) -> str | None:
    """Return short git SHA or None if git is unavailable."""
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
    """Stable SHA-256 hex digest (first 16 chars) of a resolved config."""
    if isinstance(cfg, DictConfig):
        payload = OmegaConf.to_yaml(cfg, resolve=True)
    else:
        payload = OmegaConf.to_yaml(OmegaConf.create(cfg))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def hash_file(path: str | Path) -> str:
    """SHA-256 hex digest (first 16 chars) of a file's contents."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def dataset_manifest_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DATASET_MANIFEST_FILENAME


def validate_checkpoint_bundle(run_dir: str | Path) -> list[str]:
    """Return paths of required checkpoint files that are missing."""
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
    """Factory for import/generation pipelines."""
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
