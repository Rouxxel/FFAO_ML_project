"""
#############################################################################
### Manifest schema tests
###
### @file test_manifests.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Round-trip JSON manifests, config/file hashing, and checkpoint bundle checks.
"""

# Native imports
from pathlib import Path

# Third-party imports
from omegaconf import OmegaConf

# Project imports
from ffaoml.manifests import (
    CHECKPOINT_BUNDLE_FILES,
    DatasetManifest,
    build_dataset_manifest,
    hash_config,
    hash_file,
    validate_checkpoint_bundle,
)

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""TESTS-----------------------------------------------------------"""


def test_dataset_manifest_roundtrip() -> None:
    path = REPO_ROOT / ".local_test_runs" / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_dataset_manifest(
        stage=1,
        source_id="zenodo_re100",
        source_url="https://zenodo.org/records/18669296",
        config_hash="abc123",
        git_commit="deadbeef",
    )
    manifest.write_json(path)
    loaded = DatasetManifest.read_json(path)
    assert loaded.stage == 1
    assert loaded.source_id == "zenodo_re100"
    assert loaded.schema_version == 1
    assert loaded.license_spdx == "CC-BY-4.0"


def test_hash_config_stable() -> None:
    cfg = OmegaConf.create({"seed": 42, "nested": {"a": 1}})
    assert hash_config(cfg) == hash_config(cfg)


def test_validate_checkpoint_bundle() -> None:
    run_dir = REPO_ROOT / ".local_test_runs" / "bundle_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_BUNDLE_FILES:
        (run_dir / name).unlink(missing_ok=True)
    tmp_path = run_dir
    assert validate_checkpoint_bundle(tmp_path) == list(CHECKPOINT_BUNDLE_FILES)
    for name in CHECKPOINT_BUNDLE_FILES:
        (tmp_path / name).write_bytes(b"x")
    assert validate_checkpoint_bundle(tmp_path) == []


def test_hash_file() -> None:
    p = REPO_ROOT / ".local_test_runs" / "hash_me.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"hello")
    assert len(hash_file(p)) == 16
