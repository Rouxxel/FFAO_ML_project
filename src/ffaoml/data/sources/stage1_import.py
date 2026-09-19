"""
#############################################################################
### Stage 1 data acquisition (Zenodo → generate fallback)
###
### @file stage1_import.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Resolve upstream HDF5 (Zenodo, cache, local file, or LBM), import into
``dataset/zenodo_data`` or ``dataset/generated_data``.
"""

# Native imports
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Third-party imports
from omegaconf import DictConfig, OmegaConf

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.data.sources.stage1_lbm import DEFAULT_H5_NAME, generate_stage1_h5
from ffaoml.data.sources.zenodo_re100 import (
    ImportResult,
    fetch_zenodo_record,
    import_stage1_from_config,
    pick_zenodo_data_file,
    resolve_upstream_file,
)
from ffaoml.data.stage1_layout import generated_dataset_root, zenodo_dataset_root

"""TYPES-----------------------------------------------------------"""


class Stage1ImportMode(StrEnum):
    """How to obtain Stage 1 upstream data."""

    AUTO = "auto"
    ZENODO = "zenodo"
    GENERATED = "generated"


@dataclass(frozen=True)
class Stage1ImportPlan:
    """Dry-run / logging summary for the import step."""

    mode: Stage1ImportMode
    dataset_root: Path
    upstream_strategy: str
    zenodo_has_attachment: bool
    cache_h5_exists: bool


SOURCE_ID_GENERATED = "stage1_lbm_generated"
SOURCE_ID_ZENODO = "zenodo_re100"


def zenodo_has_data_attachment(record_id: str) -> bool:
    """True when the Zenodo API lists a downloadable data file."""
    try:
        record = fetch_zenodo_record(record_id)
    except RuntimeError:
        return False
    return pick_zenodo_data_file(record) is not None


def build_import_plan(
    cfg: DictConfig,
    repo_root: Path,
    *,
    mode: Stage1ImportMode,
    local_upstream: Path | None,
) -> Stage1ImportPlan:
    """Infer import target paths without downloading or simulating."""
    import_cfg = cfg.dataset.get("import") or {}
    cache_dir = Path(str(import_cfg.get("cache_dir", ".cache/zenodo_stage1")))
    preferred = str(import_cfg.get("upstream_data_filename", DEFAULT_H5_NAME))
    record_id = str(cfg.dataset.get("zenodo_record", "18669296"))
    cached = cache_dir if cache_dir.is_absolute() else repo_root / cache_dir
    cache_h5 = (cached / preferred).is_file()
    has_attachment = zenodo_has_data_attachment(record_id)

    if mode == Stage1ImportMode.GENERATED:
        root = generated_dataset_root(repo_root)
        strategy = "lbm_generate"
    elif local_upstream is not None:
        root = zenodo_dataset_root(repo_root)
        strategy = "local_file"
    elif mode == Stage1ImportMode.ZENODO:
        root = zenodo_dataset_root(repo_root)
        strategy = "zenodo_or_cache"
    elif has_attachment or cache_h5:
        root = zenodo_dataset_root(repo_root)
        strategy = "zenodo_or_cache"
    else:
        root = generated_dataset_root(repo_root)
        strategy = "zenodo_unavailable_lbm_fallback"

    return Stage1ImportPlan(
        mode=mode,
        dataset_root=root,
        upstream_strategy=strategy,
        zenodo_has_attachment=has_attachment,
        cache_h5_exists=cache_h5,
    )


def _import_with_root(
    cfg: DictConfig,
    *,
    dataset_root: Path,
    local_upstream: Path,
    repo_root: Path,
    source_id: str,
    provenance_note: str,
) -> ImportResult:
    """Import into *dataset_root* with manifest overrides."""
    cfg_import = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_import.dataset.output_root = dataset_root.as_posix()
    cfg_import.dataset.source_id = source_id
    return import_stage1_from_config(
        cfg_import,
        local_upstream=local_upstream,
        repo_root=repo_root,
        provenance_note=provenance_note,
    )


def _import_generated(
    cfg: DictConfig,
    repo_root: Path,
    *,
    local_upstream: Path | None,
    lbm_fast: bool,
) -> tuple[ImportResult, Path]:
    target_root = generated_dataset_root(repo_root)
    import_cfg = cfg.dataset.get("import") or {}
    cache_rel = Path(str(import_cfg.get("cache_dir", ".cache/zenodo_stage1")))
    cache_dir = cache_rel if cache_rel.is_absolute() else repo_root / cache_rel
    preferred = str(import_cfg.get("upstream_data_filename", DEFAULT_H5_NAME))
    if local_upstream is not None:
        upstream = Path(local_upstream)
    else:
        upstream = generate_stage1_h5(cache_dir / preferred, fast=lbm_fast)
    result = _import_with_root(
        cfg,
        dataset_root=target_root,
        local_upstream=upstream,
        repo_root=repo_root,
        source_id=SOURCE_ID_GENERATED,
        provenance_note=(
            "LBM fallback (ffaoml.data.sources.stage1_lbm); "
            f"dataset root: {target_root.name}/."
        ),
    )
    return result, target_root


def run_stage1_import(
    cfg: DictConfig,
    repo_root: Path,
    *,
    mode: Stage1ImportMode = Stage1ImportMode.AUTO,
    local_upstream: Path | None = None,
    lbm_fast: bool = False,
) -> tuple[ImportResult, Path]:
    """
    Acquire upstream data and import to ``dataset/zenodo_data`` or ``generated_data``.

    Returns:
        tuple: Import summary and the dataset root used for downstream phases.
    """
    plan = build_import_plan(cfg, repo_root, mode=mode, local_upstream=local_upstream)
    log_handler.info(
        "Stage 1 import: mode=%s strategy=%s → %s",
        plan.mode.value,
        plan.upstream_strategy,
        plan.dataset_root,
    )

    if mode == Stage1ImportMode.GENERATED or plan.upstream_strategy.startswith(
        "zenodo_unavailable"
    ):
        return _import_generated(
            cfg, repo_root, local_upstream=local_upstream, lbm_fast=lbm_fast
        )

    if local_upstream is not None:
        target_root = zenodo_dataset_root(repo_root)
        result = _import_with_root(
            cfg,
            dataset_root=target_root,
            local_upstream=Path(local_upstream),
            repo_root=repo_root,
            source_id=SOURCE_ID_ZENODO,
            provenance_note=f"Local HDF5; dataset root: {target_root.name}/.",
        )
        return result, target_root

    import_cfg = cfg.dataset.get("import") or {}
    cache_rel = Path(str(import_cfg.get("cache_dir", ".cache/zenodo_stage1")))
    cache_dir = cache_rel if cache_rel.is_absolute() else repo_root / cache_rel
    preferred = str(import_cfg.get("upstream_data_filename", DEFAULT_H5_NAME))
    record_id = str(cfg.dataset.get("zenodo_record", "18669296"))
    override_url = import_cfg.get("upstream_data_url")
    override_url = str(override_url) if override_url else None

    try:
        upstream_path = resolve_upstream_file(
            record_id=record_id,
            cache_dir=cache_dir,
            preferred_name=preferred,
            override_url=override_url,
            local_path=None,
        )
    except FileNotFoundError as exc:
        if mode == Stage1ImportMode.ZENODO:
            raise
        log_handler.warning("Zenodo import failed (%s); using LBM fallback.", exc)
        return _import_generated(cfg, repo_root, local_upstream=None, lbm_fast=lbm_fast)

    target_root = zenodo_dataset_root(repo_root)
    result = _import_with_root(
        cfg,
        dataset_root=target_root,
        local_upstream=upstream_path,
        repo_root=repo_root,
        source_id=SOURCE_ID_ZENODO,
        provenance_note=f"Zenodo/cache upstream; dataset root: {target_root.name}/.",
    )
    return result, target_root
