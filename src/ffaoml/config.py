"""Hydra/OmegaConf helpers, run directories, and global seeding (PRD §17)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Mapping

RUN_CONFIG_FILENAME = "config.yaml"


def config_dir(repo_root: Path | None = None) -> Path:
    """Return the ``configs/`` directory (repository root inferred when omitted)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs"


def run_directory(runs_root: str | Path, run_id: str) -> Path:
    """Path for a single run: ``results/runs/<run_id>/``."""
    return Path(runs_root) / run_id


def write_resolved_config(cfg: DictConfig | Mapping, run_dir: str | Path) -> Path:
    """Write the fully resolved configuration for reproducibility."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / RUN_CONFIG_FILENAME
    OmegaConf.save(OmegaConf.create(cfg), out_path)
    return out_path


def resolve_seed(cfg: DictConfig) -> int:
    """Read global seed from composed config (top-level ``seed``)."""
    if "seed" not in cfg:
        raise KeyError("composed config must define top-level 'seed'")
    return int(cfg.seed)


def set_global_seed(seed: int) -> None:
    """Propagate ``seed`` to Python, NumPy, and PyTorch when installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_from_config(cfg: DictConfig) -> int:
    """Apply ``cfg.seed`` and return the value."""
    seed = resolve_seed(cfg)
    set_global_seed(seed)
    return seed
