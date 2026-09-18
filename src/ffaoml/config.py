"""
#############################################################################
### Hydra configuration helpers
###
### @file config.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Resolve ``configs/`` paths, persist composed runs, and apply global RNG seeds
(PRD §17).
"""

# Native imports
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Mapping

"""CONSTANTS-----------------------------------------------------------"""
RUN_CONFIG_FILENAME = "config.yaml"

"""PATHS-----------------------------------------------------------"""


def config_dir(repo_root: Path | None = None) -> Path:
    """
    Return the repository ``configs/`` directory.

    Parameters:
        repo_root (Path | None): Repository root; inferred from this file when
            omitted.

    Returns:
        Path: Absolute path to ``configs/``.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs"


def run_directory(runs_root: str | Path, run_id: str) -> Path:
    """
    Path for a single training or evaluation run.

    Parameters:
        runs_root (str | Path): Base directory (e.g. ``results/runs``).
        run_id (str): Unique run identifier.

    Returns:
        Path: ``<runs_root>/<run_id>/``.
    """
    return Path(runs_root) / run_id


"""RUN ARTIFACTS-----------------------------------------------------------"""


def write_resolved_config(cfg: DictConfig | Mapping, run_dir: str | Path) -> Path:
    """
    Write the fully resolved configuration for reproducibility.

    Parameters:
        cfg (DictConfig | Mapping): Composed Hydra config.
        run_dir (str | Path): Run output directory.

    Returns:
        Path: Written ``config.yaml`` path.
    """
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / RUN_CONFIG_FILENAME
    OmegaConf.save(OmegaConf.create(cfg), out_path)
    return out_path


"""SEEDING-----------------------------------------------------------"""


def resolve_seed(cfg: DictConfig) -> int:
    """
    Read the global seed from a composed config.

    Parameters:
        cfg (DictConfig): Composed config with top-level ``seed``.

    Returns:
        int: Seed value.

    Raises:
        KeyError: If ``seed`` is missing.
    """
    if "seed" not in cfg:
        raise KeyError("composed config must define top-level 'seed'")
    return int(cfg.seed)


def set_global_seed(seed: int) -> None:
    """
    Propagate ``seed`` to Python, NumPy, and PyTorch when installed.

    Parameters:
        seed (int): RNG seed.
    """
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
    """
    Apply ``cfg.seed`` to global RNGs and return the value.

    Parameters:
        cfg (DictConfig): Composed config.

    Returns:
        int: Applied seed.
    """
    seed = resolve_seed(cfg)
    set_global_seed(seed)
    return seed
