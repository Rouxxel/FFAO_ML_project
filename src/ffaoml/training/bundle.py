"""
#############################################################################
### Checkpoint bundle helpers
###
### @file bundle.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Finalize training runs with reproducibility metadata
(``documentation/REPRODUCIBILITY.md``).
"""

# Native imports
from __future__ import annotations

from pathlib import Path

# Third-party imports
from omegaconf import DictConfig

# Project imports
from ffaoml.manifests import write_checkpoint_bundle_manifest

"""BUNDLE-----------------------------------------------------------"""


def finalize_training_bundle(
    cfg: DictConfig,
    run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write ``bundle_manifest.json`` after config and dataset manifest are in place.

    Parameters:
        cfg (DictConfig): Resolved training config (includes ``seed``).
        run_dir (str | Path): Training output directory.
        repo_root (str | Path | None): For ``git_commit`` in the bundle manifest.

    Returns:
        Path: Path to ``bundle_manifest.json``.
    """
    return write_checkpoint_bundle_manifest(run_dir, cfg, repo_root=repo_root)
