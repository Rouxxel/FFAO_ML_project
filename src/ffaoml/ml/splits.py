"""
#############################################################################
### Dataset split helpers
###
### @file splits.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Temporal index splits (Stage 1) and Reynolds simulation lists (Stage 3 / own CFD).
"""

# Native imports
from __future__ import annotations

from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Project imports
from ffaoml.data.catalog import temporal_index_range
from ffaoml.data.metadata import read_metadata_rows

"""TEMPORAL-----------------------------------------------------------"""


def temporal_splits_disjoint(cfg: DictConfig) -> bool:
    """
    Return True when train/val/test time ranges do not overlap.

    Parameters:
        cfg (DictConfig): Config with ``dataset.temporal_split``.

    Returns:
        bool: ``False`` if any index appears in more than one split.
    """
    if bool(cfg.dataset.get("use_re_splits", False)):
        return True
    temporal = cfg.dataset.temporal_split
    ranges: dict[str, set[int]] = {}
    for name in ("train", "val", "test"):
        if name not in temporal:
            continue
        start, end = temporal_index_range(cfg, name)
        ranges[name] = set(range(start, end))
    seen: set[int] = set()
    for indices in ranges.values():
        if seen.intersection(indices):
            return False
        seen |= indices
    return True


"""REYNOLDS-----------------------------------------------------------"""


def filter_metadata_by_re(
    rows: list[dict[str, Any]],
    re_values: list[float] | list[int],
) -> list[dict[str, Any]]:
    """
    Select ``metadata.csv`` rows whose ``re`` is in ``re_values``.

    Parameters:
        rows (list[dict[str, Any]]): Parsed metadata rows.
        re_values (list[float] | list[int]): Allowed Reynolds numbers.

    Returns:
        list[dict[str, Any]]: Matching rows.
    """
    allowed = {float(r) for r in re_values}
    selected: list[dict[str, Any]] = []
    for row in rows:
        if float(row["re"]) in allowed:
            selected.append(row)
    return selected


def re_split_simulation_ids(cfg: DictConfig) -> dict[str, list[str]]:
    """
    Map split names to ``sim_id`` lists using ``train_re`` / ``val_re`` / ``test_re``.

    Parameters:
        cfg (DictConfig): ``dataset=splits`` style config.

    Returns:
        dict[str, list[str]]: Keys ``train``, ``val``, ``test``.

    Raises:
        ValueError: If ``use_re_splits`` is false.
    """
    if not bool(cfg.dataset.get("use_re_splits", False)):
        raise ValueError("re_split_simulation_ids requires use_re_splits: true")
    root = cfg.dataset.output_root
    rows = read_metadata_rows(root)
    return {
        "train": [
            r["sim_id"] for r in filter_metadata_by_re(rows, list(cfg.dataset.train_re))
        ],
        "val": [
            r["sim_id"] for r in filter_metadata_by_re(rows, list(cfg.dataset.val_re))
        ],
        "test": [
            r["sim_id"] for r in filter_metadata_by_re(rows, list(cfg.dataset.test_re))
        ],
    }
