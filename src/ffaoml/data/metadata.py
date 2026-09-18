"""
#############################################################################
### Dataset metadata.csv helpers
###
### @file metadata.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Create and append rows to ``dataset/metadata.csv`` using
``METADATA_CSV_COLUMNS`` from ``ffaoml.contracts``.
"""

# Native imports
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Project imports
from ffaoml.contracts import METADATA_CSV_COLUMNS

"""PATHS-----------------------------------------------------------"""


def metadata_csv_path(dataset_root: str | Path) -> Path:
    """
    Return ``<dataset_root>/metadata.csv``.

    Parameters:
        dataset_root (str | Path): Dataset root directory.

    Returns:
        Path: CSV path.
    """
    return Path(dataset_root) / "metadata.csv"


def ensure_metadata_csv(dataset_root: str | Path) -> Path:
    """
    Create ``metadata.csv`` with a header row if it does not exist.

    Parameters:
        dataset_root (str | Path): Dataset root directory.

    Returns:
        Path: CSV path (existing or newly created).
    """
    path = metadata_csv_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(METADATA_CSV_COLUMNS))
            writer.writeheader()
    return path


"""ROWS-----------------------------------------------------------"""


def read_metadata_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    """
    Load all rows from ``metadata.csv``.

    Parameters:
        dataset_root (str | Path): Dataset root directory.

    Returns:
        list[dict[str, Any]]: Parsed rows (empty when the file is missing).
    """
    path = metadata_csv_path(dataset_root)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def append_metadata_row(dataset_root: str | Path, row: dict[str, Any]) -> Path:
    """
    Append one simulation row to ``metadata.csv``.

    Parameters:
        dataset_root (str | Path): Dataset root directory.
        row (dict[str, Any]): Values for every ``METADATA_CSV_COLUMNS`` key.

    Returns:
        Path: Updated CSV path.

    Raises:
        ValueError: If any required column is missing from ``row``.
    """
    path = ensure_metadata_csv(dataset_root)
    missing = set(METADATA_CSV_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"metadata row missing columns: {sorted(missing)}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA_CSV_COLUMNS))
        writer.writerow({col: row[col] for col in METADATA_CSV_COLUMNS})
    return path
