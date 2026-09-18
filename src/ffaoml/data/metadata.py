"""``dataset/metadata.csv`` helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ffaoml.contracts import METADATA_CSV_COLUMNS


def metadata_csv_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "metadata.csv"


def ensure_metadata_csv(dataset_root: str | Path) -> Path:
    path = metadata_csv_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(METADATA_CSV_COLUMNS))
            writer.writeheader()
    return path


def append_metadata_row(dataset_root: str | Path, row: dict[str, Any]) -> Path:
    """Append one simulation row; creates the CSV with header if missing."""
    path = ensure_metadata_csv(dataset_root)
    missing = set(METADATA_CSV_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"metadata row missing columns: {sorted(missing)}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA_CSV_COLUMNS))
        writer.writerow({col: row[col] for col in METADATA_CSV_COLUMNS})
    return path
