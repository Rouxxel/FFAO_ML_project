"""Fast checks for CI (no CFD or PyTorch)."""

import ffaoml
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS, METADATA_CSV_COLUMNS


def test_version_is_set() -> None:
    assert ffaoml.__version__


def test_default_channels() -> None:
    assert len(DEFAULT_FIELD_CHANNELS) == 4


def test_metadata_columns() -> None:
    assert "sim_id" in METADATA_CSV_COLUMNS
    assert "re" in METADATA_CSV_COLUMNS
    assert "split" in METADATA_CSV_COLUMNS
