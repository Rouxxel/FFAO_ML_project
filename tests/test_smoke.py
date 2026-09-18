"""
#############################################################################
### Package smoke tests
###
### @file test_smoke.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Fast import and contract checks for CI (no CFD or PyTorch).
"""

# Project imports
import ffaoml
from ffaoml.contracts import DEFAULT_FIELD_CHANNELS, METADATA_CSV_COLUMNS

"""TESTS-----------------------------------------------------------"""


def test_version_is_set() -> None:
    """``ffaoml.__version__`` is defined after install."""
    assert ffaoml.__version__


def test_default_channels() -> None:
    """Default ML channel stack has four fields."""
    assert len(DEFAULT_FIELD_CHANNELS) == 4


def test_metadata_columns() -> None:
    """Metadata CSV includes core simulation identifiers."""
    assert "sim_id" in METADATA_CSV_COLUMNS
    assert "re" in METADATA_CSV_COLUMNS
    assert "split" in METADATA_CSV_COLUMNS
