"""
#############################################################################
### FFAO ML package root
###
### @file __init__.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Public entry point for the ``ffaoml`` package: version string and shared
field-layout contracts used by CFD export, datasets, and models.

See ``documentation/`` for requirements, architecture, and data contracts.
"""

# Native imports
from importlib.metadata import PackageNotFoundError, version

# Project imports
from ffaoml.contracts import (
    DEFAULT_FIELD_CHANNELS,
    METADATA_CSV_COLUMNS,
    FieldChannel,
)

try:
    __version__ = version("ffaoml")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_FIELD_CHANNELS",
    "FieldChannel",
    "METADATA_CSV_COLUMNS",
    "__version__",
]
