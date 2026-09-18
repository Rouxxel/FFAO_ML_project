"""FFAO ML: machine learning for 2D flow around a cylinder.

Research codebase for CFD simulation, datasets, and ML models predicting
2D incompressible flow around a cylinder. See repository documentation/ for
requirements and architecture.
"""

from importlib.metadata import PackageNotFoundError, version

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
