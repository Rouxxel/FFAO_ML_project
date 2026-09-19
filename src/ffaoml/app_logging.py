"""
#############################################################################
### Application logging entry point
###
### @file app_logging.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Import ``log_handler`` from ``src/utils/custom_logger`` when using the installed
``ffaoml`` package or repo scripts.
"""

# Native imports
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.custom_logger import log_handler, shutdown_logger  # noqa: E402

__all__ = ["log_handler", "shutdown_logger"]
