"""
#############################################################################
### Custom logger
###
### @file custom_logger.py
### @author Sebastian Russo
### @date 2025
#############################################################################

Project-wide logger: console + daily file under ``<repo>/log/ffao_ml_YYYY-MM-DD.log``.
"""

# Native imports
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# --- CONFIGURATION ---
LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIRECTORY = REPO_ROOT / "log"
LOG_FILE_BASENAME = "ffao_ml"
LOG_LEVEL_STR = "info"

log_level = LOG_LEVELS.get(LOG_LEVEL_STR.lower(), logging.INFO)

log_format = logging.Formatter(
    fmt="%(asctime)s %(msecs)03dZ | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log_handler = logging.getLogger("ffao_ml")
log_handler.setLevel(log_level)
log_handler.propagate = False

file_handler = None
log_file: Path | None = None

try:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = LOG_DIRECTORY / f"{LOG_FILE_BASENAME}_{date_stamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
except OSError as exc:
    sys.stderr.write(f"ERROR: Failed to create log file under '{LOG_DIRECTORY}': {exc}\n")
    sys.stderr.write("Continuing with console-only logging.\n")
    file_handler = None

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not log_handler.handlers:
    if file_handler is not None:
        log_handler.addHandler(file_handler)
    log_handler.addHandler(console_handler)

log_handler.info("FFAO ML logging initialized")
if log_file is not None:
    log_handler.info("Log file: %s (cwd: %s)", log_file, Path.cwd())
else:
    log_handler.warning("File logging unavailable — console only (cwd: %s)", Path.cwd())


def shutdown_logger() -> None:
    """
    Flush and close all handlers (call before process exit when convenient).
    """
    try:
        for handler in log_handler.handlers[:]:
            try:
                handler.flush()
                handler.close()
                log_handler.removeHandler(handler)
            except OSError as exc:
                sys.stderr.write(f"Error closing log handler: {exc}\n")
    except OSError as exc:
        sys.stderr.write(f"Error during logger shutdown: {exc}\n")
