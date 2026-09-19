"""
#############################################################################
### Training progress logging
###
### @file progress.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Per-epoch messages to the console (and log file) during long training runs.
"""

# Project imports
from ffaoml.app_logging import log_handler

"""LOGGING-----------------------------------------------------------"""


def log_training_start(
    *,
    model_name: str,
    epochs: int,
    device: str,
    output_dir: str,
) -> None:
    """Log once before the epoch loop."""
    log_handler.info(
        "Training %s: %d epochs on %s → %s",
        model_name,
        epochs,
        device,
        output_dir,
    )


def log_training_epoch(
    *,
    model_name: str,
    epoch: int,
    epochs: int,
    val_mse: float,
    best_val_mse: float,
    is_best: bool,
) -> None:
    """Log validation MSE after each epoch (* marks a new best)."""
    marker = " *best*" if is_best else ""
    log_handler.info(
        "[%s] epoch %d/%d  val_mse=%.6f  best=%.6f%s",
        model_name,
        epoch,
        epochs,
        val_mse,
        best_val_mse,
        marker,
    )
    for handler in log_handler.handlers:
        try:
            handler.flush()
        except OSError:
            pass
