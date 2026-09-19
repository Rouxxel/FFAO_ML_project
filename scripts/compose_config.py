#!/usr/bin/env python3
"""
#############################################################################
### Compose and print default Hydra config
###
### @file compose_config.py
### @author Sebastian Russo
### @date 2026
#############################################################################

CLI helper: load ``configs/config.yaml`` with default groups and print the
fully resolved YAML (no CFD or ML execution).
"""

# Native imports
from pathlib import Path

# Third-party imports
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Project imports
from ffaoml.app_logging import log_handler
from ffaoml.config import config_dir

"""CONSTANTS-----------------------------------------------------------"""
REPO_ROOT = Path(__file__).resolve().parents[1]

"""ENTRYPOINT-----------------------------------------------------------"""


def main() -> None:
    """
    Compose the default config and print YAML to stdout.

    Returns:
        None
    """
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    log_handler.info("%s", OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
