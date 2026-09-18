#!/usr/bin/env python3
"""Print the resolved default Hydra config (no CFD/ML)."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ffaoml.config import config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with initialize_config_dir(
        config_dir=str(config_dir(REPO_ROOT)),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
