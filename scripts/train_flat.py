"""Flat-ground training entry point for an IDE's Run/Debug button.

The defaults match train_flat.sh. Extra command-line flags override them.
Select the shared rl/.venvs/mjlab-1.6/bin/python interpreter in your IDE.
"""

import os
from pathlib import Path
import sys

from supermimic_lab.runners.cli import main as train_main

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARGS = [
  "--config", str(PROJECT_DIR / "configs/experiments/go2_chase_flat.toml"),
  "--num-envs", "2048",
  "--imag-batch", "4096",
  "--buffer-len", "400",
  "--compile", "false",
]


def main():
  # Keep relative configuration/checkpoint paths and logs consistent across IDEs.
  os.chdir(PROJECT_DIR)
  train_main([*DEFAULT_ARGS, *sys.argv[1:]])


if __name__ == "__main__":
  main()
