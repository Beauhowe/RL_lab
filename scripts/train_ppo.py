"""IDE-friendly mjlab PPO entry point; flags can select another PPO preset."""

import os
from pathlib import Path
import sys
from supermimic_lab.runners.cli import main as train_main

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
  os.chdir(PROJECT_DIR)
  train_main(['--config', str(PROJECT_DIR / 'configs/experiments/go2_ppo_mjlab_flat.toml'),
              *sys.argv[1:]])


if __name__ == '__main__':
  main()
