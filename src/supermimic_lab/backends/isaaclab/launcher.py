"""Launch Isaac Sim before importing environment, assets or RL code."""

from importlib.util import find_spec


def launch(cfg):
  if find_spec('isaaclab') is None or find_spec('isaacsim') is None:
    raise RuntimeError(
      'Isaac Lab backend requires its own Isaac Lab / Isaac Sim Python environment. '
      'Install Isaac Lab 2.3 with Isaac Sim 5.1 and its RSL-RL dependencies, then '
      'install this project with: python -m pip install -e ".[isaaclab]". '
      'The mjlab virtual environment is not an Isaac Sim installation.')
  from isaaclab.app import AppLauncher

  return AppLauncher({'headless': cfg.headless, 'device': cfg.device}).app
