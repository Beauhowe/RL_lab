"""Chase/dodge observations."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from .common import _term

def height_scan(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Base height above each point of the local ground grid (terrain)."""
  return _term(env, command_name).scan()

def lidar(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Planar LiDAR ranges around the robot to the static obstacles."""
  return _term(env, command_name).lidar()
