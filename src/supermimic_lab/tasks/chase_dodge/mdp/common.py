"""Chase/dodge common."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from supermimic_lab.tasks.chase_dodge.targets.terrain import ground

if TYPE_CHECKING:
  from supermimic_lab.tasks.chase_dodge.mdp.commands import TargetCommand

def _term(env: ManagerBasedRlEnv, command_name: str) -> TargetCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]

def _to_chase_ball(env: ManagerBasedRlEnv, command_name: str):
  """Unit direction robot->chase ball, horizontal distance, robot planar velocity."""
  term = _term(env, command_name)
  data = term.robot.data
  rel = term.gen.chase_pos[:, :2] - data.root_link_pos_w[:, :2]
  dist = rel.norm(dim=-1).clamp(min=1e-6)
  return term, rel / dist.unsqueeze(-1), dist, data.root_link_lin_vel_w[:, :2]

def _ground_under(env: ManagerBasedRlEnv, xy: torch.Tensor) -> torch.Tensor:
  """Terrain height under points; 0 unless the target command owns a height field."""
  terms = env.command_manager.active_terms
  field = env.command_manager.get_term("target").height_field if "target" in terms else None
  return ground(xy, field)
