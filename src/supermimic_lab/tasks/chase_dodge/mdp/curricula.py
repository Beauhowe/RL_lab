"""Chase/dodge curricula."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def terrain_levels_by_progress(env: ManagerBasedRlEnv, env_ids: torch.Tensor,
                               steps: int) -> dict[str, torch.Tensor]:
  """Terrain difficulty rises with training progress: row k (of R) opens once
  k / R of `steps` env steps have passed; each reset picks a random open row,
  so the easier rows keep being practised."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {}
  rows = terrain.terrain_origins.shape[0]
  progress = min(1.0, env.common_step_counter / max(1, steps))
  top = min(rows - 1, int(progress * rows))
  levels = torch.randint(0, top + 1, (len(env_ids),), device=env_ids.device)
  terrain.terrain_levels[env_ids] = levels
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
  return {"max_level": torch.tensor(float(top))}
