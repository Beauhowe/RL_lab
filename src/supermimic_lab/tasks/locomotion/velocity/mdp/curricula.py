"""Open harder terrain rows during training; select a fixed row in evaluation."""

import torch


def terrain_levels(env, env_ids, steps=12_000, play=False):
  terrain = env.scene.terrain
  rows = terrain.terrain_origins.shape[0]
  progress = min(1.0, env.common_step_counter / max(1, steps))
  maximum = min(rows - 1, int(progress * rows))
  if play:
    levels = torch.full((len(env_ids),), maximum, device=env_ids.device, dtype=torch.long)
  else:
    levels = torch.randint(0, maximum + 1, (len(env_ids),), device=env_ids.device)
  terrain.terrain_levels[env_ids] = levels
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
  return {"max_level": torch.tensor(float(maximum))}
