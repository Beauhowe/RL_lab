"""Environment wrappers for the shared analytic locomotion rewards."""

import math
import torch
from .. import state as st


def component(env, name):
  return env.command_manager.get_term("twist").reward_components()[name]


def fell_over(env):
  term = env.command_manager.get_term("twist")
  state = st.read_state(env, term.height_field)
  tilt = torch.acos((-state[:, st.GRAV][:, 2]).clamp(-1, 1))
  return (tilt > math.radians(term.cfg.reward.fall_angle)) | (state[:, 0] < term.cfg.reward.minimum_height)


def tile_exit(env, half_width=3.5):
  xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  return ((xy - env.scene.env_origins[:, :2]).abs() > half_width).any(-1)
