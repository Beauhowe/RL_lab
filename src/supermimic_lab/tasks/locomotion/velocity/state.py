"""Go2 state and observation codecs for velocity-command locomotion."""

import torch

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.backends.mjlab import state_reader
from supermimic_lab.envs.specs import DynamicsSpec
from supermimic_lab.envs import terrain

LAYOUT = GO2_SPEC.state_layout
Z, GRAV = LAYOUT.slice("height"), LAYOUT.slice("gravity")
V, W = LAYOUT.slice("linear_velocity"), LAYOUT.slice("angular_velocity")
Q, QD, FEET = LAYOUT.slice("joint_pos"), LAYOUT.slice("joint_vel"), LAYOUT.slice("feet")
DYNAMICS = DynamicsSpec(LAYOUT.dim, GO2_SPEC.action_dim, 4, terrain.SCAN_POINTS)
HISTORY = 5
OBS_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 12),
             ("joint_vel", 12), ("actions", 12), ("command", 3),
             ("height_scan", terrain.SCAN_POINTS))
NOISE = {"base_ang_vel": 0.2, "projected_gravity": 0.05, "joint_pos": 0.01,
         "joint_vel": 1.5, "height_scan": 0.02}


def read_state(env, field):
  xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  return state_reader.robot_state(env, GO2_SPEC, terrain.ground(xy, field))


def foot_positions(s, pose, field):
  rel = s[:, FEET].reshape(-1, len(GO2_SPEC.foot_sites), 3)
  c, si = pose[:, 2:3].cos(), pose[:, 2:3].sin()
  x = pose[:, 0:1] + c * rel[..., 0] - si * rel[..., 1]
  y = pose[:, 1:2] + si * rel[..., 0] + c * rel[..., 1]
  base_z = s[:, Z] + terrain.ground(pose[:, :2], field)[:, None]
  return torch.stack([x, y, base_z + rel[..., 2]], -1)


def foot_clearance(s, pose, field):
  feet = foot_positions(s, pose, field)
  return feet[..., 2] - terrain.ground(feet[..., :2], field) - 0.022


def observation_frame(s, action, command, encoder_bias, scan, noise=False):
  frame = {"base_ang_vel": s[:, W], "projected_gravity": s[:, GRAV],
           "joint_pos": s[:, Q] + encoder_bias, "joint_vel": s[:, QD],
           "actions": action, "command": command, "height_scan": scan}
  if noise:
    for name, width in NOISE.items():
      frame[name] = frame[name] + (2 * torch.rand_like(frame[name]) - 1) * width
  return frame
