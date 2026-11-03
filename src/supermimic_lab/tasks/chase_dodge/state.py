"""Bridge between the mjlab env and the SuperMimic learner.

Defines the world-model state, the sim context xi, the robot planar pose, and
an exact re-implementation of the actor observation so that a policy trained
in imagination can be evaluated (and deployed) in the real env unchanged.

World-model state s (70):
  base height (1), gravity in body frame (3), base linear velocity in the yaw
  frame (3), base angular velocity in body frame (3), joint pos - default (12),
  joint vel (12), then hip, knee and foot positions relative to the base in the
  yaw frame (3 x 4 x 3, legs ordered FR FL RR RL). The limb points make contact
  changes (touchdown / liftoff) explicit and let imagination compute per-leg
  collisions with obstacles exactly as the env does.
The planar pose (x, y, yaw) is not part of s: the model predicts its change in
the previous yaw frame, which keeps the dynamics translation/yaw invariant.
"""

from __future__ import annotations


import torch

from supermimic_lab.tasks.chase_dodge.targets import interface as ti
from supermimic_lab.tasks.chase_dodge.targets.interface import yaw_rotate_inverse
from supermimic_lab.tasks.chase_dodge.targets.terrain import SCAN_POINTS, ground

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC, LEGS
from supermimic_lab.envs.specs import DynamicsSpec
from supermimic_lab.backends.mjlab import state_reader as reader
from supermimic_lab.backends.mjlab.state_reader import RobotConsts
from supermimic_lab.envs.observations import push_history
from supermimic_lab.utils.math import pose_delta, apply_pose_delta

LAYOUT = GO2_SPEC.state_layout
STATE_DIM = LAYOUT.dim
NUM_JOINTS = GO2_SPEC.action_dim
POSE_DELTA_DIM = 3
CONTEXT_DIM = 4
DYNAMICS = DynamicsSpec(STATE_DIM, NUM_JOINTS, CONTEXT_DIM, SCAN_POINTS)
Z = LAYOUT.slice("height")
GRAV = LAYOUT.slice("gravity")
V_YAW = LAYOUT.slice("linear_velocity")
W_B = LAYOUT.slice("angular_velocity")
Q = LAYOUT.slice("joint_pos")
QD = LAYOUT.slice("joint_vel")
HIPS = LAYOUT.slice("hips")
KNEES = LAYOUT.slice("knees")
FEET = LAYOUT.slice("feet")
FOOT_SITES = GO2_SPEC.foot_sites
FOOT_RADIUS = 0.022


def feet_world_z(s: torch.Tensor) -> torch.Tensor:
  """World-frame foot site heights (N, 4) from the state (yaw rotation keeps z)."""
  return s[:, Z] + s[:, FEET].reshape(-1, 4, 3)[..., 2]


def limb_points_world(s: torch.Tensor, pose: torch.Tensor) -> dict[str, torch.Tensor]:
  """World-frame hips, knees and feet (each (N, 4, 3)) from the state and the
  planar pose (x, y, yaw)."""
  c, si = pose[:, 2].cos(), pose[:, 2].sin()
  base = torch.cat([pose[:, :2], s[:, Z]], -1).unsqueeze(1)
  out = {}
  for name, sl in (("hips", HIPS), ("knees", KNEES), ("feet", FEET)):
    rel = s[:, sl].reshape(-1, 4, 3)
    x = c.unsqueeze(-1) * rel[..., 0] - si.unsqueeze(-1) * rel[..., 1]
    y = si.unsqueeze(-1) * rel[..., 0] + c.unsqueeze(-1) * rel[..., 1]
    out[name] = base + torch.stack([x, y, rel[..., 2]], -1)
  return out

# Actor observation terms in env order, with per-frame dims (see env_cfg.py);
# ("lidar", rays) follows with obstacles, ("height_scan", SCAN_POINTS) on terrain.
OBS_TERMS = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("actions", 12),
  ("target", ti.DIM),
)
# Uniform observation noise half-widths, matching env_cfg.py.
OBS_NOISE = {
  "base_ang_vel": 0.2,
  "projected_gravity": 0.05,
  "joint_pos": 0.01,
  "joint_vel": 1.5,
  "lidar": 0.03,
  "height_scan": 0.02,
}


def obs_terms(env_cfg) -> tuple[tuple[str, int], ...]:
  """Actor terms (name, per-frame dim) of an env config, in order."""
  names = list(env_cfg.observations["actor"].terms)
  rays = env_cfg.commands["target"].target.obstacles.lidar_rays
  known = dict(OBS_TERMS) | {"lidar": rays, "height_scan": SCAN_POINTS}
  terms = tuple((n, known[n]) for n in names)
  assert [n for n, _ in terms[: len(OBS_TERMS)]] == [n for n, _ in OBS_TERMS], names
  return terms


def obs_frame(
  s: torch.Tensor,
  last_action: torch.Tensor,
  target_obs: torch.Tensor,
  encoder_bias: torch.Tensor,
  noise: bool,
  lidar: torch.Tensor | None = None,
  height_scan: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """One frame of each actor observation term, computed from the model state."""
  frame = {
    "base_ang_vel": s[:, W_B],
    "projected_gravity": s[:, GRAV],
    "joint_pos": s[:, Q] + encoder_bias,
    "joint_vel": s[:, QD],
    "actions": last_action,
    "target": target_obs,
  }
  if lidar is not None:
    frame["lidar"] = lidar
  if height_scan is not None:
    frame["height_scan"] = height_scan
  if noise:
    for k, w in OBS_NOISE.items():
      if k in frame:
        frame[k] = frame[k] + (torch.rand_like(frame[k]) * 2 - 1) * w
  return frame


# Task-facing bindings keep the robot choice out of generic backend code.
def robot_consts(env):
  return reader.robot_consts(env, GO2_SPEC)


def limb_positions_w(robot):
  return reader.limb_positions_w(robot, GO2_SPEC)


def robot_state(env, field=None):
  return reader.robot_state(env, GO2_SPEC, ground(env.scene["robot"].data.root_link_pos_w[:, :2], field))


def planar_pose(env):
  return reader.planar_pose(env)


class ContextReader(reader.ContextReader):
  def __init__(self, env):
    super().__init__(env, GO2_SPEC)
