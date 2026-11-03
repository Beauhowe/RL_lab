"""Shared analytic reward terms and smooth fall probability for locomotion."""

from dataclasses import dataclass
import math
import torch

from . import state as st


@dataclass(kw_only=True)
class RewardWeights:
  track_linear_velocity: float = 2.0
  track_angular_velocity: float = 1.0
  upright: float = 0.5
  base_height: float = -30.0
  vertical_velocity: float = -0.3
  tilt_rate: float = -0.05
  joint_deviation: float = -0.1
  action_rate: float = -0.05
  joint_limits: float = -1.0
  torques: float = -2e-5
  joint_acceleration: float = -2.5e-7
  foot_clearance: float = -2.0
  foot_slip: float = -0.1
  standing_motion: float = -0.2
  hard_landing: float = -0.2
  terminated: float = -200.0


@dataclass(kw_only=True)
class RewardCfg:
  linear_std: float = 0.5
  angular_std: float = 0.5
  base_height: float = 0.28
  swing_height: float = 0.08
  fall_angle: float = 60.0
  minimum_height: float = 0.12


def reward_terms(s0, s1, pose0, pose1, action, previous_action, command, constants,
                 field, dt, cfg: RewardCfg):
  error = s1[:, st.V][:, :2] - command[:, :2]
  yaw_error = s1[:, st.W][:, 2] - command[:, 2]
  q = s1[:, st.Q] + constants.default_q
  target = constants.default_q + action * constants.action_scale
  torque = (constants.kp * (target - q) - constants.kd * s1[:, st.QD]).clamp(
    -constants.effort, constants.effort)
  limits = constants.soft_limits
  clearance = st.foot_clearance(s1, pose1, field)
  foot_velocity = (st.foot_positions(s1, pose1, field) - st.foot_positions(s0, pose0, field)) / dt
  foot_speed_sq = foot_velocity[..., :2].square().sum(-1)
  contact = torch.exp(-(clearance.clamp(min=0) / 0.025).square())
  moving = (command.norm(dim=-1) > 0.05).float()
  return {
    "track_linear_velocity": torch.exp(-error.square().sum(-1) / cfg.linear_std**2),
    "track_angular_velocity": torch.exp(-yaw_error.square() / cfg.angular_std**2),
    "upright": torch.exp(-s1[:, st.GRAV][:, :2].square().sum(-1) / 0.4**2),
    "base_height": (s1[:, 0] - cfg.base_height).square(),
    "vertical_velocity": s1[:, st.V][:, 2].square(),
    "tilt_rate": s1[:, st.W][:, :2].square().sum(-1),
    "joint_deviation": s1[:, st.Q].square().sum(-1),
    "action_rate": (action - previous_action).square().sum(-1),
    "joint_limits": ((limits[..., 0] - q).clamp(min=0)
                     + (q - limits[..., 1]).clamp(min=0)).sum(-1),
    "torques": torque.square().sum(-1),
    "joint_acceleration": ((s1[:, st.QD] - s0[:, st.QD]) / dt).square().sum(-1),
    "foot_clearance": ((clearance - cfg.swing_height).square()
                       * torch.sqrt(foot_speed_sq + 1e-8) * (1 - contact)).sum(-1) * moving,
    "foot_slip": (foot_speed_sq * contact).sum(-1),
    "standing_motion": s1[:, st.QD].square().sum(-1) * (1 - moving),
    "hard_landing": (-s1[:, st.V][:, 2]).clamp(min=0).square()
                    * (clearance.min(-1).values < 0.04).float(),
  }


def fall_probability(s, cfg: RewardCfg):
  tilt = torch.acos((-s[:, st.GRAV][:, 2]).clamp(-1 + 1e-4, 1 - 1e-4))
  tipped = torch.sigmoid((tilt - math.radians(cfg.fall_angle)) / math.radians(5.0))
  low = torch.sigmoid((cfg.minimum_height - s[:, 0]) / 0.015)
  return 1 - (1 - tipped) * (1 - low)
