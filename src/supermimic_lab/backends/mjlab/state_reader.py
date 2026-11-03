"""mjlab kinematics and actuator reads parameterized by robot metadata."""

import re
from dataclasses import dataclass
import torch
from supermimic_lab.utils.math import yaw_rotate_inverse

@dataclass
class RobotConsts:
  """PD constants in controlled-joint order; A is the action dimension."""

  default_q: torch.Tensor
  action_scale: torch.Tensor
  kp: torch.Tensor  # (A,)
  kd: torch.Tensor  # (A,)
  effort: torch.Tensor  # (A,)
  soft_limits: torch.Tensor  # (N, A, 2)

def robot_consts(env, spec) -> RobotConsts:
  robot = env.scene["robot"]
  term = env.action_manager.get_term(spec.action_term)
  jids = term._target_ids
  names = term._target_names
  if len(names) != spec.action_dim:
    raise ValueError("controlled-joint count does not match the robot action specification")
  kp, kd, eff = [], [], []
  for n in names:
    for act in robot.cfg.articulation.actuators:
      if any(re.fullmatch(expr, n) for expr in act.target_names_expr):
        kp.append(act.stiffness)
        kd.append(act.damping)
        eff.append(act.effort_limit)
        break
    else:
      raise KeyError(f"no actuator for joint {n}")
  dev = env.device
  scale = term._scale
  if not torch.is_tensor(scale):
    scale = torch.full((env.num_envs, spec.action_dim), float(scale), device=dev)
  return RobotConsts(
    default_q=robot.data.default_joint_pos[:, jids].clone(),
    action_scale=scale.clone(),
    kp=torch.tensor(kp, device=dev),
    kd=torch.tensor(kd, device=dev),
    effort=torch.tensor(eff, device=dev),
    soft_limits=robot.data.soft_joint_pos_limits[:, jids].clone(),
  )

_LIMB_IDS = {}


def limb_positions_w(robot, spec) -> dict[str, torch.Tensor]:
  """World-frame keypoints in the order declared by RobotSpec."""
  key = (id(robot), spec)
  if key not in _LIMB_IDS:
    body_ids = {name: robot.find_bodies(names, preserve_order=True)[0]
                for name, names in spec.limb_bodies}
    feet, _ = robot.find_sites(spec.foot_sites, preserve_order=True)
    _LIMB_IDS[key] = (body_ids, feet)
  body_ids, feet = _LIMB_IDS[key]
  d = robot.data
  return {**{name: d.body_link_pos_w[:, ids] for name, ids in body_ids.items()},
          "feet": d.site_pos_w[:, feet]}

def robot_state(env, spec, ground_height=None) -> torch.Tensor:
  """Canonical floating-base state, flattened in the declared layout order."""
  d = env.scene["robot"].data
  heading = d.heading_w
  v_yaw = yaw_rotate_inverse(d.root_link_lin_vel_w, heading)
  limbs = limb_positions_w(env.scene["robot"], spec)
  base_h = d.root_link_pos_w[:, 2:3]
  if ground_height is not None:
    base_h = base_h - ground_height.unsqueeze(-1)
  values = {
    "height": base_h, "gravity": d.projected_gravity_b,
    "linear_velocity": v_yaw, "angular_velocity": d.root_link_ang_vel_b,
    "joint_pos": d.joint_pos - d.default_joint_pos, "joint_vel": d.joint_vel,
  }
  c, si = heading.cos().unsqueeze(-1), heading.sin().unsqueeze(-1)
  for name, points in limbs.items():
    r = points - d.root_link_pos_w.unsqueeze(1)
    values[name] = torch.stack([c * r[..., 0] + si * r[..., 1],
                                -si * r[..., 0] + c * r[..., 1], r[..., 2]], -1).flatten(1)
  for name, dim in spec.state_layout.terms:
    if values[name].shape[-1] != dim:
      raise ValueError(f"state field {name} does not match the declared layout")
  return torch.cat([values[name] for name, _ in spec.state_layout.terms], -1)

def planar_pose(env) -> torch.Tensor:
  d = env.scene["robot"].data
  return torch.cat([d.root_link_pos_w[:, :2], d.heading_w.unsqueeze(-1)], dim=-1)

class ContextReader:
  """Reads the per-env domain-randomization context xi from the sim model."""

  def __init__(self, env, spec):
    robot = env.scene["robot"]
    local_geoms, _ = robot.find_geoms(spec.foot_geoms)
    local_body, _ = robot.find_bodies(spec.base_body)
    self.foot_geom = robot.indexing.geom_ids[local_geoms[0]]
    self.base_body = robot.indexing.body_ids[local_body[0]]
    self.env = env
    self.base_ipos0 = self._ipos().clone()

  def _ipos(self) -> torch.Tensor:
    ipos = self.env.sim.model.body_ipos
    return ipos[:, self.base_body] if ipos.ndim == 3 else ipos[self.base_body].expand(
      self.env.num_envs, 3
    )

  def read(self) -> torch.Tensor:
    fr = self.env.sim.model.geom_friction
    mu = fr[:, self.foot_geom, 0] if fr.ndim == 3 else fr[self.foot_geom, 0].expand(
      self.env.num_envs
    )
    return torch.cat([mu.unsqueeze(-1), self._ipos() - self.base_ipos0], dim=-1)
