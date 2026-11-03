"""Differentiable velocity tracking through the learned robot dynamics."""

from dataclasses import dataclass, field, replace
import torch

from supermimic_lab.algorithms.supermimic.imagination import DifferentiableImagination
from supermimic_lab.envs import terrain
from supermimic_lab.envs.observations import push_history
from supermimic_lab.utils.math import apply_pose_delta
from . import state as st
from .commands import advance_commands
from .rewards import RewardCfg, RewardWeights, reward_terms, fall_probability


@dataclass(kw_only=True)
class ImagineCfg:
  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  dt: float = 0.02
  compile: bool = False
  sensor_grad: bool = True
  weights: RewardWeights = field(default_factory=RewardWeights)
  reward: RewardCfg = field(default_factory=RewardCfg)


class LocomotionImagination(DifferentiableImagination):
  critic_dim = st.LAYOUT.dim + 3 + terrain.SCAN_POINTS + st.DYNAMICS.context_dim

  def __init__(self, cfg, constants, commands, height_field, device):
    self.cfg, self.c, self.commands = cfg, constants, commands
    self.field, self.device = height_field, device
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def restore_task_state(self, packed):
    return packed

  def rollout_constants(self, seed, hint_weight):
    ids = seed["env"]
    return replace(self.c, default_q=self.c.default_q[ids], action_scale=self.c.action_scale[ids],
                   soft_limits=self.c.soft_limits[ids])

  def scan(self, s, pose, grad=False):
    return s[:, st.Z] + terrain.relief(pose[:, :2], pose[:, 2], self.field, grad=grad)

  def critic_input(self, s, pose, task, xi):
    return torch.cat([s, task[:, :3], self.scan(s, pose), xi], -1)

  def _step(self, s, pose, previous_action, hist, task, xi, encoder_bias, constants, dt, actor, wm):
    critic_in = self.critic_input(s, pose, task, xi)
    action = actor(hist)
    relief = terrain.relief(pose[:, :2], pose[:, 2], self.field, grad=self.cfg.sensor_grad)
    next_s, delta = wm(s, action, previous_action, xi, relief)
    next_pose = apply_pose_delta(pose, delta)
    terms = reward_terms(s, next_s, pose, next_pose, action, previous_action, task[:, :3],
                         constants, self.field, self.cfg.dt, self.cfg.reward)
    fall = fall_probability(next_s, self.cfg.reward)
    reward = sum(getattr(self.cfg.weights, name) * value for name, value in terms.items())
    reward = (reward + self.cfg.weights.terminated * fall) * self.cfg.dt
    next_task = advance_commands(task, dt, self.commands)
    scan = self.scan(next_s, next_pose, grad=self.cfg.sensor_grad)
    frame = st.observation_frame(next_s, action, next_task[:, :3], encoder_bias, scan, noise=True)
    hist = push_history(hist, frame, st.HISTORY, st.OBS_TERMS)
    return next_s, next_pose, action, hist, next_task, reward, fall, critic_in
