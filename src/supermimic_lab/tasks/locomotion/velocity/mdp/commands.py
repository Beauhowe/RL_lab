"""Velocity command term and terrain/state access for the mjlab environment."""

from dataclasses import dataclass, field
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.backends.mjlab import state_reader
from supermimic_lab.envs import terrain
from .. import state as st
from ..commands import VelocityCfg, sample_commands
from ..rewards import RewardCfg, reward_terms


class VelocityCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene["robot"]
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.height_field = None
    if cfg.rough:
      exclude = {int(g) for g in self.robot.indexing.geom_ids}
      self.height_field = terrain.env_height_field(
        env, exclude, half=4.5, res=cfg.field_resolution, cache_dir="logs/heightfields")
    self.previous_state = torch.zeros(self.num_envs, st.LAYOUT.dim, device=self.device)
    self.previous_pose = torch.zeros(self.num_envs, 3, device=self.device)
    self.history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._reward_step = -1
    self._reward_terms = {}
    self._constants = None

  @property
  def command(self):
    return self._command

  def _resample_command(self, env_ids):
    progress = self._env.common_step_counter / max(1, self.cfg.velocity.curriculum_steps)
    self._command[env_ids] = sample_commands(len(env_ids), self.device, progress, self.cfg.velocity)

  def _update_command(self, env_ids):
    pass

  def _update_metrics(self):
    pass

  def reset(self, env_ids):
    self.history_valid[env_ids] = False
    self._reward_step = -1
    return super().reset(env_ids)

  def scan(self):
    pose = state_reader.planar_pose(self._env)
    return st.read_state(self._env, self.height_field)[:, st.Z] + terrain.relief(
      pose[:, :2], pose[:, 2], self.height_field)

  def packed_state(self):
    progress = min(1.0, self._env.common_step_counter / max(1, self.cfg.velocity.curriculum_steps))
    return torch.cat([self.command, self.time_left[:, None],
                      torch.full_like(self.time_left[:, None], progress)], -1)

  def reward_components(self):
    step = self._env.common_step_counter
    if step == self._reward_step:
      return self._reward_terms
    if self._constants is None:
      self._constants = state_reader.robot_consts(self._env, GO2_SPEC)
    s = st.read_state(self._env, self.height_field)
    pose = state_reader.planar_pose(self._env)
    valid = self.history_valid[:, None]
    s0 = torch.where(valid, self.previous_state, s)
    pose0 = torch.where(valid, self.previous_pose, pose)
    self._reward_terms = reward_terms(
      s0, s, pose0, pose, self._env.action_manager.action, self._env.action_manager.prev_action,
      self.command, self._constants, self.height_field, self._env.step_dt, self.cfg.reward)
    self.previous_state, self.previous_pose = s.clone(), pose.clone()
    self.history_valid.fill_(True)
    self._reward_step = step
    return self._reward_terms


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
  rough: bool = False
  field_resolution: float = 0.05
  velocity: VelocityCfg = field(default_factory=VelocityCfg)
  reward: RewardCfg = field(default_factory=RewardCfg)

  def build(self, env):
    return VelocityCommand(self, env)
