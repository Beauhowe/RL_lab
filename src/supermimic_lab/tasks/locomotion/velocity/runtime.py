"""Locomotion runtime for the shared SuperMimic training loop."""

from dataclasses import asdict
import torch

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.backends.mjlab.env import make_env
from supermimic_lab.backends.mjlab import state_reader
from supermimic_lab.envs import terrain
from supermimic_lab.utils.math import pose_delta
from . import state as st
from .imagination import ImagineCfg, LocomotionImagination
from .rewards import RewardWeights


class LocomotionRuntime:
  robot_spec = GO2_SPEC
  dynamics = st.DYNAMICS
  task_dim = 5
  history = st.HISTORY
  terms = st.OBS_TERMS

  def __init__(self, cfg):
    self.cfg = cfg
    if cfg.driver_policy or cfg.expert_policy:
      raise ValueError("locomotion currently collects data with its own actor; external drivers are not configured")

    def configure(env_cfg):
      env_cfg.commands["twist"].velocity.curriculum_steps = cfg.command_curriculum_iterations
      if "terrain_levels" in env_cfg.curriculum:
        env_cfg.curriculum["terrain_levels"].params["steps"] = cfg.terrain_curriculum_iterations

    self.env, self.env_cfg = make_env(cfg, configure=configure)
    self.term = self.env.command_manager.get_term("twist")
    self.field = self.term.height_field
    obs, _ = self.env.reset()
    self.initial_obs = obs["actor"]
    self.constants = state_reader.robot_consts(self.env, GO2_SPEC)
    self.context = state_reader.ContextReader(self.env, GO2_SPEC).read()
    self.encoder_bias = self.env.scene["robot"].data.encoder_bias.clone()

  def state(self):
    return st.read_state(self.env, self.field)

  def pose(self):
    return state_reader.planar_pose(self.env)

  @staticmethod
  def pose_delta(before, after):
    return pose_delta(before, after)

  def task_state(self):
    return self.term.packed_state()

  def relief(self, pose):
    return terrain.relief(pose[:, :2], pose[:, 2], self.field)

  def make_imagination(self, cfg):
    weights = RewardWeights(**{name: term.weight for name, term in self.env_cfg.rewards.items()})
    settings = ImagineCfg(horizon=cfg.horizon, dt=self.env.step_dt, compile=cfg.compile,
                           sensor_grad=cfg.sensor_grad, weights=weights, reward=self.term.cfg.reward)
    return LocomotionImagination(settings, self.constants, self.term.cfg.velocity,
                                 self.field, self.env.device)

  def driven_actions(self, actions, obs, iteration):
    return actions

  def on_reset(self, done):
    pass

  def seed(self, env_ids, slots, buffer):
    return {"env": env_ids, "s": buffer.s[env_ids, slots], "pose": buffer.pose[env_ids, slots],
            "a_prev": buffer.a_prev[env_ids, slots], "obs": buffer.obs[env_ids, slots].float(),
            "task": buffer.task[env_ids, slots], "xi": self.context[env_ids],
            "obs_context": self.encoder_bias[env_ids]}

  def _contacts(self, windows):
    s, pose = windows["s"], windows["pose"]
    clearance = st.foot_clearance(s.reshape(-1, st.LAYOUT.dim), pose.reshape(-1, 3), self.field)
    return (clearance < 0.015).reshape(s.shape[0], s.shape[1], -1)

  def rarity(self, windows):
    contacts = self._contacts(windows).float()
    changes = (contacts[:, 1:] - contacts[:, :-1]).abs().sum((1, 2))
    flight = (contacts.sum(-1) == 0).float().sum(-1)
    return changes + flight + 4 * (windows["valid"][:, -1] < 0.5).float()

  def archive_worthy(self, windows):
    flight = (~self._contacts(windows)).all(-1).sum(-1) >= 2
    hard_vz = windows["s"][:, :, st.V][..., 2].abs().amax(-1) > 0.6
    return flight | hard_vz | (windows["valid"][:, -1] < 0.5)

  def metrics(self):
    s, command = self.state(), self.term.command
    values = {
      "linear_error": (s[:, st.V][:, :2] - command[:, :2]).norm(dim=-1).mean(),
      "yaw_error": (s[:, st.W][:, 2] - command[:, 2]).abs().mean(),
      "speed": s[:, st.V][:, :2].norm(dim=-1).mean(),
      "command_speed": command[:, :2].norm(dim=-1).mean(),
    }
    if self.term.cfg.rough:
      values["terrain_level"] = self.env.scene.terrain.terrain_levels.float().mean()
    values = {name: float(value) for name, value in values.items()}
    return {f"Metrics/locomotion/{name}": value for name, value in values.items()}

  def log_metrics(self, writer, iteration):
    # Per-step metrics are already averaged by the runner over the log window.
    return {}

  def checkpoint_metadata(self):
    action = self.env.action_manager.get_term("joint_pos")
    return {"format_version": 1, "task_family": "locomotion_velocity", "robot": GO2_SPEC.name,
            "backend": self.cfg.backend, "task": self.cfg.task,
            "state_terms": GO2_SPEC.state_layout.terms, "obs_terms": self.terms,
            "history": self.history, "action_names": list(action._target_names),
            "dynamics": asdict(self.dynamics),
            "state_frame": "base_height_above_ground;velocities_and_limbs_in_yaw_frame"}

  def checkpoint_terms(self, checkpoint):
    meta = checkpoint.get("metadata", {})
    current = self.checkpoint_metadata()
    for key in ("format_version", "task_family", "robot", "backend", "history", "state_frame"):
      if meta.get(key) != current[key]:
        raise ValueError(f"checkpoint {key} is incompatible with locomotion; use a locomotion checkpoint")
    if tuple(map(tuple, meta.get("state_terms", ()))) != GO2_SPEC.state_layout.terms:
      raise ValueError("checkpoint state layout differs")
    if meta.get("action_names") != current["action_names"]:
      raise ValueError("checkpoint action joint order differs")
    if self.cfg.continue_run and meta.get("task") != self.cfg.task:
      raise ValueError("continue-run requires the same task; warm-start a new terrain with continue-run false")
    return tuple(map(tuple, meta["obs_terms"]))
