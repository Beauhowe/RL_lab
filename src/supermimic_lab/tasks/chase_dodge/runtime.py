"""Bind the original Go2 chase/dodge task to the generic SuperMimic runner."""

from dataclasses import asdict, fields

import torch

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.backends.mjlab.env import make_env
from . import state as st
from .imagination import ImagineCfg, Imagination, RewardWeights
from .targets import terrain as tr
from .targets.serialization import GEN_DIM, pack_generator


class ChaseDodgeRuntime:
  robot_spec = GO2_SPEC
  dynamics = st.DYNAMICS
  task_dim = GEN_DIM

  def __init__(self, cfg):
    self.cfg = cfg

    def configure(env_cfg):
      env_cfg.commands["target"].target.curriculum_steps = cfg.target_curriculum_iterations
      if "terrain_levels" in env_cfg.curriculum:
        env_cfg.curriculum["terrain_levels"].params["steps"] = cfg.terrain_curriculum_iterations

    self.env, self.env_cfg = make_env(cfg, configure=configure)
    self.term = self.env.command_manager.get_term("target")
    self.history = self.env_cfg.observations["actor"].history_length
    self.terms = st.obs_terms(self.env_cfg)
    obs, _ = self.env.reset()
    self.initial_obs = obs["actor"]
    self.consts = st.robot_consts(self.env)
    self.context = st.ContextReader(self.env).read()
    self.encoder_bias = self.env.scene["robot"].data.encoder_bias.clone()
    self.field = self.term.height_field
    self.driver = self.expert = None
    if cfg.driver_policy:
      from .drivers import ChaseSteer, MoeCtsDriver

      self.driver = MoeCtsDriver(self.env, cfg.driver_policy)
      self.steer = ChaseSteer(cfg.num_envs, self.env.device,
                              (cfg.driver_vmin, cfg.driver_vmax),
                              0.8 * self.term.cfg.target.arena_radius)
    if cfg.expert_policy:
      from .drivers import ActorDriver

      self.expert = ActorDriver(cfg.expert_policy, self.initial_obs.shape[-1], self.env.device)
    self.env_index = torch.arange(cfg.num_envs, device=self.env.device)

  def state(self):
    return st.robot_state(self.env, field=self.field)

  def pose(self):
    return st.planar_pose(self.env)

  @staticmethod
  def pose_delta(before, after):
    return st.pose_delta(before, after)

  def task_state(self):
    return pack_generator(self.term.gen)

  def relief(self, pose):
    return tr.relief(pose[:, :2], pose[:, 2], self.field, self.term.obstacle_layout)

  def make_imagination(self, cfg):
    weights = RewardWeights(**{
      f.name: self.env_cfg.rewards[f.name].weight
      for f in fields(RewardWeights) if f.name in self.env_cfg.rewards
    })
    imag_cfg = ImagineCfg(horizon=cfg.horizon, compile=cfg.compile,
                          sensor_grad=cfg.sensor_grad, weights=weights,
                          dt=self.env.step_dt)
    return Imagination(imag_cfg, self.consts, self.term.cfg.target, self.history,
                       self.env.device, self.terms, self.field)

  def driven_actions(self, actions, obs, iteration):
    cfg = self.cfg
    start, end = cfg.driver_fade
    fade = min(1.0, max(0.0, (end - iteration) / max(1, end - start)))
    n_moe = int(fade * cfg.driver_frac * cfg.num_envs) if self.driver is not None else 0
    n_exp = int(fade * cfg.expert_frac * cfg.num_envs) if self.expert is not None else 0
    moe = self.env_index >= cfg.num_envs - n_moe
    exp = (self.env_index >= cfg.num_envs - n_moe - n_exp) & ~moe
    if self.driver is not None:
      pose = self.pose()
      cmd = self.steer(self.env.step_dt, pose[:, :2], pose[:, 2], self.term.gen.center,
                       self.term.gen.chase_pos[:, :2], self.term.has_chase)
      actions = torch.where(moe.unsqueeze(-1), self.driver.act(cmd), actions)
    if n_exp:
      actions = torch.where(exp.unsqueeze(-1), self.expert.act(obs), actions)
    return actions

  def on_reset(self, done):
    if self.driver is not None:
      self.driver.reset(done)
      self.steer.resample(done, self.term.gen.center)

  def seed(self, env_ids, slots, buffer):
    return {
      "env": env_ids, "s": buffer.s[env_ids, slots], "pose": buffer.pose[env_ids, slots],
      "a_prev": buffer.a_prev[env_ids, slots], "obs": buffer.obs[env_ids, slots].float(),
      "task": buffer.task[env_ids, slots], "xi": self.context[env_ids],
      "obs_context": self.encoder_bias[env_ids],
    }

  @staticmethod
  def rarity(w):
    s = w["s"]
    contact = st.feet_world_z(s.reshape(-1, st.STATE_DIM)) < 0.035
    contact = contact.reshape(s.shape[0], -1, len(GO2_SPEC.foot_sites)).float()
    changes = (contact[:, 1:] - contact[:, :-1]).abs().sum((1, 2))
    airborne = (contact.sum(-1) == 0).float().sum(-1)
    ended = (w["valid"][:, -1] < 0.5).float() * 4.0
    return changes + airborne + ended

  @staticmethod
  def archive_worthy(w):
    s = w["s"]
    feet = st.feet_world_z(s.reshape(-1, st.STATE_DIM)).reshape(
      s.shape[0], -1, len(GO2_SPEC.foot_sites))
    flight = (feet >= 0.035).all(-1).float().sum(-1) >= 2
    hard_vz = s[:, :, st.V_YAW][..., 2].abs().amax(-1) > 0.6
    return flight | hard_vz | (w["valid"][:, -1] < 0.5)

  def metrics(self):
    wz = self.env.scene["robot"].data.root_link_ang_vel_b[:, 2].abs()
    return {"Metrics/fast_yaw": (wz > 3.0).float().mean()}

  def log_metrics(self, writer, iteration):
    rates = self.term.kind_hit_rate.tolist()
    result = {}
    for name, rate in zip(("ball", "low_bar", "high_bar"), rates):
      writer.add_scalar(f"Metrics/hit_rate/{name}", rate, iteration)
      result[f"Metrics/hit_rate/{name}"] = rate
    return result

  def checkpoint_metadata(self):
    action = self.env.action_manager.get_term(GO2_SPEC.action_term)
    return {
      "format_version": 1, "robot": GO2_SPEC.name, "backend": self.cfg.backend,
      "task": self.cfg.task, "state_terms": GO2_SPEC.state_layout.terms,
      "obs_terms": self.terms, "history": self.history, "dynamics": asdict(self.dynamics),
      "action_names": list(action._target_names),
      "state_frame": "base_height_above_ground;velocities_and_limbs_in_yaw_frame",
    }

  def checkpoint_terms(self, checkpoint):
    metadata = checkpoint.get("metadata")
    if metadata:
      current = self.checkpoint_metadata()
      for key in ("format_version", "robot", "backend", "history", "state_frame"):
        if metadata.get(key) != current[key]:
          raise ValueError(f"checkpoint {key} is incompatible with the current runtime")
      if tuple(map(tuple, metadata["state_terms"])) != GO2_SPEC.state_layout.terms:
        raise ValueError("checkpoint state layout is incompatible")
      if metadata.get("action_names") != current["action_names"]:
        raise ValueError("checkpoint action joint order is incompatible")
      return tuple(map(tuple, metadata["obs_terms"]))
    # Released legacy checkpoints have no schema; only these known layouts are accepted.
    dim = checkpoint["actor"]["norm.mean"].shape[0]
    layouts = {
      350: st.OBS_TERMS,
      430: st.OBS_TERMS + (("lidar", 16),),
      735: st.OBS_TERMS + (("height_scan", tr.SCAN_POINTS),),
    }
    if dim not in layouts or self.history != 5:
      raise ValueError("legacy checkpoint observation layout cannot be inferred safely")
    return layouts[dim]
