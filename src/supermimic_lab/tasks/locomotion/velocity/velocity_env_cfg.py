"""Go2 velocity tracking on flat ground and mixed rough terrain."""

from dataclasses import asdict
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from supermimic_lab.assets.robots.unitree_go2.env_cfg import configure_env
from supermimic_lab.envs.terrain_style import clean_look
from .state import OBS_TERMS, HISTORY
from .mdp.commands import VelocityCommandCfg
from .mdp import rewards, observations, curricula
from .rewards import RewardWeights
from .terrains import make_terrain


def locomotion_env_cfg(*, rough=False, play=False):
  cfg = make_velocity_env_cfg()
  configure_env(cfg)
  if rough:
    cfg.scene.terrain = make_terrain()
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    clean_look(cfg.scene.terrain)
  cfg.commands = {"twist": VelocityCommandCfg(resampling_time_range=(3.0, 6.0), rough=rough)}
  original = cfg.observations["actor"].terms
  terms = {name: original[name] for name, _ in OBS_TERMS if name != "height_scan"}
  terms["height_scan"] = ObservationTermCfg(
    func=observations.height_scan, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02))
  cfg.observations = {"actor": ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=not play, history_length=HISTORY)}
  weights = asdict(RewardWeights(terminated=-400.0 if rough else -200.0))
  cfg.rewards = {
    name: RewardTermCfg(func=rewards.component, weight=weight, params={"name": name})
    for name, weight in weights.items() if name != "terminated"
  }
  cfg.rewards["terminated"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=weights["terminated"])
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fell_over": TerminationTermCfg(func=rewards.fell_over),
  }
  cfg.curriculum = {}
  if rough:
    cfg.terminations["tile_exit"] = TerminationTermCfg(func=rewards.tile_exit, time_out=True)
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=curricula.terrain_levels, params={"steps": 12_000, "play": play})
  cfg.viewer.distance, cfg.viewer.elevation = 3.0, -20.0
  if play:
    cfg.episode_length_s = int(1e9)
    for sensor in cfg.scene.sensors:
      sensor.debug_vis = False
  return cfg
