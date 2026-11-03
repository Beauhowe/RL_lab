"""Experiment identity shared by algorithms, without simulator imports."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class ExperimentCfg:
  algorithm: str = "supermimic"
  backend: str = "mjlab"
  robot: str = "unitree_go2"
  task: str = "Go2-Chase-Terrain"
  device: str = "cuda:0"
  num_envs: int = 4096
  iterations: int = 40_000
  actor_hidden: tuple[int, ...] = (512, 256, 128)
  critic_hidden: tuple[int, ...] = (512, 256, 128)
  save_interval: int = 1000
  seed: int = 0
  run_name: str = "supermimic"
  log_root: str = "logs/supermimic"
  resume: str = ""
  continue_run: bool = False

  def __post_init__(self):
    if min(self.num_envs, self.iterations, self.save_interval) <= 0:
      raise ValueError("num_envs, iterations and save_interval must be positive")
    for layers in (self.actor_hidden, self.critic_hidden):
      if not layers or min(layers) <= 0:
        raise ValueError("network hidden layer sizes must be positive")
