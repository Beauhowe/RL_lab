"""PPO settings; no world model or SuperMimic configuration is constructed."""

from dataclasses import dataclass
from supermimic_lab.runners.config import ExperimentCfg


@dataclass(kw_only=True)
class PpoTrainCfg(ExperimentCfg):
  algorithm: str = "ppo"
  task: str = "Go2-Locomotion-Flat"
  num_envs: int = 2048
  iterations: int = 1500
  run_name: str = "go2_ppo"
  log_root: str = "logs/ppo"
  save_interval: int = 100
  num_steps_per_env: int = 24
  learning_rate: float = 1e-3
  learning_epochs: int = 5
  minibatches: int = 4
  gamma: float = 0.99
  lam: float = 0.95
  clip_param: float = 0.2
  entropy_coef: float = 0.005
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  init_noise_std: float = 0.5
  clip_actions: float = 10.0
  observation_normalization: bool = True
  # Explicit environment control steps, independent of PPO's rollout length.
  terrain_curriculum_steps: int = 12_000
  command_curriculum_steps: int = 15_000
  headless: bool = True

  def __post_init__(self):
    super().__post_init__()
    if self.algorithm != "ppo":
      raise ValueError("PpoTrainCfg requires algorithm=ppo")
    if min(self.num_steps_per_env, self.learning_epochs, self.minibatches) <= 0:
      raise ValueError("PPO rollout sizes must be positive")
    if self.minibatches > self.num_envs * self.num_steps_per_env:
      raise ValueError("PPO minibatches exceed rollout transitions")
    if not 0 < self.gamma <= 1 or not 0 <= self.lam <= 1 or not 0 < self.clip_param < 1:
      raise ValueError("invalid PPO gamma, lambda or clipping range")
    if min(self.learning_rate, self.init_noise_std, self.clip_actions, self.max_grad_norm) <= 0:
      raise ValueError("PPO learning rate, action scales and gradient limit must be positive")
    if min(self.terrain_curriculum_steps, self.command_curriculum_steps) <= 0:
      raise ValueError("curriculum steps must be positive")
