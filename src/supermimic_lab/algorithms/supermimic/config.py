"""SuperMimic experiment settings; importable without a simulator or PyTorch."""

from dataclasses import dataclass
from supermimic_lab.runners.config import ExperimentCfg

@dataclass(kw_only=True)
class TrainCfg(ExperimentCfg):
  buffer_len: int = 1000
  warmup_iterations: int = 300
  """World-model-only iterations before policy updates start."""
  collect_steps: int = 1
  action_noise: float = 0.1
  explore_frac: float = 0.25
  """Fraction of envs collecting with larger, temporally correlated (OU) action
  noise, so the world model sees more varied motion than the current policy."""
  explore_noise: float = 0.3
  """Stationary std of the exploration OU noise (normalized action units)."""
  explore_theta: float = 2.0
  """Mean reversion of the exploration OU noise [1/s]."""
  wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_updates: int = 2
  wm_prioritized_frac: float = 0.25
  """Fraction of each world-model batch drawn from current windows with contact
  changes (touchdown / liftoff), flight or falls, where the model is least accurate."""
  wm_archive_frac: float = 0.25
  """Fraction of each world-model batch drawn from the rare-event archive."""
  archive_size: int = 100_000
  """Windows kept in the rare-event archive (jumps, landings, falls)."""
  archive_keep_prob: float = 0.02
  """Fraction of the rare candidate windows stored, so the archive turns over
  slowly and keeps old rare events."""
  wm_rewarmup_iterations: int = 1500
  """When resuming into a world model of a different shape, fit it this long
  before the policy moves again."""
  lr: float = 2e-4
  actor_lr_scale: float = 1.0
  """The actor's learning rate relative to lr. Fine-tuning on terrain wants smaller
  actor steps: at the full rate the actor drifts into states the world model
  predicts badly (falls rise once the ramp-up completes)."""
  lr_final_frac: float = 0.03
  lr_decay_iterations: int = 30_000
  horizon: int = 32
  imag_batch: int = 8192
  critic_epochs: int = 5
  critic_minibatches: int = 4
  grad_clip: float = 1.0
  grad_clip_elem: float = 0.1
  compile: bool = True
  """torch.compile one imagined step (cuts Python kernel-launch overhead ~1.5x)."""
  dodge_shaping: float = 20.0
  """Initial weight of the jump/duck hint (training wheels for the bars)."""
  dodge_shaping_decay_iterations: int = 20_000
  """The hint weight decays linearly to 0 over this many iterations."""
  target_curriculum_iterations: int = 15_000
  """Iterations over which the targets reach full difficulty. The env-side
  curriculum counts env steps, and this loop takes one env step per iteration
  (PPO takes 24), so it is set here explicitly."""
  terrain_curriculum_iterations: int = 12_000
  """Iterations over which the harder terrain rows open up (terrain task)."""
  command_curriculum_iterations: int = 15_000
  """Iterations over which locomotion commands reach their full velocity ranges."""
  sensor_grad: bool = True
  """See ImagineCfg.sensor_grad."""
  driver_policy: str = ""
  """MoE-CTS TorchScript policy that drives a fraction of the envs while
  collecting, so the world model sees terrain from the start (supermimic/driver.py)."""
  driver_frac: float = 0.25
  """Fraction of envs (the last ones; exploration noise uses the first) it drives."""
  expert_policy: str = ""
  """One of our checkpoints (e.g. a flat-ground chase-and-dodge policy) that drives
  another fraction of the envs, so the world model sees its skills (jumps, ducks)
  in this task. Its observation must be a prefix of this task's."""
  expert_frac: float = 0.0
  driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)
  """Both driven fractions fade linearly to zero between these iterations."""
  driver_vmin: float = 0.5
  driver_vmax: float = 3.0
  """Per-env top speed of the driver's commands is drawn from [vmin, vmax] [m/s]."""
  critic_warmup_iterations: int = 500
  """After resuming with a freshly initialised critic, train only the critic for
  this many policy iterations before the actor moves again."""
  actor_ramp_iterations: int = 1000
  """Whenever the actor starts (or restarts after a rollback) moving, its learning
  rate ramps up linearly over this many iterations, so the world model sees the
  new behaviour before the actor can exploit its errors."""
  reset_critic: bool = False
  """With --resume: start the critic afresh (e.g. after changing reward weights) and
  keep the actor still until it has warmed up."""
  collapse_falls: float = 3.0
  """Collapse guard: if the logged fall count exceeds this twice in a row, reload
  the last checkpoint and halve the learning rate."""
  log_interval: int = 50
  log_format: str = "pretty"
  """Terminal display: grouped pretty report or compact single-line output."""

  def __post_init__(self):
    super().__post_init__()
    if self.log_format not in ("pretty", "compact"):
      raise ValueError("log_format must be pretty or compact")
    positive = ("num_envs", "buffer_len", "iterations", "collect_steps", "wm_window",
                "wm_batch", "wm_updates", "archive_size", "horizon", "imag_batch",
                "critic_epochs", "critic_minibatches", "lr_decay_iterations",
                "log_interval", "save_interval")
    if any(getattr(self, name) <= 0 for name in positive):
      raise ValueError("training sizes, iteration counts and intervals must be positive")
    if self.buffer_len <= self.wm_window + 1:
      raise ValueError("buffer_len must exceed wm_window + 1")
    for name in ("explore_frac", "wm_prioritized_frac", "wm_archive_frac",
                 "archive_keep_prob", "driver_frac", "expert_frac"):
      if not 0 <= getattr(self, name) <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    if self.wm_prioritized_frac + self.wm_archive_frac >= 1:
      raise ValueError("world-model sampling must reserve a positive uniform fraction")
    if self.driver_frac + self.expert_frac > 1:
      raise ValueError("driver_frac + expert_frac cannot exceed 1")
    if self.critic_minibatches > self.imag_batch * self.horizon:
      raise ValueError("critic_minibatches exceeds the number of imagined transitions")
    if len(self.driver_fade) != 2 or self.driver_fade[0] > self.driver_fade[1]:
      raise ValueError("driver_fade requires an ordered pair of iterations")
    if any(not layers or min(layers) <= 0
           for layers in (self.wm_hidden, self.actor_hidden, self.critic_hidden)):
      raise ValueError("network hidden layer sizes must be positive")
