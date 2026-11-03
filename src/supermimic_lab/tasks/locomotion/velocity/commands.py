"""Batched velocity commands shared by simulation and imagined rollouts."""

from dataclasses import dataclass
import torch


@dataclass(kw_only=True)
class VelocityCfg:
  period: tuple[float, float] = (3.0, 6.0)
  initial_x: tuple[float, float] = (-0.3, 0.8)
  final_x: tuple[float, float] = (-1.0, 2.0)
  initial_y: tuple[float, float] = (-0.2, 0.2)
  final_y: tuple[float, float] = (-0.6, 0.6)
  initial_yaw: tuple[float, float] = (-0.4, 0.4)
  final_yaw: tuple[float, float] = (-1.0, 1.0)
  standing_fraction: float = 0.1
  forward_fraction: float = 0.6
  curriculum_steps: int = 15_000
  control_dt: float = 0.02
  fixed: tuple[float, float, float] | None = None


def sample_commands(count, device, progress, cfg: VelocityCfg):
  if cfg.fixed is not None:
    return torch.tensor(cfg.fixed, device=device).expand(count, 3).clone()
  progress = torch.as_tensor(progress, device=device).expand(count).clamp(0, 1)
  parts = []
  for start, end in ((cfg.initial_x, cfg.final_x), (cfg.initial_y, cfg.final_y),
                     (cfg.initial_yaw, cfg.final_yaw)):
    low = start[0] + progress * (end[0] - start[0])
    high = start[1] + progress * (end[1] - start[1])
    parts.append(low + torch.rand(count, device=device) * (high - low))
  command = torch.stack(parts, -1)
  choice = torch.rand(count, device=device)
  forward = choice < cfg.standing_fraction + cfg.forward_fraction
  command = torch.stack([torch.where(forward, command[:, 0].abs().clamp(min=0.1), command[:, 0]),
                         torch.where(forward, 0.0, command[:, 1]),
                         torch.where(forward, 0.0, command[:, 2])], -1)
  return torch.where((choice < cfg.standing_fraction).unsqueeze(-1), 0.0, command)


def advance_commands(task_state, dt, cfg: VelocityCfg):
  """Packed layout: body-yaw-frame vx/vy/yaw_rate, resampling timer, progress."""
  command, timer, progress = task_state[:, :3], task_state[:, 3], task_state[:, 4]
  progress = (progress + dt / cfg.control_dt / max(1, cfg.curriculum_steps)).clamp(0, 1)
  timer = timer - dt
  sampled = sample_commands(task_state.shape[0], task_state.device, progress, cfg)
  period = cfg.period[0] + torch.rand_like(timer) * (cfg.period[1] - cfg.period[0])
  command = torch.where((timer <= 0).unsqueeze(-1), sampled, command)
  timer = torch.where(timer <= 0, period, timer)
  return torch.cat([command, timer[:, None], progress[:, None]], -1)
