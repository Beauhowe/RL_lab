"""Deterministic actor and state-value network."""

import torch
from torch import nn
from supermimic_lab.models.normalization import RunningNorm
from supermimic_lab.models.mlp import mlp

class Actor(nn.Module):
  def __init__(self, obs_dim: int, hidden: tuple[int, ...], act_dim: int):
    super().__init__()
    self.norm = RunningNorm(obs_dim)
    self.net = mlp(obs_dim, hidden, act_dim, act=nn.ELU)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(obs))

class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.norm = RunningNorm(in_dim)
    self.net = mlp(in_dim, hidden, 1, act=nn.ELU)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x))
