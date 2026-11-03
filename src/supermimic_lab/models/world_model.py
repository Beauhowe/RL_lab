"""Learned residual dynamics with explicit state and action dimensions."""

import torch
from torch import nn
from supermimic_lab.envs.specs import DynamicsSpec
from supermimic_lab.models.normalization import RunningNorm
from supermimic_lab.models.mlp import mlp

class WorldModel(nn.Module):
  """s' = f(s, a, a_prev, xi, scan). The scan is the base height above each point
  of the local ground grid (terrain and obstacles, see target/terrain.py), so
  stepping onto a stair or pushing into a wall is predictable."""

  def __init__(self, hidden: tuple[int, ...] = (1024, 1024, 1024), *, spec: DynamicsSpec):
    super().__init__()
    self.spec = spec
    in_dim = spec.state_dim + 2 * spec.action_dim + spec.context_dim + spec.terrain_dim
    out_dim = spec.state_dim + spec.pose_dim
    self.net = mlp(in_dim, hidden, out_dim)
    self.in_norm = RunningNorm(in_dim)
    # Normalizes the per-step change of s and the pose delta.
    self.out_norm = RunningNorm(out_dim)

  def _inputs(self, s, a, a_prev, xi, relief):
    return torch.cat([s, a, a_prev, xi, s[:, self.spec.height_index:self.spec.height_index + 1] + relief], dim=-1)

  def update_stats(self, s, a, a_prev, xi, relief, s_next, pose_delta) -> None:
    self.in_norm.update(self._inputs(s, a, a_prev, xi, relief))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], dim=-1))

  def forward(self, s, a, a_prev, xi, relief):
    """relief (B, P): ground under the base minus the ground at each scan point.
    Returns (s_next, pose_delta)."""
    y = self.out_norm.unnormalize(self.net(self.in_norm(self._inputs(s, a, a_prev, xi, relief))))
    return s + y[:, : self.spec.state_dim], y[:, self.spec.state_dim :]

  def loss(self, s0, actions, actions_prev, xi, reliefs, s_targets, pose_targets, valid):
    """k-step autoregressive loss in normalized output units.

    s0 (B, S); actions, actions_prev (B, k, 12); xi (B, C); reliefs (B, k, P) at
    the recorded poses; s_targets (B, k, S); pose_targets (B, k, 3); valid (B, k)
    masks steps after a reset inside the window.
    """
    k = actions.shape[1]
    scale = self.out_norm.var.sqrt() + 1e-6
    s = s0
    total, denom = 0.0, valid.sum().clamp(min=1.0)
    for j in range(k):
      s_pred, pose_pred = self(s, actions[:, j], actions_prev[:, j], xi, reliefs[:, j])
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], dim=-1)
      total = total + ((err / scale) ** 2).mean(-1).mul(valid[:, j]).sum()
      s = s_pred
    return total / denom
