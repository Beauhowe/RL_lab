"""Per-environment transition ring buffer; dimensions come from the experiment."""

import torch
from supermimic_lab.envs.specs import DynamicsSpec

class ReplayBuffer:
  def __init__(self, num_envs: int, length: int, obs_dim: int, device: str,
               *, spec: DynamicsSpec, task_dim: int):
    self.n, self.L, self.device = num_envs, length, device
    f = lambda *shape: torch.zeros(num_envs, length, *shape, device=device)  # noqa: E731
    self.s = f(spec.state_dim)
    self.pose = f(spec.pose_dim)
    self.a = f(spec.action_dim)
    self.a_prev = f(spec.action_dim)
    # Observations dominate memory; half precision is plenty for seeding rollouts.
    self.obs = torch.zeros(num_envs, length, obs_dim, device=device, dtype=torch.float16)
    self.relief = torch.zeros(num_envs, length, spec.terrain_dim, device=device, dtype=torch.float16)
    self.task = f(task_dim)
    self.done = f()
    self.ptr = 0
    self.size = 0

  def add(self, s, pose, a, a_prev, obs, task_state, done, relief) -> None:
    i = self.ptr
    self.s[:, i], self.pose[:, i], self.a[:, i], self.a_prev[:, i] = s, pose, a, a_prev
    self.relief[:, i] = relief.half()
    self.obs[:, i], self.task[:, i], self.done[:, i] = obs.half(), task_state, done.float()
    self.ptr = (self.ptr + 1) % self.L
    self.size = min(self.size + 1, self.L)

  def _phys(self, logical: torch.Tensor) -> torch.Tensor:
    start = (self.ptr - self.size) % self.L
    return (start + logical) % self.L

  def sample_windows(self, batch: int, k: int):
    """Random k-step windows: (env_ids, slot indices (B, k+1), valid (B, k))."""
    assert self.size > k + 1, "buffer too small for windows"
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t0 = torch.randint(0, self.size - k, (batch,), device=self.device)
    idx = self._phys(t0.unsqueeze(-1) + torch.arange(k + 1, device=self.device))
    done = self.done[env.unsqueeze(-1), idx[:, :k]]
    # Transition j (slot j -> j+1) is valid if no slot 0..j ended an episode.
    valid = torch.cumprod(1.0 - done, dim=1)
    return env, idx, valid

  def sample_states(self, batch: int):
    """Random (env_ids, slot) pairs to seed imagined rollouts."""
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t = self._phys(torch.randint(0, self.size, (batch,), device=self.device))
    return env, t
