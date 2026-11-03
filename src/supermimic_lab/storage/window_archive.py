"""Long-lived archive of model-training windows."""

import torch
from supermimic_lab.envs.specs import DynamicsSpec

WINDOW_KEYS = ("s", "pose", "a", "a_prev", "xi", "valid", "relief")

class WindowArchive:
  """Ring store of k-step world-model windows with rare events (contact changes,
  flight, falls). Unlike the replay buffer it is not overwritten every few
  seconds of sim time, so the world model keeps seeing jumps and landings."""

  def __init__(self, capacity: int, k: int, context_dim: int, device: str, *, spec: DynamicsSpec):
    self.cap, self.device = capacity, device
    z = lambda *shape: torch.zeros(capacity, *shape, device=device)  # noqa: E731
    self.data = {
      "s": z(k + 1, spec.state_dim), "pose": z(k + 1, spec.pose_dim), "a": z(k, spec.action_dim),
      "a_prev": z(k, spec.action_dim), "xi": z(context_dim), "valid": z(k),
      "relief": torch.zeros(capacity, k, spec.terrain_dim, device=device, dtype=torch.float16),
    }
    self.ptr = 0
    self.size = 0

  def add(self, windows: dict[str, torch.Tensor]) -> None:
    n = windows["s"].shape[0]
    if n == 0:
      return
    n = min(n, self.cap)
    idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
    for k in WINDOW_KEYS:
      self.data[k][idx] = windows[k][:n]
    self.ptr = (self.ptr + n) % self.cap
    self.size = min(self.size + n, self.cap)

  def sample(self, batch: int) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, self.size, (batch,), device=self.device)
    return {k: v[idx] for k, v in self.data.items()}
