"""Yaw-frame coordinate transforms."""

import torch

def yaw_rotate_inverse(vec: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """Express world-frame vectors (N, 3) in the yaw frame given heading (N,)."""
  c, s = heading.cos(), heading.sin()
  x = c * vec[:, 0] + s * vec[:, 1]
  y = -s * vec[:, 0] + c * vec[:, 1]
  return torch.stack([x, y, vec[:, 2]], dim=-1)

def pose_delta(pose0: torch.Tensor, pose1: torch.Tensor) -> torch.Tensor:
  """Planar motion from pose0 to pose1, expressed in pose0's yaw frame."""
  c, s = pose0[:, 2].cos(), pose0[:, 2].sin()
  dx, dy = pose1[:, 0] - pose0[:, 0], pose1[:, 1] - pose0[:, 1]
  dyaw = torch.atan2((pose1[:, 2] - pose0[:, 2]).sin(), (pose1[:, 2] - pose0[:, 2]).cos())
  return torch.stack([c * dx + s * dy, -s * dx + c * dy, dyaw], dim=-1)

def apply_pose_delta(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
  c, s = pose[:, 2].cos(), pose[:, 2].sin()
  x = pose[:, 0] + c * delta[:, 0] - s * delta[:, 1]
  y = pose[:, 1] + s * delta[:, 0] + c * delta[:, 1]
  return torch.stack([x, y, pose[:, 2] + delta[:, 2]], dim=-1)
