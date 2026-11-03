"""Task-specific static obstacles layered onto the shared terrain height field."""

from __future__ import annotations
import torch
from supermimic_lab.envs.terrain import (
  SCAN_X, SCAN_Y, SCAN_POINTS, RELIEF_CLIP, HeightField, scan_offsets,
  build_height_field, env_height_field, ground, scan_world,
)
from . import obstacles as ob

OBSTACLE_EDGE = 0.005

def surface(xy: torch.Tensor, obst: torch.Tensor | None, field: HeightField | None) -> torch.Tensor:
  """Height (N, P) of what a foot or body meets at points (N, P, 2): terrain, or
  the top of a tall obstacle (smoothed edge)."""
  h = ground(xy, field)
  if obst is not None:
    inside = torch.sigmoid(-ob.sdf(xy, obst) / OBSTACLE_EDGE)  # (N, P, K)
    h = h + ob.OBSTACLE_HEIGHT * inside.amax(-1)
  return h

def relief(base_xy: torch.Tensor, yaw: torch.Tensor, field: HeightField | None,
           obst_flat: torch.Tensor | None = None, grad: bool = False) -> torch.Tensor:
  """Ground under the base minus the surface at each scan point (N, P).
  Not differentiated unless grad=True."""
  if not grad:
    with torch.no_grad():
      return relief(base_xy, yaw, field, obst_flat, grad=True)
  obst = ob.unpack(obst_flat) if obst_flat is not None else None
  under = ground(base_xy, field).unsqueeze(-1)
  r = under - surface(scan_world(base_xy, yaw), obst, field)
  return r.clamp(*RELIEF_CLIP)
