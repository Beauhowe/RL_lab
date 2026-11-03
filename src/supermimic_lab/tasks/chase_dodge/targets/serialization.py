"""Packing of chase/dodge task state for replay and imagination."""

import torch
from .generator import BOOL_FIELDS, STATE_DIMS

GEN_DIM = sum(STATE_DIMS.values())

def pack_generator(gen) -> torch.Tensor:
  parts = [getattr(gen, name).float().reshape(-1, dim) for name, dim in STATE_DIMS.items()]
  return torch.cat(parts, dim=-1)

def unpack_generator(gen, packed: torch.Tensor) -> torch.Tensor:
  """Write packed generator state into `gen` (batch must match)."""
  i = 0
  for name, dim in STATE_DIMS.items():
    x = packed[:, i : i + dim]
    if name in BOOL_FIELDS:
      x = x[:, 0] > 0.5
    elif dim == 1:
      x = x[:, 0]
    setattr(gen, name, x.clone())
    i += dim
