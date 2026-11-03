"""Observation history utilities, independent of robot and task."""

import torch

def push_history(hist: torch.Tensor, frame: dict[str, torch.Tensor], h: int,
                 terms: tuple[tuple[str, int], ...]) -> torch.Tensor:
  """Shift each term's flattened history (oldest->newest) and append the new frame."""
  out, i = [], 0
  for name, dim in terms:
    block = hist[:, i : i + dim * h]
    out.append(torch.cat([block[:, dim:], frame[name]], dim=-1))
    i += dim * h
  return torch.cat(out, dim=-1)
