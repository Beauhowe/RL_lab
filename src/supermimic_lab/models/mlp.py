"""Feed-forward network construction."""

from torch import nn

def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU) -> nn.Sequential:
  layers, d = [], in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)
