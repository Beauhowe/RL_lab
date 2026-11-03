"""Serializable tensor layouts shared by collectors, models and checkpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorLayout:
  """Ordered named fields; ordering is part of the checkpoint contract."""

  terms: tuple[tuple[str, int], ...]

  def __post_init__(self):
    names = [name for name, _ in self.terms]
    if len(names) != len(set(names)) or any(dim <= 0 for _, dim in self.terms):
      raise ValueError("layout fields must have unique names and positive dimensions")

  @property
  def dim(self) -> int:
    return sum(dim for _, dim in self.terms)

  def slice(self, name: str) -> slice:
    offset = 0
    for field, dim in self.terms:
      if field == name:
        return slice(offset, offset + dim)
      offset += dim
    raise KeyError(name)


@dataclass(frozen=True)
class DynamicsSpec:
  """Dimensions for residual state dynamics plus a separate pose delta.

  terrain_dim stores ground relief, converted to height-above-surface using
  height_index. Other dynamics representations can supply zero terrain inputs.
  """

  state_dim: int
  action_dim: int
  context_dim: int
  terrain_dim: int
  pose_dim: int = 3
  height_index: int = 0

  def __post_init__(self):
    if min(self.state_dim, self.action_dim, self.pose_dim) <= 0:
      raise ValueError("state, action and pose dimensions must be positive")
    if min(self.context_dim, self.terrain_dim) < 0:
      raise ValueError("context and terrain dimensions cannot be negative")
    if not 0 <= self.height_index < self.state_dim:
      raise ValueError("height_index must refer to a state component")


@dataclass(frozen=True)
class RobotSpec:
  """Robot identity and named kinematic points, independent of a simulator."""

  name: str
  state_layout: TensorLayout
  action_dim: int
  limb_bodies: tuple[tuple[str, tuple[str, ...]], ...]
  foot_sites: tuple[str, ...]
  foot_geoms: tuple[str, ...]
  base_body: str = "base"
  action_term: str = "joint_pos"
