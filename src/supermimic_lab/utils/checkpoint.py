"""Explicit policy and dynamics warm-start transformations."""

import torch
from supermimic_lab.models.actor_critic import Actor
from supermimic_lab.models.world_model import WorldModel

@torch.no_grad()
def expand_actor_new_terms(old: dict, actor: Actor, old_terms, new_terms, history: int) -> None:
  """Warm-start an actor whose observation gained whole terms (e.g. a height scan),
  exactly: old terms keep their weights and normaliser statistics, new terms start
  with zero weight (their statistics are re-estimated as data comes in)."""
  if any(dict(new_terms).get(name) != dim for name, dim in old_terms):
    raise ValueError("warm starts require all old observation fields with unchanged dimensions")
  sd = actor.state_dict()
  for k, v in old.items():
    if k in sd and sd[k].shape == v.shape:
      sd[k] = v.clone()
  w_old, mu_old, var_old = old["net.0.weight"], old["norm.mean"], old["norm.var"]
  w, mu, var = sd["net.0.weight"], sd["norm.mean"], sd["norm.var"]
  w.zero_()
  mu.zero_()
  var.fill_(1.0)
  old_off, o = {}, 0
  for name, dim in old_terms:
    old_off[name] = (o, dim * history)
    o += dim * history
  assert o == w_old.shape[1], (o, w_old.shape)
  n = 0
  for name, dim in new_terms:
    size = dim * history
    if name in old_off:
      o, osize = old_off[name]
      assert osize == size, name
      w[:, n : n + size] = w_old[:, o : o + size]
      mu[n : n + size], var[n : n + size] = mu_old[o : o + size], var_old[o : o + size]
    n += size
  assert n == w.shape[1]
  actor.load_state_dict(sd)

@torch.no_grad()
def expand_world_model_scan(old: dict, wm: WorldModel) -> bool:
  """Warm-start a world model with ground-scan inputs from one without, exactly:
  the scan inputs (appended last) start with zero weight. False if the old model
  differs in any other way."""
  sd = wm.state_dict()
  w_old, w_new = old.get("net.0.weight"), sd["net.0.weight"]
  if w_old is None or w_old.shape[0] != w_new.shape[0] or w_new.shape[1] - w_old.shape[1] != wm.spec.terrain_dim:
    return False
  for k, v in old.items():
    if k not in sd or (k not in ("net.0.weight", "in_norm.mean", "in_norm.var")
                       and sd[k].shape != v.shape):
      return False
  d = w_old.shape[1]
  for k, v in old.items():
    sd[k] = v.clone() if k not in ("net.0.weight", "in_norm.mean", "in_norm.var") else sd[k]
  sd["net.0.weight"].zero_()
  sd["net.0.weight"][:, :d] = w_old
  sd["in_norm.mean"][:d], sd["in_norm.var"][:d] = old["in_norm.mean"], old["in_norm.var"]
  sd["in_norm.mean"][d:], sd["in_norm.var"][d:] = 0.3, 0.04
  wm.load_state_dict(sd)
  return True
