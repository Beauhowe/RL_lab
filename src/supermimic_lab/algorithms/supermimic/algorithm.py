"""SuperMimic parameter updates, independent of robot and simulator imports."""

import torch
from .config import TrainCfg

def clip_grads(params, global_norm: float, elem: float) -> torch.Tensor:
  params = [p for p in params if p.grad is not None]
  norm = torch.nn.utils.clip_grad_norm_(params, global_norm)
  for p in params:
    p.grad.clamp_(-elem, elem)
  return norm

def lr_at(cfg: TrainCfg, it: int) -> float:
  import math

  t = min(1.0, it / cfg.lr_decay_iterations)
  frac = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t))
  return cfg.lr * frac


class SuperMimic:
  """Update phases operating on runner-owned models, buffers and task runtime."""

  def _gather(self, env, idx, valid) -> dict[str, torch.Tensor]:
    b, k = self.buf, self.cfg.wm_window
    e = env.unsqueeze(-1)
    return {
      "s": b.s[e, idx], "pose": b.pose[e, idx], "a": b.a[e, idx[:, :k]],
      "a_prev": b.a_prev[e, idx[:, :k]], "xi": self.xi[env], "valid": valid,
      "relief": b.relief[e, idx[:, :k]],
    }

  def _wm_batch(self) -> dict[str, torch.Tensor]:
    """Uniform windows, rare windows from the buffer, and archived rare windows."""
    cfg, b = self.cfg, self.buf
    n_prio = int(cfg.wm_batch * cfg.wm_prioritized_frac)
    n_arch = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size > 0 else 0
    n_uni = cfg.wm_batch - n_prio - n_arch
    parts = [self._gather(*b.sample_windows(n_uni, cfg.wm_window))]
    cand = self._gather(*b.sample_windows(4 * max(n_prio, 1), cfg.wm_window))
    score = self.runtime.rarity(cand)
    keep = self.runtime.archive_worthy(cand) & (torch.rand_like(score) < cfg.archive_keep_prob)
    self.archive.add({k: v[keep] for k, v in cand.items()})
    top = (score + 1e-3 * torch.rand_like(score)).topk(n_prio).indices
    parts.append({k: v[top] for k, v in cand.items()})
    if n_arch:
      parts.append(self.archive.sample(n_arch))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

  def fit_world_model(self) -> float:
    cfg = self.cfg
    k = cfg.wm_window
    losses = []
    for _ in range(cfg.wm_updates):
      w = self._wm_batch()
      pose_tgt = torch.stack(
        [self.runtime.pose_delta(w["pose"][:, j], w["pose"][:, j + 1]) for j in range(k)], 1
      )
      loss = self.wm.loss(
        w["s"][:, 0], w["a"], w["a_prev"], w["xi"], w["relief"].float(), w["s"][:, 1:],
        pose_tgt, w["valid"],
      )
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = clip_grads(self.wm.parameters(), cfg.grad_clip, float("inf"))
      if bool(torch.isfinite(gn)):
        self.opt_wm.step()
      else:
        self.nonfinite_steps += 1
      losses.append(loss.item())
    return sum(losses) / len(losses)

  def improve_policy(self) -> dict[str, float]:
    cfg, b = self.cfg, self.buf
    env, t = b.sample_states(cfg.imag_batch)
    seed = self.runtime.seed(env, t, b)
    for p in self.wm.parameters():
      p.requires_grad_(False)
    hint = cfg.dodge_shaping * max(0.0, 1.0 - self.it / max(1, cfg.dodge_shaping_decay_iterations))
    objective, data = self.imag.rollout(self.wm, self.actor, self.critic, seed, hint)
    loss = -objective.mean()
    self.opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = clip_grads(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_elem)
    finite = bool(torch.isfinite(gnorm)) and bool(torch.isfinite(loss))
    if not finite:
      self.nonfinite_steps += 1
    if self.it > self.actor_frozen_until and finite:
      self.opt_actor.step()
    for p in self.wm.parameters():
      p.requires_grad_(True)

    # Critic on the imagined transitions.
    targets = self.imag.gae_targets(self.critic, data)
    x = data["critic_in"][:, :-1].detach().reshape(-1, data["critic_in"].shape[-1])
    y = targets.reshape(-1)
    self.critic.norm.update(x)
    n = x.shape[0]
    mb = n // cfg.critic_minibatches
    c_losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.dev)
      for i in range(cfg.critic_minibatches):
        sel = perm[i * mb : (i + 1) * mb]
        c_loss = (self.critic(x[sel]).squeeze(-1) - y[sel]).square().mean()
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        gn = clip_grads(self.critic.parameters(), cfg.grad_clip, float("inf"))
        if bool(torch.isfinite(gn)):
          self.opt_critic.step()
        c_losses.append(c_loss.item())
    return {
      "imagined_return": objective.mean().item(),
      "imagined_reward": data["rewards"].mean().item(),
      "imagined_survival": data["surv"][:, -1].mean().item(),
      "actor_grad_norm": float(gnorm),
      "critic_loss": sum(c_losses) / len(c_losses),
    }

