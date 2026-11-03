"""Differentiable return accumulation and TD(lambda) targets.

Subclasses implement task state restoration, rollout constants, _step and
critic_input; the accumulation does not depend on robot topology or a simulator.
"""

import torch


class DifferentiableImagination:
  def rollout(self, wm, actor, critic, seed: dict, hint_weight: float = 0.0):
    """Imagine H steps from replay seeds. Returns differentiable objective and critic data."""
    cfg = self.cfg
    B = seed["s"].shape[0]
    gst = self.restore_task_state(seed["task"])
    s, pose, a_prev, hist = seed["s"], seed["pose"], seed["a_prev"], seed["obs"]
    xi, obs_context = seed["xi"], seed["obs_context"]
    dt = torch.full((B,), cfg.dt, device=self.device)
    k = self.rollout_constants(seed, hint_weight)

    surv = torch.ones(B, device=self.device)
    ret = torch.zeros(B, device=self.device)
    disc = 1.0
    rewards, survs, crit_in = [], [], []
    for _ in range(cfg.horizon):
      survs.append(surv)
      s, pose, a_prev, hist, gst, r, d, x = self._step_fn(
        s, pose, a_prev, hist, gst, xi, obs_context, k, dt, actor, wm
      )
      crit_in.append(x)
      ret = ret + disc * surv * r
      rewards.append(r)
      surv = surv * (1.0 - d)
      disc *= cfg.gamma
    x_H = self.critic_input(s, pose, gst, xi)
    v_H = critic(x_H).squeeze(-1)
    objective = ret + disc * surv * v_H
    return objective, {
      "rewards": torch.stack(rewards, 1),  # (B, H)
      "surv": torch.stack(survs + [surv], 1),  # (B, H+1)
      "critic_in": torch.stack(crit_in + [x_H], 1),  # (B, H+1, D)
    }

  @torch.no_grad()
  def gae_targets(self, critic, data) -> torch.Tensor:
    """TD(lambda) returns over the imagined steps (SuperMimic Eq. 3)."""
    cfg = self.cfg
    r = data["rewards"].detach()
    surv = data["surv"].detach()
    v = critic(data["critic_in"].detach()).squeeze(-1)  # (B, H+1)
    H = r.shape[1]
    targets = torch.zeros_like(r)
    nxt = v[:, H]
    for t in reversed(range(H)):
      cont = surv[:, t + 1] / surv[:, t].clamp(min=1e-6)  # 1 - d_{t+1}
      nxt = r[:, t] + cfg.gamma * cont * ((1 - cfg.lam) * v[:, t + 1] + cfg.lam * nxt)
      targets[:, t] = nxt
    return targets
