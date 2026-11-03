"""SuperMimic experiment orchestration through a registered task runtime.

One loop, three phases per iteration (SuperMimic Alg. A.1):
  1. Collect: the deterministic actor plus Gaussian action noise steps the mjlab
     env; transitions go into a fixed-size replay buffer.
  2. Fit world model: k-step autoregressive windows from the buffer.
  3. Improve policy: BPTT of the imagined H-step return through the frozen
     world model, bootstrapped by a critic fit to TD(lambda) targets.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from supermimic_lab.algorithms.supermimic.config import TrainCfg
from supermimic_lab.algorithms.supermimic.algorithm import SuperMimic, lr_at
from supermimic_lab.models.actor_critic import Actor, Critic
from supermimic_lab.models.world_model import WorldModel
from supermimic_lab.storage.replay_buffer import ReplayBuffer
from supermimic_lab.storage.window_archive import WindowArchive
from supermimic_lab.tasks.registry import make_runtime
from supermimic_lab.utils.checkpoint import expand_actor_new_terms, expand_world_model_scan
from .terminal import TerminalLogger, training_stage


class SuperMimicRunner(SuperMimic):
  def __init__(self, cfg: TrainCfg):
    from torch.utils.tensorboard import SummaryWriter

    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    self.runtime = make_runtime(cfg)
    self.env, self.env_cfg = self.runtime.env, self.runtime.env_cfg
    self.dev = self.env.device
    self.history, self.terms = self.runtime.history, self.runtime.terms
    self.obs = self.runtime.initial_obs
    self.xi = self.runtime.context
    dims = self.runtime.dynamics
    self.action_dim = dims.action_dim
    obs_dim = self.obs.shape[-1]
    self.wm = WorldModel(cfg.wm_hidden, spec=dims).to(self.dev)
    self.actor = Actor(obs_dim, cfg.actor_hidden, dims.action_dim).to(self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, obs_dim, self.dev,
                            spec=dims, task_dim=self.runtime.task_dim)
    self.archive = WindowArchive(cfg.archive_size, cfg.wm_window, dims.context_dim,
                                  self.dev, spec=dims)
    self.explore = torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
    self.ou = torch.zeros(cfg.num_envs, dims.action_dim, device=self.dev)
    self.imag = self.runtime.make_imagination(cfg)
    self.critic = Critic(self.imag.critic_dim, cfg.critic_hidden).to(self.dev)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    self.a_prev = torch.zeros(cfg.num_envs, dims.action_dim, device=self.dev)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.log_dir = Path(cfg.log_root) / f"{stamp}_{cfg.run_name}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(str(self.log_dir))
    self.ep_stats: dict[str, list[float]] = defaultdict(list)
    self.env_steps = 0
    self.actor_frozen_until = 0
    self.policy_start = cfg.warmup_iterations
    self.lr_scale = 1.0
    self.ramp_start = 0
    self.last_ckpt: Path | None = None
    self.collapse_strikes = 0
    self.nonfinite_steps = 0
    self.start_it = 0
    if cfg.resume:
      self.load(cfg.resume)

  def load(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    if not all(key in ckpt for key in ("actor", "critic", "world_model")):
      raise ValueError("resuming training requires actor, critic and world_model; "
                       "the released weights contain only the actor and support evaluation/export")
    old_terms = self.runtime.checkpoint_terms(ckpt)
    wm_sd = self.wm.state_dict()
    same_wm = all(
      k in ckpt["world_model"] and ckpt["world_model"][k].shape == v.shape for k, v in wm_sd.items()
    )
    if same_wm:
      self.wm.load_state_dict(ckpt["world_model"])
      if "opt_wm" in ckpt:
        self.opt_wm.load_state_dict(ckpt["opt_wm"])
    elif expand_world_model_scan(ckpt["world_model"], self.wm):
      print("world model: ground-scan inputs added (zero weight, same predictions)", flush=True)
    else:
      self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iterations)
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(f"world model re-initialised (shape changed); policy updates from iteration "
            f"{self.policy_start}", flush=True)
    old_obs_dim = ckpt["actor"]["norm.mean"].shape[0]
    new_obs_dim = self.obs.shape[-1]
    if old_terms == self.terms and old_obs_dim == new_obs_dim:
      self.actor.load_state_dict(ckpt["actor"])
      if "opt_actor" in ckpt:
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
    elif all(dict(self.terms).get(name) == dim for name, dim in old_terms):
      expand_actor_new_terms(ckpt["actor"], self.actor, old_terms, self.terms, self.history)
      print(f"actor observation fields mapped from {old_obs_dim} to {new_obs_dim} inputs", flush=True)
    else:
      raise ValueError(f"actor inputs {old_obs_dim} -> {new_obs_dim}: no warm start for this")
    critic_sd = self.critic.state_dict()
    same_critic = all(
      k in ckpt["critic"] and ckpt["critic"][k].shape == v.shape for k, v in critic_sd.items()
    )
    if same_critic and not self.cfg.reset_critic:
      self.critic.load_state_dict(ckpt["critic"])
      if "opt_critic" in ckpt:
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
    else:
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(f"critic re-initialised; actor frozen until iteration {self.actor_frozen_until}",
            flush=True)
    print(f"resumed from {path} (iteration {ckpt.get('iteration')})", flush=True)
    if self.cfg.continue_run:
      # One env step per iteration: the env-side curricula count env steps.
      self.start_it = int(ckpt.get("iteration", 0))
      self.env.common_step_counter = int(ckpt.get("env_step_counter",
                                                  self.start_it * self.cfg.collect_steps))
      print(f"continuing at iteration {self.start_it}", flush=True)

  # Phase 1.

  @torch.no_grad()
  def collect(self) -> None:
    env, cfg = self.env, self.cfg
    for _ in range(cfg.collect_steps):
      s = self.runtime.state()
      pose = self.runtime.pose()
      task_state = self.runtime.task_state()
      relief = self._relief(pose)
      a = self.actor(self.obs)
      dt = env.step_dt
      a = self.runtime.driven_actions(a, self.obs, getattr(self, "it", 0))
      sigma_ou = cfg.explore_noise * (2 * cfg.explore_theta) ** 0.5
      self.ou = self.ou * (1 - cfg.explore_theta * dt) + sigma_ou * dt**0.5 * torch.randn_like(a)
      noise = torch.where(
        self.explore.unsqueeze(-1), self.ou, cfg.action_noise * torch.randn_like(a)
      )
      a = a + noise
      obs, _, term, trunc, extras = env.step(a)
      done = term | trunc
      self.runtime.on_reset(done)
      self.ou = torch.where(done.unsqueeze(-1), torch.zeros_like(self.ou), self.ou)
      self.buf.add(s, pose, a, self.a_prev, self.obs, task_state, done, relief)
      keep = ~done
      if keep.any():
        s1 = self.runtime.state()
        dpose = self.runtime.pose_delta(pose, self.runtime.pose())
        self.wm.update_stats(
          s[keep], a[keep], self.a_prev[keep], self.xi[keep], relief[keep], s1[keep], dpose[keep]
        )
      self.actor.norm.update(self.obs)
      for name, value in self.runtime.metrics().items():
        self.ep_stats[name].append(float(value))
      self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)
      self.obs = obs["actor"]
      self.env_steps += env.num_envs
      for k, v in extras.get("log", {}).items():
        if k.startswith(("Episode_Reward", "Metrics/", "Episode_Termination")):
          self.ep_stats[k].append(float(v))

  def _relief(self, pose: torch.Tensor) -> torch.Tensor:
    return self.runtime.relief(pose)
  def save(self, it: int) -> Path:
    path = self.log_dir / f"model_{it}.pt"
    torch.save(
      {
        "supermimic": True,
        "iteration": it,
        "env_step_counter": self.env.common_step_counter,
        "cfg": asdict(self.cfg),
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "world_model": self.wm.state_dict(),
        "opt_wm": self.opt_wm.state_dict(),
        "opt_actor": self.opt_actor.state_dict(),
        "opt_critic": self.opt_critic.state_dict(),
        "obs_dim": self.obs.shape[-1],
        "metadata": {"algorithm": "supermimic", **self.runtime.checkpoint_metadata()},
      },
      path,
    )
    return path

  def run(self) -> None:
    cfg = self.cfg
    terminal = TerminalLogger(cfg)
    terminal.start(self.log_dir, self.obs.shape[-1], self.action_dim, self.start_it)
    t_start = time.perf_counter()
    print(f"[Replay] Collecting initial transitions (need {cfg.wm_window + 2} steps per env)...",
          flush=True)
    while self.buf.size <= cfg.wm_window + 1:
      self.collect()
    print(f"[Replay] Ready: {self.buf.size} steps per env. Starting training.", flush=True)
    window_start = time.perf_counter()
    window_it, window_steps = self.start_it, self.env_steps
    phase_seconds = defaultdict(float)
    for it in range(self.start_it + 1, cfg.iterations + 1):
      self.it = it
      lr = lr_at(cfg, it) * self.lr_scale
      start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
      ramp = min(1.0, max(0.0, it - start) / max(1, cfg.actor_ramp_iterations))
      actor_scale = ramp * cfg.actor_lr_scale
      for opt, scale in ((self.opt_wm, 1.0), (self.opt_actor, actor_scale), (self.opt_critic, 1.0)):
        for g in opt.param_groups:
          g["lr"] = lr * scale
      phase_start = time.perf_counter()
      self.collect()
      phase_seconds["collect"] += time.perf_counter() - phase_start
      phase_start = time.perf_counter()
      wm_loss = self.fit_world_model()
      phase_seconds["world_model"] += time.perf_counter() - phase_start
      stats = {"wm_loss": wm_loss}
      if it > self.policy_start:
        phase_start = time.perf_counter()
        stats.update(self.improve_policy())
        phase_seconds["policy"] += time.perf_counter() - phase_start
      if it % cfg.log_interval == 0 or it == cfg.iterations:
        now = time.perf_counter()
        elapsed = now - t_start
        for k, v in stats.items():
          self.writer.add_scalar(f"train/{k}", v, it)
        falls = self.ep_stats.get("Episode_Termination/fell_over", [])
        falls_mean = sum(falls) / len(falls) if falls else 0.0
        metrics = {}
        for k, vals in self.ep_stats.items():
          mean = sum(vals) / len(vals)
          self.writer.add_scalar(k, mean, it)
          metrics[k] = mean
        metrics.update(self.runtime.log_metrics(self.writer, it))
        self.ep_stats.clear()
        self.writer.add_scalar("time/minutes", elapsed / 60, it)
        self.writer.add_scalar("time/env_fps", (self.env_steps - window_steps) / max(1e-9, now - window_start), it)
        terminal.report(
          iteration=it, stage=training_stage(it, self.policy_start, self.actor_frozen_until),
          elapsed=elapsed, window_seconds=now - window_start, window_iterations=it - window_it,
          env_steps=self.env_steps, window_env_steps=self.env_steps - window_steps,
          stats=stats, metrics=metrics,
          learning_rates=tuple(opt.param_groups[0]["lr"] for opt in (self.opt_wm, self.opt_actor, self.opt_critic)),
          archive_size=self.archive.size, replay_size=self.buf.size,
          nonfinite_steps=self.nonfinite_steps, phase_seconds=phase_seconds,
        )
        window_start, window_it, window_steps = time.perf_counter(), it, self.env_steps
        phase_seconds.clear()
        if it % cfg.log_interval == 0:
          self._collapse_guard(falls_mean)
      if it % cfg.save_interval == 0 or it == cfg.iterations:
        if self.collapse_strikes == 0:
          self.last_ckpt = self.save(it)
          print(f"[Checkpoint] Saved: {self.last_ckpt}", flush=True)

  def _collapse_guard(self, falls: float) -> None:
    """Model exploitation shows up as many real falls: roll back and slow down."""
    if self.it <= self.actor_frozen_until or falls <= self.cfg.collapse_falls:
      self.collapse_strikes = 0
      return
    self.collapse_strikes += 1
    if self.collapse_strikes >= 2 and self.last_ckpt is not None:
      # The world model is kept: it has just learned from the real falls the actor
      # exploited, and reloading it would let the same exploit happen again.
      ckpt = torch.load(self.last_ckpt, map_location=self.dev, weights_only=False)
      self.actor.load_state_dict(ckpt["actor"])
      self.critic.load_state_dict(ckpt["critic"])
      self.lr_scale *= 0.5
      self.collapse_strikes = 0
      self.ramp_start = self.it
      # Fresh optimizer moments: the old ones carry the step that diverged.
      self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95))
      self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
      print(f"collapse guard: {falls:.1f} falls, rolled back to {self.last_ckpt}, "
            f"lr scale {self.lr_scale:.3g}", flush=True)

  def close(self):
    self.writer.close()
    self.env.close()


# Backward-compatible class name for callers of the original training module.
Trainer = SuperMimicRunner
