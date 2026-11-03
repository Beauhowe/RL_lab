"""Plain-text training reports that also remain readable in redirected logs."""

import math


def duration(seconds):
  if not math.isfinite(seconds) or seconds < 0:
    return "--:--:--"
  seconds = int(seconds)
  hours, seconds = divmod(seconds, 3600)
  minutes, seconds = divmod(seconds, 60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def number(value):
  if not math.isfinite(value):
    return str(value).upper()
  if value and (abs(value) < 0.0001 or abs(value) >= 1_000_000):
    return f"{value:.3e}"
  return f"{value:,.4f}"


TRAIN_LABELS = {
  "wm_loss": "World model loss",
  "critic_loss": "Critic loss",
  "imagined_return": "Imagined return",
  "imagined_reward": "Imagined reward / step",
  "imagined_survival": "Imagined survival (horizon)",
  "actor_grad_norm": "Actor gradient norm (before clipping)",
}
TASK_LABELS = {
  "Metrics/locomotion/linear_error": ("Linear velocity error", " m/s"),
  "Metrics/locomotion/yaw_error": ("Yaw-rate error", " rad/s"),
  "Metrics/locomotion/speed": ("Robot planar speed", " m/s"),
  "Metrics/locomotion/command_speed": ("Command planar speed", " m/s"),
  "Metrics/locomotion/terrain_level": ("Terrain level (mean)", ""),
  "Metrics/hit_rate/ball": ("Hit rate / ball", "%"),
  "Metrics/hit_rate/low_bar": ("Hit rate / low bar", "%"),
  "Metrics/hit_rate/high_bar": ("Hit rate / high bar", "%"),
  "Metrics/fast_yaw": ("Fast-yaw fraction", "%"),
  "Metrics/target/tags": ("Tags (task metric)", ""),
  "Metrics/target/hits": ("Hits (task metric)", ""),
  "Metrics/target/dist": ("Target distance", " m"),
  "Metrics/target/obstacle_contact": ("Obstacle contact (task metric)", ""),
}


def training_stage(iteration, policy_start, actor_frozen_until):
  if iteration <= policy_start:
    return f"World-model warmup (policy starts at {policy_start + 1:,})"
  if iteration <= actor_frozen_until:
    return f"Critic warmup (Actor frozen through {actor_frozen_until:,})"
  return "World model + Actor + Critic"


class TerminalLogger:
  width = 84

  def __init__(self, cfg):
    self.cfg = cfg

  def start(self, log_dir, obs_dim, action_dim, start_iteration):
    cfg = self.cfg
    lines = ["=" * self.width, f"SuperMimic | {cfg.task}",
             f"  Run: {cfg.run_name} | Robot: {cfg.robot} | Device: {cfg.device}",
             f"  Environments: {cfg.num_envs:,} | Observations: {obs_dim:,} | Actions: {action_dim}",
             f"  Iterations: {start_iteration:,} -> {cfg.iterations:,} | Log every {cfg.log_interval} iterations",
             f"  Logs / checkpoints: {log_dir}",
             f"  TensorBoard: tensorboard --logdir {cfg.log_root}",
             "=" * self.width]
    print("\n".join(lines), flush=True)

  def report(self, *, iteration, stage, elapsed, window_seconds, window_iterations,
             env_steps, window_env_steps, stats, metrics, learning_rates,
             archive_size, replay_size, nonfinite_steps, phase_seconds):
    cfg = self.cfg
    progress = min(1.0, iteration / cfg.iterations)
    per_iteration = window_seconds / max(1, window_iterations)
    eta = per_iteration * max(0, cfg.iterations - iteration)
    fps = window_env_steps / max(1e-9, window_seconds)
    if cfg.log_format == "compact":
      items = [f"it {iteration}/{cfg.iterations}", f"stage {stage}",
               f"elapsed {duration(elapsed)}", f"ETA {duration(eta)}", f"FPS {fps:,.0f}",
               f"env_steps {env_steps:,}"]
      items += [f"{key} {value:.4g}" for key, value in stats.items()]
      items += [f"{key} {value:.4g}" for key, value in metrics.items()]
      items += [f"archive {archive_size:,}/{cfg.archive_size:,}",
                f"nonfinite_skipped {nonfinite_steps}"]
      print(" | ".join(items), flush=True)
      return

    lines = ["", "=" * self.width,
             f"SuperMimic | {cfg.task} | Iteration {iteration:,} / {cfg.iterations:,}"]
    filled = int(progress * 30)
    lines.append(f"[{'#' * filled}{'-' * (30 - filled)}] {progress:6.2%} | {stage}")

    def section(title):
      lines.append(f"-- {title} ".ljust(self.width, "-"))

    def row(label, value):
      lines.append(f"  {label + ':':<43} {value:>35}")

    section("Progress")
    row("Elapsed / ETA", f"{duration(elapsed)} / {duration(eta)}")
    row("Env steps (this process) / FPS", f"{env_steps:,} / {fps:,.0f}")
    row("Iteration time (window mean)", f"{per_iteration:.3f} s")
    phase = " / ".join(f"{phase_seconds.get(key, 0) / max(1, window_iterations):.3f}"
                       for key in ("collect", "world_model", "policy"))
    row("Collect / WM / policy (s per iteration)", phase)
    row("Replay steps per env / capacity", f"{replay_size:,} / {cfg.buffer_len:,}")
    row("Rare-event windows / capacity", f"{archive_size:,} / {cfg.archive_size:,}")
    section("Training (latest iteration)")
    for key, value in stats.items():
      text = f"{value:.2%}" if key == "imagined_survival" else number(value)
      row(TRAIN_LABELS.get(key, key.replace("_", " ")), text)
    row("LR world model / Actor / Critic", " / ".join(f"{v:.2e}" for v in learning_rates))
    row("Nonfinite updates skipped (cumulative)", str(nonfinite_steps))

    task = {key: value for key, value in metrics.items() if key.startswith("Metrics/")}
    if task:
      section("Task metrics (log-window mean; hit rates from task)")
      for key, value in task.items():
        label, unit = TASK_LABELS.get(key, (key.removeprefix("Metrics/"), ""))
        row(label, f"{value:.2%}" if unit == "%" else number(value) + unit)
    rewards = {key: value for key, value in metrics.items() if key.startswith("Episode_Reward/")}
    if rewards:
      section("Episode rewards (normalized by episode time limit; mean)")
      for key, value in rewards.items():
        row(key.split("/", 1)[1], number(value))
    terminations = {key: value for key, value in metrics.items()
                    if key.startswith("Episode_Termination/")}
    if terminations:
      section("Terminations (mean count per reset batch)")
      for key, value in terminations.items():
        row(key.split("/", 1)[1], number(value))
    lines.append("=" * self.width)
    print("\n".join(lines), flush=True)
