"""Velocity errors, falls and optional videos for commanded locomotion."""

from pathlib import Path
import torch

from mjlab.envs import ManagerBasedRlEnv
from supermimic_lab.models.policy import load_actor
from supermimic_lab.tasks.registry import load_env_cfg
from supermimic_lab.utils.visualization import polish_renderer
from . import state as st
from .playback import configure_play


def evaluate(args):
  if args.seconds <= 0 or args.num_envs < 1 or not 0 <= args.difficulty <= 1:
    raise ValueError("seconds/num-envs must be positive and difficulty must be in [0, 1]")
  torch.manual_seed(args.seed)
  cfg = load_env_cfg(args.task, play=True, robot=args.robot)
  cfg.scene.num_envs = args.num_envs
  configure_play(cfg, args)
  if args.video:
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
  checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device,
                          render_mode="rgb_array" if args.video else None)
  try:
    env.common_step_counter = int(args.difficulty * 10_000)
    obs, _ = env.reset()
    term = env.command_manager.get_term("twist")
    meta = checkpoint.get("metadata", {})
    expected = meta.get("obs_dim") if meta.get("algorithm") == "ppo" else checkpoint["actor"]["norm.mean"].numel()
    if expected != obs["actor"].shape[-1] or meta.get("backend", "mjlab") != "mjlab":
      raise ValueError("checkpoint observation layout does not match this locomotion task")
    actor = load_actor(checkpoint, args.device)
    if args.video:
      polish_renderer(env)
    count = max(1, int(args.seconds / env.step_dt))
    totals = torch.zeros(6, device=args.device)
    frames = []
    with torch.inference_mode():
      for _ in range(count):
        command = term.command.clone()
        obs, reward, terminated, truncated, _ = env.step(actor(obs["actor"]))
        s = st.read_state(env, term.height_field)
        valid = ~(terminated | truncated)
        linear_error = (s[:, st.V][:, :2] - command[:, :2]).norm(dim=-1)
        yaw_error = (s[:, st.W][:, 2] - command[:, 2]).abs()
        speed = s[:, st.V][:, :2].norm(dim=-1)
        totals += torch.stack([linear_error[valid].sum(), yaw_error[valid].sum(),
                                speed[valid].sum(), reward[valid].sum(),
                                valid.sum(), terminated.sum()])
        if args.video:
          frames.append(env.render())
    linear, yaw, speed, reward, samples, falls = totals.tolist()
    print(f"{args.task} | command={term.cfg.velocity.fixed} | difficulty={args.difficulty:g}")
    print(f"  valid tracking samples    {int(samples)}")
    if samples:
      print(f"  linear velocity error     {linear / samples:.3f} m/s")
      print(f"  yaw-rate error            {yaw / samples:.3f} rad/s")
      print(f"  mean speed                {speed / samples:.3f} m/s")
      print(f"  mean reward rate          {reward / samples / env.step_dt:.3f}")
    print(f"  falls per robot per min   {falls / args.num_envs / (count * env.step_dt / 60):.3f}")
    if args.video:
      import mediapy

      Path(args.video).parent.mkdir(parents=True, exist_ok=True)
      mediapy.write_video(args.video, frames, fps=round(1 / env.step_dt))
      print(f"  video                     {args.video}")
  finally:
    env.close()
