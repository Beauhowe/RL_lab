"""Native Isaac Sim playback/evaluation for project PPO checkpoints."""

from types import SimpleNamespace
from .launcher import launch


def play(args, checkpoint_loader=None, *, evaluation=False):
  if getattr(args, 'terrain', None):
    raise ValueError('Isaac Lab native playback currently uses its complete terrain generator; '
                     '--terrain selection is provided by the mjlab backend')
  if getattr(args, 'video', None):
    raise ValueError('Isaac Lab video recording is not configured; use native playback or the mjlab recorder')
  app = launch(SimpleNamespace(headless=getattr(args, 'headless', evaluation), device=args.device))
  env = None
  try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from supermimic_lab.models.policy import load_actor
    from .env import load_env_cfg

    if checkpoint_loader is not None:
      path, data = checkpoint_loader(torch, args)
    else:
      path = args.checkpoint
      data = torch.load(path, map_location='cpu', weights_only=False)
    meta = data.get('metadata', {})
    if meta.get('backend') != 'isaaclab' or meta.get('algorithm') != 'ppo':
      raise ValueError('Isaac Lab playback requires an Isaac Lab PPO checkpoint')
    if meta.get('task') != args.task:
      raise ValueError('select the task recorded in this Isaac Lab checkpoint')
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs or 1
    cfg.sim.device, cfg.seed = args.device, args.seed
    command = tuple(args.command) if args.command is not None else (1., 0., 0.)
    twist = cfg.commands.base_velocity
    twist.heading_command = False
    twist.rel_standing_envs = 0.
    twist.ranges.lin_vel_x = (command[0], command[0])
    twist.ranges.lin_vel_y = (command[1], command[1])
    twist.ranges.ang_vel_z = (command[2], command[2])
    cfg.curriculum.terrain_levels = None
    env = ManagerBasedRLEnv(cfg=cfg)
    if env.scene.terrain.terrain_origins is not None:
      terrain = env.scene.terrain
      level = min(terrain.terrain_origins.shape[0] - 1,
                  int(args.difficulty * terrain.terrain_origins.shape[0]))
      terrain.terrain_levels.fill_(level)
      terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
    obs, _ = env.reset()
    actor = load_actor(data, args.device)
    print(f'[Play] Isaac Lab PPO: {path}; command={command}', flush=True)
    steps = None if args.seconds is None else max(1, round(args.seconds / env.step_dt))
    count, falls, valid_count = 0, 0, 0
    linear_total, yaw_total = 0., 0.
    with torch.inference_mode():
      while app.is_running() and (steps is None or count < steps):
        obs, _, terminated, truncated, _ = env.step(actor(obs['policy']))
        valid = ~(terminated | truncated)
        robot = env.scene['robot'].data
        linear_total += float((robot.root_lin_vel_b[valid, :2]
                               - torch.tensor(command[:2], device=env.device)).norm(dim=-1).sum())
        yaw_total += float((robot.root_ang_vel_b[valid, 2] - command[2]).abs().sum())
        valid_count += int(valid.sum())
        falls += int(terminated.sum())
        count += 1
    if valid_count:
      print(f'[Evaluation] Linear error {linear_total / valid_count:.3f} m/s; '
            f'yaw error {yaw_total / valid_count:.3f} rad/s; '
            f'terminations {falls}; valid samples {valid_count}', flush=True)
  finally:
    try:
      if env is not None:
        env.close()
    finally:
      app.close()


def evaluate(args):
  args.headless = True
  return play(args, evaluation=True)
