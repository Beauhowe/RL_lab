"""Construction of mjlab environments behind the training entry points."""


def make_env(cfg, *, play=False, render_mode=None, configure=None):
  from mjlab.envs import ManagerBasedRlEnv
  from supermimic_lab.tasks.registry import load_env_cfg

  if cfg.backend != "mjlab":
    raise ValueError(f"unsupported backend: {cfg.backend}")
  env_cfg = load_env_cfg(cfg.task, play=play, robot=cfg.robot)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if configure is not None:
    configure(env_cfg)
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=render_mode), env_cfg
