"""Lazy backend construction; importing one simulator never imports the other."""

from supermimic_lab.tasks.registry import resolve

PPO_FACTORIES = {
  'mjlab': 'supermimic_lab.backends.mjlab.ppo:make_ppo_env',
  'isaaclab': 'supermimic_lab.backends.isaaclab.env:make_ppo_env',
}


def make_ppo_env(cfg):
  return resolve(PPO_FACTORIES[cfg.backend])(cfg)


def launch_backend(cfg):
  if cfg.backend == 'isaaclab':
    from .isaaclab.launcher import launch
    return launch(cfg)
  return None
