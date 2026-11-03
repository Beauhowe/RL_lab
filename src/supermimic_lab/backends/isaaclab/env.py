"""Adapter for Isaac Lab 2.3's official Go2 velocity baselines.

These are native Isaac Lab observations/rewards/assets, not a claim of parity
with the mjlab task or a cross-simulator checkpoint conversion.
"""

NATIVE_TASKS = {
  'Go2-Locomotion-Flat': 'Isaac-Velocity-Flat-Unitree-Go2-v0',
  'Go2-Locomotion-Rough': 'Isaac-Velocity-Rough-Unitree-Go2-v0',
}


def native_task_id(task, play=False):
  name = NATIVE_TASKS[task]
  return name.removesuffix('-v0') + '-Play-v0' if play else name


def load_env_cfg(task, *, play=False):
  # Only called after AppLauncher has started the simulation application.
  import isaaclab_tasks  # noqa: F401; registers the official Gym tasks
  from isaaclab_tasks.utils import load_cfg_from_registry

  return load_cfg_from_registry(native_task_id(task, play), 'env_cfg_entry_point')


def make_ppo_env(cfg):
  from isaaclab.envs import ManagerBasedRLEnv
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.sim.device, env_cfg.seed = cfg.device, cfg.seed
  env = ManagerBasedRLEnv(cfg=env_cfg)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
    observations = wrapped.get_observations()
    action = env.action_manager.get_term('joint_pos')
    names = getattr(action, '_joint_names', None)
    if names is None:
      names = env.scene['robot'].joint_names
    info = dict(obs_group='policy', obs_dim=observations['policy'].shape[-1],
                action_dim=wrapped.num_actions, action_names=list(names),
                environment_profile='isaaclab_2_3_native_go2_velocity',
                native_task=native_task_id(cfg.task))
    return wrapped, info
  except BaseException:
    env.close()
    raise
