"""mjlab PPO adapter; it does not construct a world model or replay buffer."""


def make_ppo_env(cfg):
  import torch
  from tensordict import TensorDict
  from mjlab.rl import RslRlVecEnvWrapper
  from supermimic_lab.backends.mjlab.env import make_env
  from supermimic_lab.tasks.locomotion.velocity import state as st

  def configure(env_cfg):
    env_cfg.commands['twist'].velocity.curriculum_steps = cfg.command_curriculum_steps
    if 'terrain_levels' in env_cfg.curriculum:
      env_cfg.curriculum['terrain_levels'].params['steps'] = cfg.terrain_curriculum_steps

  class CachedWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
      super().__init__(env, clip_actions=cfg.clip_actions)
      self._observations = TensorDict(env.observation_manager.compute(update_history=False),
                                     batch_size=[self.num_envs])

    def get_observations(self):
      # RSL-RL queries this repeatedly; reading must not advance history.
      return self._observations

    def reset(self):
      self._observations, info = super().reset()
      return self._observations, info

    def step(self, actions):
      obs, reward, done, info = super().step(actions)
      self._observations = obs
      term = self.env.command_manager.get_term('twist')
      state = st.read_state(self.env, term.height_field)
      values = {
        'linear_error': (state[:, st.V][:, :2] - term.command[:, :2]).norm(dim=-1).mean(),
        'yaw_error': (state[:, st.W][:, 2] - term.command[:, 2]).abs().mean(),
        'speed': state[:, st.V][:, :2].norm(dim=-1).mean(),
        'command_speed': term.command[:, :2].norm(dim=-1).mean(),
      }
      if term.cfg.rough:
        values['terrain_level'] = self.env.scene.terrain.terrain_levels.float().mean()
      info.setdefault('log', {}).update({f'Metrics/locomotion/{k}': v for k, v in values.items()})
      return obs, reward, done, info

  env, env_cfg = make_env(cfg, configure=configure)
  try:
    wrapped = CachedWrapper(env)
    action = env.action_manager.get_term('joint_pos')
    info = dict(obs_group='actor', obs_dim=wrapped.get_observations()['actor'].shape[-1],
                action_dim=wrapped.num_actions, action_names=list(action._target_names),
                environment_profile='supermimic_lab_velocity_v1')
    return wrapped, info
  except BaseException:
    env.close()
    raise
