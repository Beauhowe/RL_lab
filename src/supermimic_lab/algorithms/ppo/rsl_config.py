"""RSL-RL 3.x (Isaac Lab 2.3) and 5.x (mjlab) configuration schemas."""

from importlib.metadata import version


def rsl_major():
  major = int(version("rsl-rl-lib").split(".")[0])
  if major not in (3, 5):
    raise RuntimeError(f"RSL-RL {major}.x is not supported by this adapter; use 3.x (Isaac Lab 2.3) or 5.x (mjlab)")
  return major


def runner_config(cfg, *, obs_group="actor", major=None):
  major = rsl_major() if major is None else major
  if major not in (3, 5):
    raise ValueError("unsupported RSL-RL configuration schema")
  algorithm = dict(class_name="PPO", num_learning_epochs=cfg.learning_epochs,
    num_mini_batches=cfg.minibatches, learning_rate=cfg.learning_rate, schedule="adaptive",
    gamma=cfg.gamma, lam=cfg.lam, entropy_coef=cfg.entropy_coef, desired_kl=cfg.desired_kl,
    max_grad_norm=cfg.max_grad_norm, value_loss_coef=1.0, use_clipped_value_loss=True,
    clip_param=cfg.clip_param, normalize_advantage_per_mini_batch=False,
    rnd_cfg=None, symmetry_cfg=None)
  result = dict(seed=cfg.seed, device=cfg.device, num_steps_per_env=cfg.num_steps_per_env,
    save_interval=cfg.save_interval, logger="tensorboard", algorithm=algorithm,
    obs_groups={"critic": [obs_group]}, check_for_nan=True)
  if major == 5:
    result["obs_groups"]["actor"] = [obs_group]
    result["actor"] = dict(class_name="MLPModel", hidden_dims=cfg.actor_hidden,
      activation="elu", obs_normalization=cfg.observation_normalization,
      distribution_cfg=dict(class_name="GaussianDistribution", init_std=cfg.init_noise_std,
                            std_type="log"))
    result["critic"] = dict(class_name="MLPModel", hidden_dims=cfg.critic_hidden,
      activation="elu", obs_normalization=cfg.observation_normalization)
  else:
    result["obs_groups"]["policy"] = [obs_group]
    result["policy"] = dict(class_name="ActorCritic", init_noise_std=cfg.init_noise_std,
      noise_std_type="log", actor_hidden_dims=cfg.actor_hidden, critic_hidden_dims=cfg.critic_hidden,
      activation="elu", actor_obs_normalization=cfg.observation_normalization,
      critic_obs_normalization=cfg.observation_normalization)
  return result
