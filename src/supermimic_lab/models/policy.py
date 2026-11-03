"""Deterministic actor loading for SuperMimic and project PPO checkpoints."""

import copy
import torch
from torch import nn


def load_actor(checkpoint, device='cpu'):
  meta = checkpoint.get('metadata', {})
  if meta.get('algorithm') != 'ppo':
    from .actor_critic import Actor

    state = checkpoint['actor']
    hidden = tuple(checkpoint['cfg']['actor_hidden'])
    actions = state[f'net.{2 * len(hidden)}.weight'].shape[0]
    actor = Actor(state['norm.mean'].numel(), hidden, actions)
    actor.load_state_dict(state)
    return actor.to(device).eval()

  from tensordict import TensorDict
  from supermimic_lab.algorithms.ppo.rsl_config import rsl_major

  major = rsl_major()
  if meta['rsl_major'] != major:
    raise ValueError('PPO checkpoint RSL-RL version differs; export with its original environment first')
  settings = copy.deepcopy(checkpoint['rsl_config'])
  group = meta['obs_group']
  obs = TensorDict({group: torch.zeros(1, meta['obs_dim'])}, batch_size=[1])
  if major == 5:
    from rsl_rl.models import MLPModel

    config = settings['actor']
    config.pop('class_name')
    actor = MLPModel(obs, settings['obs_groups'], 'actor', meta['action_dim'], **config)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    result = actor.as_jit()
  else:
    from rsl_rl.modules import ActorCritic

    config = settings['policy']
    config.pop('class_name')
    model = ActorCritic(obs, settings['obs_groups'], meta['action_dim'], **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Deterministic PPO mean action, including the learned observation normalizer.
    result = nn.Sequential(model.actor_obs_normalizer, model.actor)
  return result.to(device).eval()
