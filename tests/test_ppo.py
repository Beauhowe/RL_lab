"""PPO dispatch, optimizer/checkpoint lifecycle and deterministic inference."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from supermimic_lab.algorithms.ppo.config import PpoTrainCfg
from supermimic_lab.algorithms.ppo.rsl_config import runner_config
from supermimic_lab.runners.cli import parse_train_cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('backend', ['mjlab', 'isaaclab'])
@pytest.mark.parametrize('terrain', ['flat', 'rough'])
def test_ppo_presets_and_cli_overrides(backend, terrain):
  cfg = parse_train_cfg(['--config', str(ROOT / f'configs/experiments/go2_ppo_{backend}_{terrain}.toml'),
                         '--iterations', '7'])
  assert isinstance(cfg, PpoTrainCfg)
  assert cfg.algorithm == 'ppo' and cfg.backend == backend and cfg.iterations == 7
  assert not hasattr(cfg, 'wm_updates')


def test_backend_dispatch_is_lazy_and_unsupported_combinations_fail():
  source = '''import sys
from supermimic_lab.runners.cli import parse_train_cfg
c=parse_train_cfg(['--algorithm','ppo','--backend','isaaclab'])
assert c.backend == 'isaaclab'
assert not any(k in sys.modules for k in ('torch','mjlab','isaaclab','isaacsim'))
'''
  result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, timeout=10)
  assert result.returncode == 0, result.stderr
  for args in (['--backend','isaaclab'], ['--algorithm','ppo','--wm-updates','4'],
               ['--algorithm','ppo','--task','Go2-Chase-Flat']):
    with pytest.raises(SystemExit):
      parse_train_cfg(args)


def test_rsl_config_schemas_use_actual_environment_observation_groups():
  cfg = PpoTrainCfg()
  modern = runner_config(cfg, major=5)
  legacy = runner_config(cfg, major=3, obs_group='policy')
  assert modern['obs_groups'] == {'actor': ['actor'], 'critic': ['actor']}
  assert legacy['obs_groups'] == {'policy': ['policy'], 'critic': ['policy']}
  assert 'policy' not in modern and 'actor' not in legacy


class ToyVecEnv:
  """Tiny continuous task for exercising the actual RSL-RL update and resume."""
  def __init__(self):
    self.num_envs, self.num_actions, self.device = 4, 2, torch.device('cpu')
    self.cfg = {}
    self.max_episode_length = 8
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)
    self.common_step_counter = 0
    self.unwrapped = self
    self.obs = TensorDict({'actor': torch.randn(4, 8)}, batch_size=[4])
    self.closed = False

  def get_observations(self):
    return self.obs

  def reset(self):
    self.episode_length_buf.zero_()
    return self.obs, {}

  def step(self, actions):
    self.common_step_counter += 1
    self.episode_length_buf += 1
    reward = 1 - (actions - self.obs['actor'][:, :2]).square().sum(-1)
    done = self.episode_length_buf >= self.max_episode_length
    self.episode_length_buf[done] = 0
    self.obs = TensorDict({'actor': torch.randn(4, 8)}, batch_size=[4])
    return self.obs, reward, done.long(), {'time_outs': done}

  def close(self):
    self.closed = True


def test_real_ppo_updates_resume_and_exportable_actor(tmp_path, monkeypatch):
  from supermimic_lab.runners import ppo_runner as module
  from supermimic_lab.models.policy import load_actor

  def factory(cfg):
    return ToyVecEnv(), dict(obs_group='actor', obs_dim=8, action_dim=2,
      action_names=['left','right'], environment_profile='test_velocity')

  monkeypatch.setattr(module, 'make_ppo_env', factory)
  cfg = PpoTrainCfg(device='cpu', num_envs=4, iterations=3, num_steps_per_env=4,
    actor_hidden=(16,), critic_hidden=(16,), learning_epochs=1, minibatches=2,
    save_interval=2, log_root=str(tmp_path), run_name='first')
  runner = module.PpoRunner(cfg)
  try:
    runner.run()
    path = runner.log_dir / 'model_3.pt'
    assert (runner.log_dir / 'model_2.pt').is_file()
    assert not (runner.log_dir / 'model_1.pt').exists()
    data = torch.load(path, weights_only=False)
    assert data['iteration'] == 3 and data['env_step_counter'] == 12
    assert 'world_model' not in data
    previous = torch.load(runner.log_dir / 'model_2.pt', weights_only=False)
    assert any(not torch.equal(value, previous['actor_state_dict'][key])
               for key, value in data['actor_state_dict'].items() if key.startswith('mlp.'))
    actor = load_actor(data)
    obs = TensorDict({'actor': torch.randn(3, 8)}, batch_size=[3])
    reference = runner.runner.get_inference_policy()(obs)
    assert torch.allclose(actor(obs['actor']), reference, atol=1e-6)
    traced = torch.jit.trace(actor, obs['actor'])
    assert torch.allclose(traced(obs['actor']), reference, atol=1e-6)
    with pytest.raises(ValueError, match='backend'):
      module.validate_checkpoint(data, cfg, {**data['metadata'], 'backend': 'isaaclab'})
  finally:
    runner.close()
  assert runner.env.closed
  cfg2 = replace(cfg, resume=str(path), continue_run=True, iterations=4, run_name='continued')
  continued = module.PpoRunner(cfg2)
  try:
    assert continued.runner.current_learning_iteration == 3
    continued.run()
    resumed = torch.load(continued.log_dir / 'model_4.pt', weights_only=False)
    assert resumed['iteration'] == 4 and resumed['env_step_counter'] == 16
  finally:
    continued.close()
