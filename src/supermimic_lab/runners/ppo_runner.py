"""PPO lifecycle backed by RSL-RL, independent of SuperMimic runtimes."""

from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import copy
import json
import torch
from rsl_rl.runners import OnPolicyRunner

from supermimic_lab.algorithms.ppo.rsl_config import rsl_major, runner_config
from supermimic_lab.backends.registry import make_ppo_env


def validate_checkpoint(checkpoint, cfg, metadata):
  old = checkpoint.get('metadata', {})
  for key in ('format_version', 'algorithm', 'backend', 'robot', 'environment_profile', 'obs_dim',
              'action_dim', 'action_names', 'rsl_major'):
    if old.get(key) != metadata[key]:
      raise ValueError(f'PPO checkpoint {key} is incompatible; use a matching PPO checkpoint')
  if cfg.continue_run and old.get('task') != cfg.task:
    raise ValueError('continue-run requires the same PPO task; omit it to warm-start')


class PpoRunner:
  def __init__(self, cfg):
    self.cfg = cfg
    self.major = rsl_major()
    torch.manual_seed(cfg.seed)
    self.env, info = make_ppo_env(cfg)
    try:
      self.metadata = dict(format_version=1, algorithm='ppo', backend=cfg.backend,
        robot=cfg.robot, task=cfg.task, rsl_major=self.major,
        rsl_version=version('rsl-rl-lib'), **info)
      self.log_dir = Path(cfg.log_root) / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + cfg.run_name)
      self.log_dir.mkdir(parents=True, exist_ok=True)
      settings = runner_config(cfg, obs_group=info['obs_group'], major=self.major)
      # RSL-RL consumes/populates its configuration during model construction.
      self.settings = copy.deepcopy(settings)
      self._last_saved = None
      owner = self

      class CheckpointRunner(OnPolicyRunner):
        def save(self, path, infos=None):
          completed = self.current_learning_iteration + 1
          if (completed % cfg.save_interval and completed != cfg.iterations) or owner._last_saved == completed:
            return
          if owner.major == 5:
            data = self.alg.save()
          else:
            data = dict(model_state_dict=self.alg.policy.state_dict(),
                        optimizer_state_dict=self.alg.optimizer.state_dict())
          data.update(iter=self.current_learning_iteration, iteration=completed,
            infos=infos, cfg=asdict(cfg), metadata=owner.metadata,
            rsl_config=owner.settings,
            env_step_counter=self.env.unwrapped.common_step_counter)
          destination = Path(path).with_name(f'model_{completed}.pt')
          torch.save(data, destination)
          owner._last_saved = completed
          print(f'[Checkpoint] Saved PPO: {destination}', flush=True)

      native_settings = copy.deepcopy(settings)
      # Native runners use zero-based save scheduling; filter by completed updates above.
      native_settings['save_interval'] = 1
      self.runner = CheckpointRunner(self.env, native_settings, str(self.log_dir), cfg.device)
      if cfg.resume:
        data = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        validate_checkpoint(data, cfg, self.metadata)
        if self.major == 5:
          self.runner.alg.load(data, load_cfg=None, strict=True)
        else:
          self.runner.alg.policy.load_state_dict(data['model_state_dict'])
          self.runner.alg.optimizer.load_state_dict(data['optimizer_state_dict'])
          self.runner.alg.learning_rate = self.runner.alg.optimizer.param_groups[0]['lr']
        if cfg.continue_run:
          self.runner.current_learning_iteration = int(data['iteration'])
          self.env.unwrapped.common_step_counter = int(data['env_step_counter'])
          self.env.reset()
        print(f'[Resume] PPO {cfg.resume}; start iteration {self.runner.current_learning_iteration}', flush=True)
      (self.log_dir / 'experiment.json').write_text(json.dumps(asdict(cfg), indent=2))
      (self.log_dir / 'metadata.json').write_text(json.dumps(self.metadata, indent=2))
    except BaseException:
      self.env.close()
      raise

  def run(self):
    remaining = self.cfg.iterations - self.runner.current_learning_iteration
    if remaining <= 0:
      raise ValueError('iterations must exceed the completed iteration count in the PPO checkpoint')
    print(f'[PPO] {self.cfg.backend} / {self.cfg.task} | {self.env.num_envs} environments | '
          f'{self.cfg.num_steps_per_env} steps/env/update | logs: {self.log_dir}', flush=True)
    self.runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)

  def close(self):
    logger = getattr(self.runner, 'logger', None)
    writer = getattr(logger, 'writer', None) if logger is not None else getattr(self.runner, 'writer', None)
    if writer is not None:
      writer.close()
    self.env.close()
