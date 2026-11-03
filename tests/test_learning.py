"""CPU regression checks for dimensions, replay boundaries and legacy policies."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from supermimic_lab.envs.specs import DynamicsSpec
from supermimic_lab.models.actor_critic import Actor, Critic
from supermimic_lab.models.world_model import WorldModel
from supermimic_lab.storage.replay_buffer import ReplayBuffer
from supermimic_lab.storage.window_archive import WindowArchive
from supermimic_lab.utils.checkpoint import expand_actor_new_terms


@pytest.mark.parametrize("actions,terrain", [(6, 0), (23, 5)])
def test_world_model_and_storage_support_non_go2_dimensions(actions, terrain):
  dims = DynamicsSpec(17, actions, 2, terrain, pose_dim=3, height_index=4)
  model = WorldModel((32, 32), spec=dims)
  batch, steps = 3, 2
  s = torch.randn(batch, 17)
  a = torch.randn(batch, steps, actions, requires_grad=True)
  xi = torch.randn(batch, 2)
  relief = torch.randn(batch, steps, terrain)
  loss = model.loss(s, a, torch.zeros_like(a), xi, relief,
                    torch.randn(batch, steps, 17), torch.randn(batch, steps, 3),
                    torch.ones(batch, steps))
  loss.backward()
  assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
  buf = ReplayBuffer(batch, 8, 11, "cpu", spec=dims, task_dim=4)
  assert buf.s.shape == (batch, 8, 17) and buf.a.shape[-1] == actions
  assert buf.relief.shape[-1] == terrain and buf.task.shape[-1] == 4
  archive = WindowArchive(5, steps, 2, "cpu", spec=dims)
  assert archive.data["a"].shape == (5, steps, actions)


def test_replay_windows_do_not_cross_resets_or_ring_boundary():
  dims = DynamicsSpec(2, 1, 0, 0)
  buf = ReplayBuffer(1, 6, 1, "cpu", spec=dims, task_dim=1)
  for step in range(10):
    buf.add(torch.full((1, 2), float(step)), torch.zeros(1, 3), torch.zeros(1, 1),
            torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
            torch.tensor([step == 6]), torch.empty(1, 0))
  env, slots, valid = buf.sample_windows(256, 2)
  values = buf.s[env[:, None], slots, 0]
  assert torch.equal(values[:, 1:] - values[:, :-1], torch.ones(256, 2))
  for j in range(2):
    expected = (values[:, :j + 1] != 6).all(-1).float()
    assert torch.equal(valid[:, j], expected)


@pytest.mark.parametrize("filename", ["flat.pt", "obstacles.pt", "terrain.pt"])
def test_released_actor_matches_explicit_legacy_forward(filename):
  path = Path(__file__).resolve().parents[1] / "weights" / filename
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  sd = ckpt["actor"]
  actor = Actor(ckpt["obs_dim"], tuple(ckpt["cfg"]["actor_hidden"]), 12)
  actor.load_state_dict(sd)
  x = torch.randn(8, ckpt["obs_dim"])
  ref = ((x - sd["norm.mean"]) / (sd["norm.var"].sqrt() + 1e-6)).clamp(-10, 10)
  for i in (0, 2, 4, 6):
    ref = F.linear(ref, sd[f"net.{i}.weight"], sd[f"net.{i}.bias"])
    if i != 6:
      ref = F.elu(ref)
  with torch.no_grad():
    assert torch.allclose(actor(x), ref, atol=1e-6)


def test_flat_to_lidar_warm_start_preserves_actions():
  old_terms, new_terms = (("proprio", 4),), (("proprio", 4), ("lidar", 3))
  old, new = Actor(20, (16,), 6), Actor(35, (16,), 6)
  old.norm.mean.normal_()
  old.norm.var.uniform_(0.5, 2)
  expand_actor_new_terms(old.state_dict(), new, old_terms, new_terms, history=5)
  obs = torch.randn(7, 20)
  with torch.no_grad():
    assert torch.allclose(old(obs), new(torch.cat([obs, torch.randn(7, 15)], -1)), atol=1e-6)


def test_runner_load_accepts_flat_to_obstacles_checkpoint(tmp_path):
  from supermimic_lab.algorithms.supermimic.config import TrainCfg
  from supermimic_lab.runners.supermimic_runner import SuperMimicRunner
  from supermimic_lab.tasks.chase_dodge import state as st
  from supermimic_lab.tasks.chase_dodge.runtime import ChaseDodgeRuntime

  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.cfg = TrainCfg()
  runner.dev = "cpu"
  runner.wm = WorldModel((16,), spec=st.DYNAMICS)
  runner.actor, old = Actor(430, (16,), 12), Actor(350, (16,), 12)
  runner.critic = Critic(107, (16,))
  runner.obs = torch.zeros(1, 430)
  runner.terms = st.OBS_TERMS + (("lidar", 16),)
  runner.history, runner.actor_frozen_until, runner.policy_start = 5, 0, 300
  runtime = ChaseDodgeRuntime.__new__(ChaseDodgeRuntime)
  runtime.history = 5
  runner.runtime = runtime
  path = tmp_path / "full.pt"
  torch.save({"world_model": runner.wm.state_dict(), "actor": old.state_dict(),
              "critic": Critic(91, (16,)).state_dict()}, path)
  runner.load(str(path))
  x = torch.randn(4, 350)
  with torch.no_grad():
    assert torch.allclose(old(x), runner.actor(torch.cat([x, torch.randn(4, 80)], -1)), atol=1e-6)
