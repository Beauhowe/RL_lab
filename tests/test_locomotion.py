"""Locomotion command semantics, terrain geometry and differentiable learning."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.backends.mjlab.state_reader import RobotConsts
from supermimic_lab.envs.terrain import HeightField, SCAN_POINTS
from supermimic_lab.models.actor_critic import Actor, Critic
from supermimic_lab.models.world_model import WorldModel
from supermimic_lab.tasks.locomotion.velocity import state as st
from supermimic_lab.tasks.locomotion.velocity.commands import VelocityCfg, sample_commands, advance_commands
from supermimic_lab.tasks.locomotion.velocity.rewards import RewardCfg, RewardWeights, reward_terms, fall_probability
from supermimic_lab.tasks.locomotion.velocity.imagination import ImagineCfg, LocomotionImagination


def robot_constants(n):
  return RobotConsts(default_q=torch.zeros(n, 12), action_scale=torch.ones(n, 12) * 0.25,
                      kp=torch.ones(12) * 20, kd=torch.ones(12) * 0.5,
                      effort=torch.ones(12) * 30,
                      soft_limits=torch.tensor([-1.5, 1.5]).expand(n, 12, 2).clone())


def standing_state(n):
  s = torch.zeros(n, st.LAYOUT.dim)
  s[:, st.Z] = 0.28
  s[:, st.GRAV] = torch.tensor([0.0, 0.0, -1.0])
  s[:, st.FEET] = torch.tensor([[0.2, -0.1, -0.258], [0.2, 0.1, -0.258],
                                [-0.2, -0.1, -0.258], [-0.2, 0.1, -0.258]]).flatten()
  return s


@pytest.mark.parametrize("progress", [0.0, 1.0])
def test_commands_respect_ranges_and_include_standing(progress):
  torch.manual_seed(12)
  cfg = VelocityCfg()
  command = sample_commands(4096, "cpu", progress, cfg)
  ranges = (cfg.initial_x, cfg.initial_y, cfg.initial_yaw) if progress == 0 else (
    cfg.final_x, cfg.final_y, cfg.final_yaw)
  for i, (lo, hi) in enumerate(ranges):
    assert ((command[:, i] >= lo) & (command[:, i] <= hi)).all()
  standing = (command == 0).all(-1).float().mean()
  assert 0.07 < standing < 0.13
  assert ((command[:, 0] > 0) & (command[:, 1:] == 0).all(-1)).float().mean() > 0.5


def test_command_timer_resamples_only_expired_commands():
  cfg = VelocityCfg(fixed=(0.5, -0.2, 0.3))
  task = torch.tensor([[1.0, 0.0, 0.0, 0.01, 0.2], [1.0, 0.0, 0.0, 1.0, 0.2]])
  result = advance_commands(task, torch.full((2,), 0.02), cfg)
  assert torch.allclose(result[0, :3], torch.tensor(cfg.fixed))
  assert torch.equal(result[1, :3], task[1, :3])
  assert cfg.period[0] <= result[0, 3] <= cfg.period[1]
  assert result[1, 3] == pytest.approx(0.98)
  assert torch.equal(task[:, 3], torch.tensor([0.01, 1.0]))


def test_clearance_uses_ground_under_each_foot_and_is_height_invariant():
  s, pose = standing_state(1), torch.tensor([[1.0, 1.0, 0.0]])
  raised = HeightField(torch.ones(41, 41), (0.0, 0.0), 0.1)
  assert torch.allclose(st.foot_clearance(s, pose, None), st.foot_clearance(s, pose, raised), atol=1e-6)
  xs = torch.arange(41) * 0.1
  step = HeightField(((xs >= 1.1).float() * 0.1).expand(41, -1).clone(), (0.0, 0.0), 0.1)
  clearance = st.foot_clearance(s, pose, step)
  assert (clearance[:, :2] < -0.09).all()
  assert (clearance[:, 2:].abs() < 1e-5).all()


def test_tracking_rewards_and_fall_probability_have_correct_direction():
  cfg = RewardCfg()
  s, pose, constants = standing_state(2), torch.zeros(2, 3), robot_constants(2)
  command = torch.tensor([[0.8, 0.0, 0.2]]).expand(2, -1)
  s[0, st.V] = torch.tensor([0.8, 0.0, 0.0])
  s[0, st.W] = torch.tensor([0.0, 0.0, 0.2])
  r = reward_terms(s, s, pose, pose, torch.zeros(2, 12), torch.zeros(2, 12),
                    command, constants, None, 0.02, cfg)
  assert r["track_linear_velocity"][0] == 1 and r["track_linear_velocity"][1] < 0.1
  assert r["track_angular_velocity"][0] > r["track_angular_velocity"][1]
  p_standing = fall_probability(s, cfg)
  s[1, 0] = 0.08
  assert fall_probability(s, cfg)[1] > 0.9
  assert (p_standing < 0.001).all()


def test_shared_rewards_have_finite_state_and_action_gradients():
  s0, pose = standing_state(2), torch.zeros(2, 3)
  s1 = s0.clone().requires_grad_()
  action = torch.full((2, 12), 0.05, requires_grad=True)
  terms = reward_terms(s0, s1, pose, pose, action, torch.zeros_like(action),
                        torch.ones(2, 3) * 0.2, robot_constants(2), None, 0.02, RewardCfg())
  weights = asdict(RewardWeights())
  value = sum(weights[name] * term.mean() for name, term in terms.items())
  value.backward()
  assert torch.isfinite(s1.grad).all() and torch.isfinite(action.grad).all()
  assert action.grad.abs().sum() > 0


def test_imagination_updates_actor_through_world_model():
  torch.manual_seed(0)
  n = 2
  s, pose = standing_state(n), torch.zeros(n, 3)
  command = torch.tensor([[0.8, 0.0, 0.0]]).expand(n, 3)
  frame = st.observation_frame(s, torch.zeros(n, 12), command, torch.zeros(n, 12),
                               torch.full((n, SCAN_POINTS), 0.28))
  obs = torch.cat([frame[name].repeat(1, st.HISTORY) for name, _ in st.OBS_TERMS], -1)
  actor, wm = Actor(obs.shape[-1], (16,), 12), WorldModel((16,), spec=st.DYNAMICS)
  with torch.no_grad():
    wm.net[-1].weight.mul_(0.01)
    wm.net[-1].bias.zero_()
  imagination = LocomotionImagination(ImagineCfg(horizon=3), robot_constants(n), VelocityCfg(), None, "cpu")
  critic = Critic(imagination.critic_dim, (16,))
  seed = {"env": torch.arange(n), "s": s, "pose": pose, "a_prev": torch.zeros(n, 12),
          "obs": obs, "task": torch.cat([command, torch.full((n, 1), 4.0), torch.zeros(n, 1)], -1),
          "xi": torch.zeros(n, 4), "obs_context": torch.zeros(n, 12)}
  objective, data = imagination.rollout(wm, actor, critic, seed)
  (-objective.mean()).backward()
  assert torch.isfinite(objective).all()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
  assert sum(p.grad.abs().sum() for p in actor.parameters()) > 0
  assert data["critic_in"].shape == (n, 4, imagination.critic_dim)


def test_environment_presets_share_layout_and_cover_all_terrain_types():
  from supermimic_lab.tasks.registry import load_env_cfg

  flat, rough = (load_env_cfg(name) for name in ("Go2-Locomotion-Flat", "Go2-Locomotion-Rough"))
  assert tuple(flat.observations["actor"].terms) == tuple(rough.observations["actor"].terms)
  assert sum(d for _, d in st.OBS_TERMS) * st.HISTORY == 610
  assert list(flat.commands) == ["twist"] and "target" not in rough.commands
  terrains = rough.scene.terrain.terrain_generator
  assert set(terrains.sub_terrains) == {"flat", "stairs", "stairs_inv", "slope", "slope_inv", "boxes", "rough"}
  assert all(t.proportion == pytest.approx(1 / 7) for t in terrains.sub_terrains.values())
  assert terrains.num_cols == 7 and terrains.num_rows == 5
  assert not any("push" in name for name in rough.events)


def test_playback_selects_terrain_and_fixed_command():
  from supermimic_lab.tasks.registry import load_env_cfg, configure_play

  cfg = load_env_cfg("Go2-Locomotion-Rough", play=True)
  configure_play("Go2-Locomotion-Rough", cfg,
                 SimpleNamespace(command=(0.6, 0.0, 0.0), terrain="stairs", seed=9))
  assert tuple(cfg.scene.terrain.terrain_generator.sub_terrains) == ("stairs",)
  assert cfg.commands["twist"].velocity.fixed == (0.6, 0.0, 0.0)
  assert not cfg.observations["actor"].enable_corruption
  assert cfg.curriculum["terrain_levels"].params["play"]


def test_checkpoint_transfer_accepts_locomotion_and_rejects_other_task_family():
  from supermimic_lab.tasks.locomotion.velocity.runtime import LocomotionRuntime

  runtime = LocomotionRuntime.__new__(LocomotionRuntime)
  runtime.cfg = SimpleNamespace(continue_run=False, task="Go2-Locomotion-Rough")
  meta = {"format_version": 1, "task_family": "locomotion_velocity", "robot": GO2_SPEC.name,
          "backend": "mjlab", "history": 5, "state_terms": GO2_SPEC.state_layout.terms,
          "state_frame": "base_height_above_ground;velocities_and_limbs_in_yaw_frame",
          "action_names": [str(i) for i in range(12)], "obs_terms": st.OBS_TERMS,
          "task": "Go2-Locomotion-Flat"}
  runtime.checkpoint_metadata = lambda: meta
  assert runtime.checkpoint_terms({"metadata": meta}) == st.OBS_TERMS
  with pytest.raises(ValueError, match="task_family"):
    runtime.checkpoint_terms({"metadata": {**meta, "task_family": "chase_dodge"}})
  runtime.cfg.continue_run = True
  with pytest.raises(ValueError, match="same task"):
    runtime.checkpoint_terms({"metadata": meta})
