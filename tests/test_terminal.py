"""Reporting semantics: resumed-run ETA, phases, metric units and file output."""

from supermimic_lab.algorithms.supermimic.config import TrainCfg
from supermimic_lab.runners.terminal import TerminalLogger, duration, training_stage


def report(cfg, **overrides):
  values = dict(iteration=20_050, stage=training_stage(20_050, 300, 0),
                elapsed=30, window_seconds=30, window_iterations=50,
                env_steps=102_400, window_env_steps=102_400,
                stats={"wm_loss": 0.2468, "imagined_survival": 0.9987},
                metrics={"Metrics/locomotion/linear_error": 0.2,
                         "Metrics/hit_rate/ball": 0.15,
                         "Episode_Termination/fell_over": 2.5},
                learning_rates=(2e-4, 6e-5, 2e-4), archive_size=50, replay_size=100,
                nonfinite_steps=0, phase_seconds={"collect": 4, "world_model": 8, "policy": 18})
  TerminalLogger(cfg).report(**(values | overrides))


def test_pretty_report_resumed_eta_units_and_no_ansi(capsys):
  report(TrainCfg(iterations=40_000))
  out = capsys.readouterr().out
  # 19,950 remaining iterations * 0.6 s/iteration; do not divide by 20,050.
  assert "03:19:30" in out and "3,413" in out
  assert "99.87%" in out and "15.00%" in out and "0.2000 m/s" in out
  assert "mean count per reset batch" in out and "falls/min" not in out
  assert "\x1b" not in out


def test_compact_is_one_line_and_preserves_machine_searchable_keys(capsys):
  report(TrainCfg(log_format="compact"))
  out = capsys.readouterr().out
  assert len(out.splitlines()) == 1
  assert "wm_loss 0.2468" in out and "Episode_Termination/fell_over 2.5" in out


def test_warmup_does_not_display_fake_policy_losses(capsys):
  report(TrainCfg(), iteration=50, stats={"wm_loss": 1.2}, metrics={},
         stage=training_stage(50, 300, 0), window_seconds=0, window_env_steps=0)
  out = capsys.readouterr().out
  assert "World-model warmup" in out
  assert "Imagined return" not in out and "Critic loss" not in out
  assert "Episode rewards" not in out and "Terminations" not in out


def test_training_stage_boundaries_and_long_durations():
  assert "World-model warmup" in training_stage(300, 300, 800)
  assert "Actor frozen" in training_stage(301, 300, 800)
  assert "Actor frozen" in training_stage(800, 300, 800)
  assert training_stage(801, 300, 800) == "World model + Actor + Critic"
  assert duration(100 * 3600 + 61) == "100:01:01"
  assert duration(float("inf")) == "--:--:--"
