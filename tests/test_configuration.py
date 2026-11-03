"""Configuration and resource checks that do not require torch or a simulator."""

from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from supermimic_lab.assets.robots.unitree_go2.spec import GO2_SPEC
from supermimic_lab.envs.specs import DynamicsSpec, TensorLayout
from supermimic_lab.runners.cli import parse_train_cfg
from supermimic_lab.tasks.registry import TASKS, registration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", ["flat", "obstacles", "terrain"])
def test_experiment_presets_and_cli_precedence(variant):
  cfg = parse_train_cfg([
    "--config", str(ROOT / f"configs/experiments/go2_chase_{variant}.toml"),
    "--num-envs", "7", "--compile", "false", "--actor-hidden", "64", "32",
  ])
  assert cfg.task == f"Go2-Chase-{variant.capitalize()}"
  assert cfg.num_envs == 7 and not cfg.compile and cfg.actor_hidden == (64, 32)
  assert cfg.robot == "unitree_go2" and cfg.backend == "mjlab"


@pytest.mark.parametrize("text", ["typo = 3", 'num_envs = "many"', "compile = 1"])
def test_invalid_presets_fail_before_creating_environment(tmp_path, text):
  preset = tmp_path / "invalid.toml"
  preset.write_text(text)
  with pytest.raises(SystemExit):
    parse_train_cfg(["--config", str(preset)])


def test_numeric_preset_keeps_float_cli_type(tmp_path):
  preset = tmp_path / "lr.toml"
  preset.write_text("lr = 1")
  assert parse_train_cfg(["--config", str(preset), "--lr", "0.0003"]).lr == 0.0003


@pytest.mark.parametrize("args", [
  ["--buffer-len", "9"], ["--num-envs", "0"], ["--compile", "maybe"],
  ["--wm-prioritized-frac", "0.8", "--wm-archive-frac", "0.3"],
])
def test_invalid_training_sizes_and_flags(args):
  with pytest.raises(SystemExit):
    parse_train_cfg(args)


def test_robot_layout_preserves_legacy_go2_state():
  assert GO2_SPEC.state_layout.dim == 70
  assert GO2_SPEC.state_layout.slice("joint_pos") == slice(10, 22)
  assert GO2_SPEC.state_layout.slice("feet") == slice(58, 70)
  assert GO2_SPEC.action_dim == 12
  assert TensorLayout((("base", 3), ("arm", 7))).slice("arm") == slice(3, 10)
  with pytest.raises(ValueError):
    TensorLayout((("x", 1), ("x", 2)))
  with pytest.raises(ValueError):
    DynamicsSpec(10, 6, 0, 0, height_index=10)


def test_task_robot_compatibility_is_checked():
  assert {"Go2-Chase-Flat", "Go2-Chase-Obstacles", "Go2-Chase-Terrain",
          "Go2-Locomotion-Flat", "Go2-Locomotion-Rough"}.issubset(TASKS)
  with pytest.raises(ValueError, match="requires robot"):
    registration("Go2-Chase-Flat", "unitree_g1")
  with pytest.raises(ValueError, match="unknown task"):
    registration("unknown")


def test_robot_mesh_references_resolve_after_move():
  directory = ROOT / "src/supermimic_lab/assets/robots/unitree_go2"
  xml = ET.parse(directory / "robot.xml")
  meshdir = directory / xml.find("compiler").attrib["meshdir"]
  for mesh in xml.findall(".//mesh"):
    assert (meshdir / mesh.attrib["file"]).is_file()
  assert (directory / "LICENSE_menagerie_go2").is_file()


@pytest.mark.parametrize("script", ["train", "eval", "export"])
def test_entry_point_help_does_not_initialize_torch_or_simulator(script):
  env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
  result = subprocess.run([sys.executable, str(ROOT / f"scripts/{script}.py"), "--help"],
                          env=env, capture_output=True, text=True, timeout=10)
  assert result.returncode == 0, result.stderr
  assert "usage:" in result.stdout
