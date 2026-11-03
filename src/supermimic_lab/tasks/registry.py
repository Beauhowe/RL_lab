"""Task factories use lazy imports so CLI help does not initialize a simulator."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class TaskRegistration:
  robot: str
  backend: str
  config_factory: str
  runtime_factory: str
  evaluation_factory: str
  play_configurator: str = ""


TASKS = {
  task: TaskRegistration(
    robot="unitree_go2", backend="mjlab",
    config_factory="supermimic_lab.tasks.chase_dodge.config.unitree_go2:make_env_cfg",
    runtime_factory="supermimic_lab.tasks.chase_dodge.runtime:ChaseDodgeRuntime",
    evaluation_factory="supermimic_lab.tasks.chase_dodge.evaluation:evaluate",
    play_configurator="supermimic_lab.tasks.chase_dodge.playback:configure_play",
  )
  for task in ("Go2-Chase-Flat", "Go2-Chase-Obstacles", "Go2-Chase-Terrain")
}

TASKS.update({
  task: TaskRegistration(
    robot="unitree_go2", backend="mjlab",
    config_factory="supermimic_lab.tasks.locomotion.velocity.config.unitree_go2:make_env_cfg",
    runtime_factory="supermimic_lab.tasks.locomotion.velocity.runtime:LocomotionRuntime",
    evaluation_factory="supermimic_lab.tasks.locomotion.velocity.evaluation:evaluate",
    play_configurator="supermimic_lab.tasks.locomotion.velocity.playback:configure_play",
  ) for task in ("Go2-Locomotion-Flat", "Go2-Locomotion-Rough")
})

# The Isaac Lab variant uses the official native velocity environment. Its
# observation/reward contract is deliberately distinct from the mjlab variant.
ISAACLAB_TASKS = {
  task: TaskRegistration(
    robot="unitree_go2", backend="isaaclab",
    config_factory="supermimic_lab.backends.isaaclab.env:load_env_cfg",
    runtime_factory="", evaluation_factory="supermimic_lab.backends.isaaclab.play:evaluate", play_configurator="",
  ) for task in ("Go2-Locomotion-Flat", "Go2-Locomotion-Rough")
}


def configure_play(task, env_cfg, args):
  factory = registration(task).play_configurator
  if factory:
    resolve(factory)(env_cfg, args)


def resolve(path):
  module, name = path.split(":", 1)
  return getattr(import_module(module), name)


def registration(task, robot=None, *, backend="mjlab"):
  variants = {"mjlab": TASKS, "isaaclab": ISAACLAB_TASKS}
  if backend not in variants:
    raise ValueError(f"unsupported backend {backend!r}; choose mjlab or isaaclab")
  tasks = variants[backend]
  if task not in tasks:
    raise ValueError(f"unknown task {task!r} for {backend}; available tasks: {', '.join(tasks)}")
  spec = tasks[task]
  if robot is not None and robot != spec.robot:
    raise ValueError(f"{task} requires robot {spec.robot!r}, got {robot!r}")
  return spec


def load_env_cfg(task, *, play=False, robot=None, backend="mjlab"):
  spec = registration(task, robot, backend=backend)
  return resolve(spec.config_factory)(task, play=play)


def make_runtime(cfg):
  spec = registration(cfg.task, cfg.robot, backend=cfg.backend)
  if not spec.runtime_factory:
    raise ValueError(f"{cfg.backend}/{cfg.task} has no SuperMimic runtime")
  return resolve(spec.runtime_factory)(cfg)
