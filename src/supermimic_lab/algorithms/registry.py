"""Algorithm dispatch, independent of the selected simulator."""

from supermimic_lab.tasks.registry import registration, resolve

ALGORITHMS = {
  "supermimic": ("supermimic_lab.algorithms.supermimic.config:TrainCfg",
                  "supermimic_lab.runners.supermimic_runner:SuperMimicRunner"),
  "ppo": ("supermimic_lab.algorithms.ppo.config:PpoTrainCfg",
           "supermimic_lab.runners.ppo_runner:PpoRunner"),
}


def config_class(algorithm):
  if algorithm not in ALGORITHMS:
    raise ValueError(f"unknown algorithm {algorithm!r}; choose from {list(ALGORITHMS)}")
  return resolve(ALGORITHMS[algorithm][0])


def validate_experiment(cfg):
  registration(cfg.task, cfg.robot, backend=cfg.backend)
  if cfg.algorithm == "supermimic" and cfg.backend != "mjlab":
    raise ValueError("SuperMimic currently supports mjlab only; use algorithm=ppo for Isaac Lab")
  if cfg.algorithm == "ppo" and cfg.task not in ("Go2-Locomotion-Flat", "Go2-Locomotion-Rough"):
    raise ValueError("PPO is currently registered for Go2-Locomotion-Flat/Rough")
  if cfg.algorithm not in ALGORITHMS:
    raise ValueError(f"unknown algorithm {cfg.algorithm!r}")


def make_runner(cfg):
  validate_experiment(cfg)
  return resolve(ALGORITHMS[cfg.algorithm][1])(cfg)
