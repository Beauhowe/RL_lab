"""Training CLI with TOML presets and explicit command-line overrides."""

import argparse
from dataclasses import asdict
from pathlib import Path

from supermimic_lab.algorithms.supermimic.config import TrainCfg
from supermimic_lab.algorithms.registry import ALGORITHMS, config_class, validate_experiment


def boolean(value):
  if value.lower() in ("1", "true"):
    return True
  if value.lower() in ("0", "false"):
    return False
  raise argparse.ArgumentTypeError("expected true or false")


def read_toml(path):
  try:
    import tomllib
  except ModuleNotFoundError:
    import tomli as tomllib
  with Path(path).open("rb") as stream:
    return tomllib.load(stream)


def load_config(path, cfg_class=TrainCfg):
  values = read_toml(path)
  defaults = asdict(cfg_class())
  unknown = set(values) - defaults.keys()
  if unknown:
    raise ValueError(f"unknown experiment settings: {', '.join(sorted(unknown))}")
  for name, value in values.items():
    default = defaults[name]
    if isinstance(default, tuple):
      valid = isinstance(value, list) and all(type(v) is int for v in value)
      if valid:
        values[name] = tuple(value)
    elif isinstance(default, float):
      valid = type(value) in (int, float)
      if valid:
        values[name] = float(value)
    else:
      valid = type(value) is type(default)
    if not valid:
      raise ValueError(f"invalid value type for {name}")
  return values


def parse_train_cfg(argv=None):
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--config", help="Path to a TOML experiment preset")
  pre.add_argument("--algorithm", choices=list(ALGORITHMS))
  args, _ = pre.parse_known_args(argv)
  parser = argparse.ArgumentParser(parents=[pre], description="Train with SuperMimic Lab")
  try:
    preset = read_toml(args.config) if args.config else {}
    selected = args.algorithm or preset.get("algorithm", "supermimic")
    cfg_class = config_class(selected)
  except (OSError, ValueError) as error:
    parser.error(str(error))
  defaults = asdict(cfg_class())
  if args.config:
    try:
      defaults.update(load_config(args.config, cfg_class))
    except (OSError, ValueError) as error:
      parser.error(str(error))
  parser.set_defaults(algorithm=selected)
  for name, value in defaults.items():
    if name == "algorithm":
      continue
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, tuple):
      parser.add_argument(flag, type=int, nargs="+", default=value)
    elif isinstance(value, bool):
      parser.add_argument(flag, type=boolean, default=value)
    else:
      parser.add_argument(flag, type=type(value), default=value)
  values = vars(parser.parse_args(argv))
  values.pop("config")
  values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
  try:
    cfg = cfg_class(**values)
    validate_experiment(cfg)
    return cfg
  except ValueError as error:
    parser.error(str(error))


def main(argv=None):
  cfg = parse_train_cfg(argv)
  # Isaac Sim must start before importing simulation environments or torch RL code.
  from supermimic_lab.backends.registry import launch_backend
  app = launch_backend(cfg)
  runner = None
  try:
    import torch
    from supermimic_lab.algorithms.registry import make_runner

    torch.set_float32_matmul_precision("high")
    runner = make_runner(cfg)
    runner.run()
  finally:
    try:
      if runner is not None:
        runner.close()
    finally:
      if app is not None:
        app.close()
