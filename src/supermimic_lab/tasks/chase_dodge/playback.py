"""Task-specific playback settings for chase/dodge."""


def configure_play(env_cfg, args):
  modes = {"chase": (1.0, 0.0, 0.0), "dodge": (0.0, 1.0, 0.0),
           "both": (0.0, 0.0, 1.0), "mixed": (0.3, 0.3, 0.4)}
  target = env_cfg.commands["target"]
  target.mode_probs = modes[args.mode]
  target.target.curriculum_steps = 10_000
  if "terrain_levels" in env_cfg.curriculum:
    env_cfg.curriculum["terrain_levels"].params["steps"] = 10_000
  env_cfg.viewer.distance, env_cfg.viewer.elevation = 4.0, -20.0
