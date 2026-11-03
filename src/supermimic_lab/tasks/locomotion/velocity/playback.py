"""Evaluation/play settings for fixed velocity commands and selected terrain."""


def configure_play(env_cfg, args):
  command = env_cfg.commands["twist"]
  command.velocity.curriculum_steps = 10_000
  value = getattr(args, "command", None)
  command.velocity.fixed = tuple(value) if value is not None else (1.0, 0.0, 0.0)
  env_cfg.seed = getattr(args, "seed", 0)
  if "terrain_levels" in env_cfg.curriculum:
    env_cfg.curriculum["terrain_levels"].params["steps"] = 10_000
  selected = getattr(args, "terrain", None)
  generator = env_cfg.scene.terrain.terrain_generator
  # Interactive play leaves num_envs unset by default. Evaluation always supplies it.
  if getattr(args, "num_envs", 1) is None and generator is not None and not selected:
    env_cfg.scene.num_envs = len(generator.sub_terrains)
  if selected:
    if generator is None:
      if selected != "flat":
        raise ValueError("select Go2-Locomotion-Rough to play stairs, slopes or rough terrain")
    else:
      if selected not in generator.sub_terrains:
        raise ValueError(f"unknown terrain {selected!r}; choose from {list(generator.sub_terrains)}")
      sub = generator.sub_terrains[selected]
      sub.proportion = 1.0
      generator.sub_terrains = {selected: sub}
      generator.num_cols = 1
  env_cfg.viewer.distance, env_cfg.viewer.elevation = 3.0, -20.0
  if (getattr(args, "viewer", None) is not None and generator is not None
      and not selected and env_cfg.scene.num_envs >= len(generator.sub_terrains)):
    # Center the lineup and render every dog, including in the native viewer.
    env_cfg.viewer.env_idx = env_cfg.scene.num_envs // 2
    env_cfg.viewer.max_extra_envs = env_cfg.scene.num_envs - 1
    env_cfg.viewer.distance = max(12.0, generator.size[1] * len(generator.sub_terrains))
    env_cfg.viewer.azimuth, env_cfg.viewer.elevation = 0.0, -45.0
