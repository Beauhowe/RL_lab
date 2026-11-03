"""Robot-specific presets for commanded locomotion."""


def make_env_cfg(task, *, play=False):
  from ...velocity_env_cfg import locomotion_env_cfg

  return locomotion_env_cfg(rough=task == "Go2-Locomotion-Rough", play=play)
