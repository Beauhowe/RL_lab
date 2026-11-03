"""The original three Go2 task presets."""


def make_env_cfg(task, *, play=False):
  from supermimic_lab.tasks.chase_dodge.chase_dodge_env_cfg import go2_chase_env_cfg

  variants = {
    "Go2-Chase-Flat": {},
    "Go2-Chase-Obstacles": {"obstacles": True},
    "Go2-Chase-Terrain": {"terrain": True},
  }
  return go2_chase_env_cfg(play=play, **variants[task])
