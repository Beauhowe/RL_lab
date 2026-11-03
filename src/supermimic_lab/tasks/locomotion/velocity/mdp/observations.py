"""Locomotion observations computed in the same yaw frame as imagination."""


def height_scan(env):
  return env.command_manager.get_term("twist").scan()
