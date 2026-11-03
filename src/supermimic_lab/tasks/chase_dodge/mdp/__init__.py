"""Environment observations, rewards and curricula."""

from .observations import height_scan, lidar
from .rewards import chase_approach, chase_tag, facing_target, evade_danger, evade_anticipate, hit, crouch, joint_deviation_l2, outside_arena, tilt_rate_l2, hard_landing, upright, base_height_low, obstacle_contact
from .curricula import terrain_levels_by_progress
