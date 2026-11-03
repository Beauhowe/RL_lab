"""Go2 kinematic layout. No simulator or neural-network imports."""

from supermimic_lab.envs.specs import RobotSpec, TensorLayout

LEGS = ("FR", "FL", "RR", "RL")
GO2_SPEC = RobotSpec(
  name="unitree_go2",
  state_layout=TensorLayout((
    ("height", 1), ("gravity", 3), ("linear_velocity", 3),
    ("angular_velocity", 3), ("joint_pos", 12), ("joint_vel", 12),
    ("hips", 12), ("knees", 12), ("feet", 12),
  )),
  action_dim=12,
  limb_bodies=(
    ("hips", tuple(f"{leg}_thigh" for leg in LEGS)),
    ("knees", tuple(f"{leg}_calf" for leg in LEGS)),
  ),
  foot_sites=LEGS,
  foot_geoms=tuple(f"{leg}_foot_collision" for leg in LEGS),
)
