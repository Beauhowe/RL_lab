"""Go2 environment wiring shared across locomotion and chase/dodge tasks."""

import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from supermimic_lab.backends.mjlab.compat import quiet_solver_warnings as _quiet_solver_warnings
from .robot_cfg import FOOT_GEOM_NAMES, GO2_ACTION_SCALE, get_go2_robot_cfg
from .spec import GO2_SPEC

FOOT_SITE_NAMES = GO2_SPEC.foot_sites

def configure_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Go2 entity, sim settings, foot contact sensing and domain randomisation."""
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
  cfg.scene.sensors += (ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  ),)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = GO2_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Per-axis foot friction for condim 6, as in mjlab's Go1 config.
  del cfg.events["foot_friction"]
  for name, axis, dist, rng in (
    ("foot_friction_slide", 0, "uniform", (0.3, 1.5)),
    ("foot_friction_spin", 1, "log_uniform", (1e-4, 2e-2)),
    ("foot_friction_roll", 2, "log_uniform", (1e-5, 5e-3)),
  ):
    cfg.events[name] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "distribution": dist,
        "axes": [axis],
        "ranges": rng,
        "shared_random": True,
      },
    )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
  cfg.events["quiet_solver_warnings"] = EventTermCfg(
    mode="startup", func=_quiet_solver_warnings, params={}
  )
  # A deterministic world model cannot represent velocity kicks (SuperMimic A).
  cfg.events.pop("push_robot", None)

  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fell_over": TerminationTermCfg(func=vel_mdp.bad_orientation,
                                    params={"limit_angle": math.radians(70.0)}),
  }
