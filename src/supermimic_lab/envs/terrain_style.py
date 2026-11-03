"""Shared terrain materials and lighting."""

from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

GROUND_RGBA = (0.62, 0.62, 0.61, 1.0)
GRID_RGB = ((0.66, 0.66, 0.65), (0.62, 0.62, 0.61), (0.46, 0.47, 0.50))  # cells, lines
SKY_RGB = ((0.93, 0.95, 0.98), (0.62, 0.74, 0.87))


def clean_look(terrain: TerrainEntityCfg) -> TerrainEntityCfg:
  """Light grey matte ground with a faint 0.5 m grid (its lines break at step
  edges and bend over slopes, so the terrain's shape reads on video), a sky
  gradient and soft fill light."""
  terrain.textures = (
    TextureCfg(name="sky", type="skybox", builtin="gradient", rgb1=SKY_RGB[0], rgb2=SKY_RGB[1],
               width=512, height=3072),
    TextureCfg(name="grid", type="2d", builtin="checker", mark="edge", rgb1=GRID_RGB[0],
               rgb2=GRID_RGB[1], markrgb=GRID_RGB[2], width=512, height=512),
  )
  terrain.materials = (
    MaterialCfg(name="ground", texture="grid", texuniform=True, texrepeat=(1.0, 1.0),
                reflectance=0.0, geom_names_expr=(".*",)),
  )
  # Fill light only: the shadow-casting sun follows the robot (assets/robots/unitree_go2/robot.xml),
  # so its shadow map stays sharp wherever the robot runs.
  terrain.lights = (
    LightCfg(name="fill", type="directional", pos=(0.0, 0.0, 10.0), dir=(0.3, 0.2, -1.0),
             diffuse=(0.12, 0.12, 0.12), specular=(0.0, 0.0, 0.0), ambient=(0.12, 0.13, 0.15),
             castshadow=False),
  )
  return terrain
