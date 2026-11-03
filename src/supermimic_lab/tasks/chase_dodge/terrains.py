"""Generated terrain and the scene look.

The mix follows Isaac Lab's rough-terrain set (pyramid stairs up and down, box
grids, random rough ground, slopes), with 12 m tiles that leave room for the
chase. One column per terrain type, five rows of rising difficulty: steps go
from 2 to 10 cm, boxes from +-2 to +-5 cm (neighbours differ by up to 10 cm),
and the flat platforms are small so the robot spends its time on the features. Sharp edges are where the world model is least
accurate, so robots start on the easiest row and harder rows open up during
training (mdp.terrain_levels_by_progress).

The look is a light matte ground under a sky gradient with a sun that casts
shadows, instead of mjlab's height-coloured terrain.
"""

from __future__ import annotations

import copy

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from supermimic_lab.envs.terrain_style import GROUND_RGBA, GRID_RGB, SKY_RGB, clean_look

ROUGH_TERRAINS = TerrainGeneratorCfg(
  seed=7,
  size=(12.0, 12.0),
  border_width=10.0,
  num_rows=5,
  num_cols=7,
  curriculum=True,
  color_scheme="none",
  sub_terrains={
    "flat": flat(proportion=0.10),
    "stairs": pyramid_stairs(proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.4,
                             platform_width=1.5),
    "stairs_inv": pyramid_stairs_inv(proportion=0.20, step_height_range=(0.02, 0.10),
                                     step_width=0.4, platform_width=1.5),
    "boxes": box_random_grid(proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.05),
                             platform_width=1.0),
    "rough": random_rough(proportion=0.15, noise_range=(0.01, 0.04), noise_step=0.01),
    "slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.3)),
    "slope_inv": hf_pyramid_slope_inv(proportion=0.10, slope_range=(0.0, 0.3)),
  },
  add_lights=False,
)



def rough_terrain() -> TerrainEntityCfg:
  return clean_look(TerrainEntityCfg(
    terrain_type="generator", terrain_generator=copy.deepcopy(ROUGH_TERRAINS),
    max_init_terrain_level=None,
  ))
