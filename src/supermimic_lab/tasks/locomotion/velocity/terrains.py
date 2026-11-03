"""Equal terrain columns ensure the training scene contains every terrain kind."""

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
  flat, pyramid_stairs, pyramid_stairs_inv, box_random_grid,
  random_rough, hf_pyramid_slope, hf_pyramid_slope_inv,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from supermimic_lab.envs.terrain_style import clean_look


def make_terrain():
  proportion = 1.0 / 7
  generator = TerrainGeneratorCfg(
    seed=7, size=(8.0, 8.0), border_width=5.0, num_rows=5, num_cols=7,
    curriculum=True, color_scheme="none", add_lights=False,
    sub_terrains={
      "flat": flat(proportion=proportion),
      "stairs": pyramid_stairs(proportion=proportion, step_height_range=(0.02, 0.12),
                                step_width=0.35, platform_width=1.4),
      "stairs_inv": pyramid_stairs_inv(proportion=proportion, step_height_range=(0.02, 0.12),
                                        step_width=0.35, platform_width=1.4),
      "slope": hf_pyramid_slope(proportion=proportion, slope_range=(0.0, 0.35)),
      "slope_inv": hf_pyramid_slope_inv(proportion=proportion, slope_range=(0.0, 0.35)),
      "boxes": box_random_grid(proportion=proportion, grid_width=0.45,
                                grid_height_range=(0.02, 0.06), platform_width=1.0),
      "rough": random_rough(proportion=proportion, noise_range=(0.01, 0.04), noise_step=0.01),
    },
  )
  return clean_look(TerrainEntityCfg(terrain_type="generator", terrain_generator=generator,
                                    max_init_terrain_level=0))
