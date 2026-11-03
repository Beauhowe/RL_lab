"""Optional MuJoCo Warp capabilities across supported mjlab installations."""


def quiet_solver_warnings(env, env_ids):
  """Disable iteration-limit warnings when Warp exposes that option.

  MuJoCo Warp 3.11 has neither OverflowType nor opt.warn_overflow. Warning
  suppression is optional; its absence must not prevent environment startup.
  """
  import mujoco_warp

  overflow = getattr(mujoco_warp, "OverflowType", None)
  opt = env.sim.wp_model.opt
  if overflow is None or not hasattr(opt, "warn_overflow"):
    return
  opt.warn_overflow = int(opt.warn_overflow) & ~int(
    overflow.ITERATIONS | overflow.LS_ITERATIONS
  )
  # The option is captured by the solver kernels in CUDA graphs.
  env.sim.create_graph()
