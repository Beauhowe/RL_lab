"""Play a trained SuperMimic actor in a browser or native MuJoCo viewer."""

import argparse
import math
import pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


class PlaybackEnv:
  """Reuse step observations so viewer reads do not advance observation history."""

  def __init__(self, env, obs):
    self.env, self.obs = env, obs

  def __getattr__(self, name):
    return getattr(self.env, name)

  def get_observations(self):
    return self.obs

  def step(self, actions):
    result = self.env.step(actions)
    self.obs = result[0]
    return result

  def reset(self, *args, **kwargs):
    result = self.env.reset(*args, **kwargs)
    self.obs = result[0]
    return result


def load_checkpoint(torch, args):
  if args.checkpoint:
    path = Path(args.checkpoint).expanduser().resolve()
    return path, torch.load(path, map_location="cpu", weights_only=False)
  algorithm = getattr(args, "algorithm", "supermimic")
  root = Path(args.run_dir).expanduser() if args.run_dir else PROJECT_DIR / "logs" / algorithm
  pattern = "model_*.pt" if args.run_dir else "*/model_*.pt"
  candidates = sorted((p for p in root.glob(pattern) if p.stem.removeprefix("model_").isdigit()), key=lambda p: (
    p.stat().st_mtime_ns, int(p.stem.removeprefix("model_"))), reverse=True)
  for path in candidates:
    if not args.run_dir and "smoke" in path.parent.name:
      continue
    try:
      checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, pickle.UnpicklingError):
      # A running trainer may still be writing its newest checkpoint.
      continue
    task = checkpoint.get("metadata", {}).get("task") or checkpoint.get("cfg", {}).get("task")
    meta = checkpoint.get("metadata", {})
    if (task == args.task and meta.get("algorithm", "supermimic") == algorithm
        and meta.get("backend", "mjlab") == getattr(args, "backend", "mjlab")):
      return path.resolve(), checkpoint
  raise FileNotFoundError(f"no training checkpoint for {args.task} under {root}; use --checkpoint")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Go2-Chase-Flat")
  parser.add_argument("--algorithm", choices=("supermimic", "ppo"), default="supermimic")
  parser.add_argument("--backend", choices=("mjlab", "isaaclab"), default="mjlab")
  parser.add_argument("--headless", action="store_true", help="Isaac Lab only: disable its native window")
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--checkpoint", help="Checkpoint to play; defaults to the latest matching run")
  source.add_argument("--run-dir", help="Select the latest matching checkpoint from this run")
  parser.add_argument("--mode", choices=("chase", "dodge", "both", "mixed"), default="both")
  parser.add_argument("--difficulty", type=float, default=1.0)
  parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW_RATE"),
                       help="Fixed body-frame velocity command for locomotion")
  parser.add_argument("--terrain", help="Locomotion terrain: flat, stairs, stairs_inv, slope, slope_inv, boxes, rough")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num-envs", type=int, default=None,
                       help="Defaults to one per terrain for mixed locomotion, otherwise 1")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--viewer", choices=("auto", "viser", "native"), default="auto")
  parser.add_argument("--port", type=int, default=8080, help="Local browser-viewer port")
  parser.add_argument("--seconds", type=float, default=None, help="Close after this much simulation time")
  args = parser.parse_args()
  if not 0 <= args.difficulty <= 1 or (args.num_envs is not None and args.num_envs < 1):
    parser.error("difficulty must be in [0, 1] and num-envs must be positive")
  if args.seconds is not None and args.seconds <= 0:
    parser.error("seconds must be positive")
  if not 1 <= args.port <= 65535:
    parser.error("port must be in [1, 65535]")
  if args.backend == "isaaclab":
    if args.algorithm != "ppo" or args.viewer == "viser":
      parser.error("Isaac Lab playback uses --algorithm ppo and its native viewer")
    from supermimic_lab.backends.isaaclab.play import play
    return play(args, load_checkpoint)
  if args.viewer == "auto":
    args.viewer = "viser"

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  from supermimic_lab.models.policy import load_actor
  from supermimic_lab.tasks.registry import load_env_cfg, configure_play

  path, checkpoint = load_checkpoint(torch, args)
  if checkpoint.get("metadata", {}).get("backend", "mjlab") != "mjlab":
    raise ValueError("use --backend isaaclab --algorithm ppo for this checkpoint")
  print(f"Playing checkpoint: {path}", flush=True)
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else 1
  configure_play(args.task, env_cfg, args)
  if args.viewer == "viser":
    # MuJoCo elevations look downward when negative; Viser uses positive height.
    env_cfg.viewer.elevation = abs(env_cfg.viewer.elevation)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  server = None
  try:
    env.common_step_counter = int(args.difficulty * 10_000)
    obs, _ = env.reset()
    terrain = env.scene.terrain
    terrain_rows = []
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
      names = list(generator.sub_terrains)
      for index, (kind, level) in enumerate(zip(terrain.terrain_types.tolist(), terrain.terrain_levels.tolist())):
        name = names[kind]
        terrain_rows.append(f"| {index} | {name} | {level} |")
        print(f"[Terrain] Dog {index}: {name}, difficulty row {level}", flush=True)
    actions = len(env.action_manager.get_term("joint_pos")._target_names)
    actor = load_actor(checkpoint, args.device)
    if actor(obs["actor"]).shape != (env.num_envs, actions):
      raise ValueError("checkpoint action shape differs from the playback environment")
    playback = PlaybackEnv(env, obs)
    policy = lambda observation: actor(observation["actor"])
    if args.viewer == "native":
      viewer = NativeMujocoViewer(playback, policy)
      print("Space: pause/resume | Enter: reset | -/=: playback speed | close window: quit", flush=True)
    else:
      import viser

      server = viser.ViserServer(host="127.0.0.1", port=args.port, label="SuperMimic Lab")
      if terrain_rows:
        with server.gui.add_folder("Terrain lineup"):
          server.gui.add_markdown("| Dog / Env | Terrain | Row |\n|---|---|---|\n" + "\n".join(terrain_rows))
          server.gui.add_markdown("All dogs run the same policy. Use **Controls → Scene → Environment → Select** "
                                  "to follow a dog. Leave **Hide others** unchecked to see all dogs.")
      viewer = ViserPlayViewer(playback, policy, viser_server=server)
      print("Open the local URL above in a browser; Ctrl+C in this terminal quits playback.", flush=True)
    steps = math.ceil(args.seconds / env.step_dt) if args.seconds is not None else None
    with torch.inference_mode():
      viewer.run(num_steps=steps)
    print("Playback completed.", flush=True)
  finally:
    if server is not None:
      server.stop()
    env.close()


if __name__ == "__main__":
  main()
