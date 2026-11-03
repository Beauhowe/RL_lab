"""Evaluate a registered task and optionally record video."""

import argparse

MODE_PROBS = ("chase", "dodge", "both", "mixed")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--mode", choices=list(MODE_PROBS), default=None)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--video", default=None, help="Write an mp4 of env 0 here.")
  ap.add_argument("--difficulty", type=float, default=1.0,
                  help="Curriculum progress in [0, 1] for the targets and terrain rows (1: hardest).")
  ap.add_argument("--terrain", default=None,
                  help="Select terrain (e.g. stairs, boxes); locomotion applies it to all environments.")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--robot", default="unitree_go2")
  ap.add_argument("--backend", choices=["mjlab", "isaaclab"], default="mjlab")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW_RATE"),
                   help="Fixed velocity command for locomotion")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  from supermimic_lab.tasks.registry import registration, resolve

  if args.num_envs < 1 or args.seconds <= 0 or not 0 <= args.difficulty <= 1:
    ap.error("num-envs/seconds must be positive and difficulty must be in [0, 1]")
  spec = registration(args.task, args.robot, backend=args.backend)
  resolve(spec.evaluation_factory)(args)


if __name__ == "__main__":
  main()
