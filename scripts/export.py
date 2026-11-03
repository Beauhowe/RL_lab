"""Export an actor-only or full checkpoint to a TorchScript policy."""

import argparse
import json
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  import torch
  from supermimic_lab.models.policy import load_actor

  checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  meta = checkpoint.get("metadata", {})
  if meta.get("algorithm") == "ppo":
    obs_dim, action_dim = meta["obs_dim"], meta["action_dim"]
  else:
    obs_dim = checkpoint["actor"]["norm.mean"].shape[0]
    hidden = tuple(checkpoint["cfg"]["actor_hidden"])
    action_dim = checkpoint["actor"][f"net.{2 * len(hidden)}.weight"].shape[0]
  actor = load_actor(checkpoint, args.device)
  with torch.inference_mode():
    policy = torch.jit.trace(actor, torch.zeros(1, obs_dim, device=args.device))
  metadata = {"obs_dim": obs_dim, "action_dim": action_dim,
              "metadata": checkpoint.get("metadata"), "cfg": checkpoint["cfg"]}
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.jit.save(policy, str(output), _extra_files={"metadata.json": json.dumps(metadata)})
  print(f"exported {output} ({obs_dim} inputs, {action_dim} actions)")


if __name__ == "__main__":
  main()
