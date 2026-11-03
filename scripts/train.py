"""Train a chase / dodge policy with SuperMimic.

Flat ground, from scratch:
  python scripts/train.py --task Go2-Chase-Flat --run-name flat

Obstacles or terrain, warm-started from a flat policy (see README.md for the
terrain recipe with the MoE-CTS and flat-policy data drivers):
  python scripts/train.py --task Go2-Chase-Obstacles --resume <flat model.pt> --run-name obstacles

Every TrainCfg field (src/supermimic_lab/algorithms/supermimic/config.py) is a flag.
"""

from supermimic_lab.runners.cli import main

if __name__ == "__main__":
  main()
