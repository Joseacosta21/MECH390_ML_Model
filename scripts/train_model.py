"""
train_model.py - trains the surrogate model using configs/train/surrogate.yaml.

Usage: python scripts/train_model.py
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mech390.ml.train import run_training

_CFG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'train' / 'surrogate.yaml'


def _parse_args():
    p = argparse.ArgumentParser(prog='train_model')
    p.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = '%(asctime)s  %(levelname)-8s  %(name)s - %(message)s',
        datefmt = '%H:%M:%S',
    )

    if not _CFG_PATH.exists():
        print(f"Config not found: {_CFG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(_CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    run_training(cfg)


if __name__ == '__main__':
    main()
