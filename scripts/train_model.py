"""
CLI entry point: train the CrankSlider surrogate model.

Usage
-----
    python scripts/train_model.py --config configs/train/surrogate.yaml
    python scripts/train_model.py --config configs/train/surrogate.yaml --log-level DEBUG
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mech390.ml.train import run_training


def _parse_args():
    p = argparse.ArgumentParser(description='Train the CrankSlider surrogate NN.')
    p.add_argument(
        '--config', '-c',
        default='configs/train/surrogate.yaml',
        help='Path to surrogate training config YAML (default: configs/train/surrogate.yaml)',
    )
    p.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = '%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt = '%H:%M:%S',
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_training(cfg)


if __name__ == '__main__':
    main()
