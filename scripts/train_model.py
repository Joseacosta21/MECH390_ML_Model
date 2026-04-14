"""
CLI entry point: train the CrankSlider surrogate model.

Usage
-----
    python scripts/train_model.py --config configs/train/surrogate.yaml
    python scripts/train_model.py --config configs/train/surrogate.yaml --log-level DEBUG
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mech390.ml.train import run_training


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _parse_args():
    p = argparse.ArgumentParser(description='Train the CrankSlider surrogate NN.')
    p.add_argument(
        '--config', '-c',
        default='configs/train/surrogate.yaml',
        help='Path to surrogate training config YAML (default: configs/train/surrogate.yaml)',
    )
    p.add_argument(
        '--seed', '-s',
        type=int, default=None,
        help='Global random seed for reproducibility (overrides config seed if set).',
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

    if args.seed is not None:
        _seed_everything(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_training(cfg)


if __name__ == '__main__':
    main()
