"""
CLI entry point: run the surrogate optimizer to find the best mechanism geometry.

Usage
-----
    python scripts/optimize_design.py \\
        --generate-config configs/generate/baseline.yaml \\
        --optimize-config configs/optimize/search.yaml \\
        --model data/models/surrogate_best.pt

The geometry bounds are read from --generate-config, so any change to
baseline.yaml is automatically respected here — no duplication.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mech390.ml.optimize import run_optimization


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _parse_args():
    p = argparse.ArgumentParser(
        description='Surrogate-based optimizer for crank-slider mechanism design.'
    )
    p.add_argument(
        '--generate-config', '-g',
        default='configs/generate/baseline.yaml',
        help='Generate config supplying geometry bounds (default: baseline.yaml)',
    )
    p.add_argument(
        '--optimize-config', '-o',
        default='configs/optimize/search.yaml',
        help='Optimize config with weight table and optimizer settings',
    )
    p.add_argument(
        '--model', '-m',
        default='data/models/surrogate_best.pt',
        help='Path to trained surrogate checkpoint (.pt)',
    )
    p.add_argument(
        '--scaler',
        default='data/models/scaler.pkl',
        help='Path to scaler pickle (default: data/models/scaler.pkl)',
    )
    p.add_argument(
        '--stats',
        default='data/models/target_stats.json',
        help='Path to target_stats.json (default: data/models/target_stats.json)',
    )
    p.add_argument(
        '--seed', '-s',
        type=int, default=None,
        help='Global random seed for reproducibility.',
    )
    p.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    p.add_argument(
        '--out-json',
        default='data/results/candidates.json',
        metavar='PATH',
        help='Write top-N candidate dicts to this JSON file (for summarize_results.py).',
    )
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Config not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


def _print_results(results):
    from mech390.ml import features as F
    sep = '─' * 72
    print(f'\n{sep}')
    print(f'  TOP-{len(results)} OPTIMISED GEOMETRIES')
    print(sep)
    for i, r in enumerate(results, 1):
        print(f'\n  Rank {i}  |  score={r["weighted_score"]:.4f}  |  pass_prob={r["pass_prob"]:.1%}')
        print('  Geometry:')
        for k in F.INPUT_FEATURES:
            unit = 'mm' if k not in ('pass_prob', 'weighted_score') else ''
            val  = r[k] * 1000 if k in ('r','l','e','width_r','thickness_r',
                                          'width_l','thickness_l',
                                          'd_shaft_A','pin_diameter_B','pin_diameter_C') else r[k]
            print(f'    {k:20s}: {val:8.3f} mm')
        print('  Predicted performance:')
        print(f'    {"total_mass":20s}: {r["pred_total_mass"]*1000:.1f} g')
        print(f'    {"volume_envelope":20s}: {r["pred_volume_envelope"]*1e6:.2f} cm³')
        print(f'    {"tau_A_max":20s}: {r["pred_tau_A_max"]:.4f} N·m')
        print(f'    {"E_rev":20s}: {r["pred_E_rev"]:.4f} J/rev')
        print(f'    {"min_n_static":20s}: {r["pred_min_n_static"]:.2f}')
        print(f'    {"utilization":20s}: {r["pred_utilization"]:.3f}')
        print(f'    {"n_buck":20s}: {r["pred_n_buck"]:.2f}')
    print(f'\n{sep}')
    print('  NOTE: Cross-check the top candidate through the full physics')
    print('  pipeline before committing to 3D printing.')
    print(sep)


def main():
    args = _parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = '%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt = '%H:%M:%S',
    )

    if args.seed is not None:
        _seed_everything(args.seed)

    gen_cfg = _load_yaml(args.generate_config)
    opt_cfg = _load_yaml(args.optimize_config)

    results = run_optimization(
        gen_cfg      = gen_cfg,
        opt_cfg      = opt_cfg,
        checkpoint   = args.model,
        scaler_path  = args.scaler,
        stats_path   = args.stats,
    )

    _print_results(results)

    # Write candidates to JSON so summarize_results.py can run physics validation
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    logging.getLogger('optimize_design').info("Candidates written to %s", out_json)


if __name__ == '__main__':
    main()
