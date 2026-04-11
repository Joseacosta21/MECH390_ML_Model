"""
CLI entry point: run the surrogate optimizer and print a pre-manufacturing report.

Workflow
--------
1. Load surrogate checkpoint + generate/optimize configs.
2. Run differential_evolution over the 10-dim geometry space.
3. Save top-N candidates to JSON (for re-use by summarize_results.py standalone).
4. Run full physics validation on every candidate and print a unified report
   (geometry + surrogate-predicted vs physics-actual performance table).

No separate summarize_results.py run is needed after this script — the report
is printed inline.  summarize_results.py can still be run standalone later to
re-validate the saved JSON without re-optimizing.

Usage
-----
    python scripts/optimize_design.py \\
        --generate-config configs/generate/baseline.yaml \\
        --optimize-config configs/optimize/search.yaml \\
        --model data/models/surrogate_best.pt
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

from mech390.ml.optimize import find_dataset_best, run_optimization
import summarize_results as _report


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
        description='Surrogate optimizer for crank-slider mechanism design.'
    )
    p.add_argument('--generate-config', '-g', default='configs/generate/baseline.yaml',
        help='Generate config supplying geometry bounds.')
    p.add_argument('--optimize-config', '-o', default='configs/optimize/search.yaml',
        help='Optimize config with weight table and constraint thresholds.')
    p.add_argument('--model', '-m', default='data/models/surrogate_best.pt',
        help='Path to trained surrogate checkpoint (.pt).')
    p.add_argument('--scaler', default='data/models/scaler.pkl',
        help='Path to scaler pickle.')
    p.add_argument('--stats', default='data/models/target_stats.json',
        help='Path to target_stats.json.')
    p.add_argument('--seed', '-s', type=int, default=None,
        help='Global random seed.')
    p.add_argument('--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    p.add_argument('--out-json', default='data/results/candidates.json', metavar='PATH',
        help='Write top-N candidates to this JSON (for standalone re-validation later).')
    p.add_argument('--top', '-n', type=int, default=3, metavar='N',
        help='Number of top candidates to report.')
    p.add_argument('--pass-csv', default='data/preview/passed_configs.csv', metavar='PATH',
        help='Path to passed_configs.csv for dataset-best comparison.')
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Config not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


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

    import json as _json

    results = run_optimization(
        gen_cfg     = gen_cfg,
        opt_cfg     = opt_cfg,
        checkpoint  = args.model,
        scaler_path = args.scaler,
        stats_path  = args.stats,
    )

    # Append dataset best for side-by-side comparison in the report
    with open(args.stats) as _fh:
        _target_stats = _json.load(_fh)
    db = find_dataset_best(args.pass_csv, opt_cfg['objectives'], _target_stats)
    if db is not None:
        results.append(db)

    # Save JSON for standalone re-validation
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    logging.getLogger('optimize_design').info("Candidates saved → %s", out_json)

    # Unified pre-manufacturing report (physics validation inline)
    _report.run(
        candidates_path = str(out_json),
        config_path     = args.generate_config,
        top_n           = args.top,
    )


if __name__ == '__main__':
    main()
