"""
optimize_design.py - runs surrogate optimizer and prints a pre-manufacturing report.

Usage: python scripts/optimize_design.py [--top N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from mech390.ml.optimize import find_dataset_best, run_optimization
import summarize_results as _report

_GEN_CFG    = _ROOT / 'configs' / 'generate' / 'baseline.yaml'
_OPT_CFG    = _ROOT / 'configs' / 'optimize' / 'search.yaml'
_CHECKPOINT = _ROOT / 'data' / 'models' / 'surrogate_best.pt'
_SCALER     = _ROOT / 'data' / 'models' / 'scaler.pkl'
_STATS      = _ROOT / 'data' / 'models' / 'target_stats.json'
_OUT_JSON   = _ROOT / 'data' / 'results' / 'candidates.json'
_PASS_CSV   = _ROOT / 'data' / 'preview' / 'passed_configs.csv'


def _parse_args():
    p = argparse.ArgumentParser(prog='optimize_design')
    p.add_argument('--top', '-n', type=int, default=3, metavar='N',
        help='Number of top candidates to report.')
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        print(f"Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = _parse_args()
    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s  %(levelname)-8s  %(name)s - %(message)s',
        datefmt = '%H:%M:%S',
    )

    gen_cfg = _load_yaml(_GEN_CFG)
    opt_cfg = _load_yaml(_OPT_CFG)

    results = run_optimization(
        gen_cfg     = gen_cfg,
        opt_cfg     = opt_cfg,
        checkpoint  = str(_CHECKPOINT),
        scaler_path = str(_SCALER),
        stats_path  = str(_STATS),
    )

    with open(_STATS) as fh:
        _target_stats = json.load(fh)
    db = find_dataset_best(str(_PASS_CSV), opt_cfg['objectives'], _target_stats)
    if db is not None:
        results.append(db)

    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)
    logging.getLogger('optimize_design').info("Candidates saved -> %s", _OUT_JSON)

    _report.run(
        candidates_path = str(_OUT_JSON),
        top_n           = args.top,
        config_path     = str(_GEN_CFG),
    )


if __name__ == '__main__':
    main()
