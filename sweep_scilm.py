"""Random-search hyperparameter sweep over SciLM.

Edit SWEEP_SPEC below to change what's searched. Each trial calls train.train_one
with a sampled config, writes a per-trial subdirectory under runs/<sweep-id>/, and
appends one row to runs/<sweep-id>/summary.csv with the config and final
val metrics.

Sweep epoch budget defaults to 10 (shorter than the v1 baseline's 40) so that
20 trials fit overnight on M1. Override with --sweep-epochs.

Examples:
    python sweep_scilm.py --sweep-id v2_random20
    python sweep_scilm.py --sweep-id v2_quick --n-trials 5 --sweep-epochs 5
    python sweep_scilm.py --sweep-id v2_random20 --resume   # skip already-done trials
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scilm.config import Config, make_config
from scilm.train import train_one


# Hyperparameter axes to sample from. Edit freely.
SWEEP_SPEC = {
    'lr':              [1e-4, 3e-4, 1e-3, 3e-3],
    'dropout':         [0.0, 0.1, 0.2, 0.3],
    'lambda_scalar':   [0.3, 1.0, 3.0],
    'batch_size':      [32, 64, 128],
    'weight_decay':    [0.0, 0.01, 0.1],
    # architecture sizes (sampled together as a tier so n_heads divides d_model)
    'arch':            [
        ('small',  dict(d_model=128, n_layers=4, n_heads=4, d_ff=512)),
        ('medium', dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024)),
    ],
    'min_max_masked':  [(1, 2), (1, 3), (2, 4)],
}


def sample_config(rng: random.Random, base_overrides: dict) -> tuple[Config, dict]:
    """Sample one config; return (cfg, sampled_dict_for_logging)."""
    sampled = {}
    overrides = dict(base_overrides)
    for key, choices in SWEEP_SPEC.items():
        choice = rng.choice(choices)
        if key == 'arch':
            tier_name, arch = choice
            sampled['arch_tier'] = tier_name
            overrides.update(arch)
        elif key == 'min_max_masked':
            mn, mx = choice
            sampled['min_pairs_masked'] = mn
            sampled['max_pairs_masked'] = mx
            overrides['min_pairs_masked'] = mn
            overrides['max_pairs_masked'] = mx
        else:
            sampled[key] = choice
            overrides[key] = choice
    return make_config('full', **overrides), sampled


def trial_name(idx: int, sampled: dict) -> str:
    parts = [f'{idx:03d}']
    if 'arch_tier' in sampled:
        parts.append(sampled['arch_tier'])
    if 'lr' in sampled:
        parts.append(f"lr{sampled['lr']:.0e}")
    return '_'.join(parts)


SUMMARY_COLUMNS = [
    'trial', 'name', 'arch_tier', 'd_model', 'n_layers', 'n_heads', 'd_ff',
    'lr', 'dropout', 'lambda_scalar', 'batch_size', 'weight_decay',
    'min_pairs_masked', 'max_pairs_masked', 'n_epochs', 'n_params',
    'best_val_total', 'final_val_total', 'final_val_ce', 'final_val_mse',
    'wall_seconds', 'median_samples_per_sec', 'min_samples_per_sec',
    'max_samples_per_sec',
]


def append_summary_row(summary_path: Path, row: dict):
    write_header = not summary_path.exists()
    with summary_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in SUMMARY_COLUMNS})


def main():
    p = argparse.ArgumentParser(description='SciLM random-search sweep.')
    p.add_argument('--sweep-id', required=True)
    p.add_argument('--n-trials', type=int, default=20)
    p.add_argument('--sweep-epochs', type=int, default=10,
                   help='n_epochs to use for each trial (default 10).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--resume', action='store_true',
                   help='skip trials whose model.pt already exists.')
    p.add_argument('--device', default=None,
                   help='cpu | mps | cuda | auto (default: auto = cuda>cpu)')
    args = p.parse_args()
    device = None if (args.device is None or args.device == 'auto') else torch.device(args.device)

    sweep_dir = Path(args.runs_dir) / args.sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'sweep_spec.json').write_text(json.dumps({
        'spec': {k: [list(v) if isinstance(v, tuple) else v for v in vs] if isinstance(vs, list) else vs
                 for k, vs in SWEEP_SPEC.items()},
        'n_trials': args.n_trials,
        'sweep_epochs': args.sweep_epochs,
        'seed': args.seed,
    }, indent=2, default=str))

    base_overrides = {'n_epochs': args.sweep_epochs}
    rng = random.Random(args.seed)
    summary_path = sweep_dir / 'summary.csv'

    for i in range(args.n_trials):
        cfg, sampled = sample_config(rng, base_overrides)
        name = trial_name(i, sampled)
        run_dir = sweep_dir / name

        if args.resume and (run_dir / 'model.pt').exists():
            print(f'[{i+1}/{args.n_trials}] {name}: resume — already done, skipping')
            continue

        print(f'\n[{i+1}/{args.n_trials}] {name}')
        print(f'  sampled={sampled}')
        t0 = time.time()
        history = train_one(cfg, run_dir, device=device)
        wall = time.time() - t0

        sps = history.get('samples_per_sec', [])
        sps_sorted = sorted(sps)
        median_sps = sps_sorted[len(sps_sorted)//2] if sps_sorted else 0.0
        row = {
            'trial': i, 'name': name, **asdict(cfg),
            'arch_tier': sampled.get('arch_tier', ''),
            'n_params': history['n_params'],
            'best_val_total': min(history['val_total']),
            'final_val_total': history['val_total'][-1],
            'final_val_ce':    history['val_ce'][-1],
            'final_val_mse':   history['val_mse'][-1],
            'wall_seconds': round(wall, 1),
            'median_samples_per_sec': round(median_sps, 1),
            'min_samples_per_sec': round(min(sps_sorted), 1) if sps_sorted else 0.0,
            'max_samples_per_sec': round(max(sps_sorted), 1) if sps_sorted else 0.0,
        }
        append_summary_row(summary_path, row)
        print(f'  done in {wall/60:.1f}m  '
              f'best_val_total={row["best_val_total"]:.4f}  '
              f'final={row["final_val_total"]:.4f}')

    print(f'\nsweep done. summary: {summary_path}')


if __name__ == '__main__':
    main()
