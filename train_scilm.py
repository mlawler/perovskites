"""CLI: train a single SciLM model.

Examples:
    python train_scilm.py --mode sanity --name scilm_smoke
    python train_scilm.py --mode full   --name v2_baseline
    python train_scilm.py --mode full   --lr 1e-3 --dropout 0.2 --name v2_lr1e3_dp02
"""
from __future__ import annotations
import argparse
import dataclasses
import typing
from pathlib import Path

import torch

from scilm.config import Config, make_config
from scilm.train import train_one, pick_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train a single SciLM model.')
    p.add_argument('--mode', choices=['sanity', 'full'], default='full')
    p.add_argument('--name', required=True, help='run name; artifacts go to runs/<name>/')
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--device', default=None,
                   help='cpu | mps | cuda | auto (default: auto = mps>cuda>cpu)')

    # any Config field can be overridden from the CLI; types come from the resolved hints
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        t = hints[f.name]
        flag = f'--{f.name.replace("_", "-")}'
        if t is bool:
            p.add_argument(flag, dest=f.name,
                           action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, dest=f.name, type=t, default=None)
    return p


def main():
    args = build_parser().parse_args()
    overrides = {f.name: getattr(args, f.name)
                 for f in dataclasses.fields(Config)
                 if getattr(args, f.name) is not None}
    cfg = make_config(args.mode, **overrides)

    save_dir = Path(args.runs_dir) / args.name
    print(f'training mode={args.mode} -> {save_dir}')
    print(cfg)
    device = None if (args.device is None or args.device == 'auto') else torch.device(args.device)
    history = train_one(cfg, save_dir, device=device)

    print(f'\ndone. final val_total={history["val_total"][-1]:.4f} '
          f'val_ce={history["val_ce"][-1]:.4f} val_mse={history["val_mse"][-1]:.4f}')
    print(f'artifacts in {save_dir}/')


if __name__ == '__main__':
    main()
