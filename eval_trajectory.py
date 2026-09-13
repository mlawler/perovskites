"""Trajectory analysis: replay every val-improvement checkpoint of a run and
track how predictions on a few featured formulations evolved.

Looks at runs/<run-name>/checkpoints/best_ep{N}.pt (saved by train.py whenever
val_total improved). For each checkpoint and each featured formulation:
  predicted_ionic[k] for k = the formulation's actual measurement rows
  residual[k] = predicted - measured (in log10 σ)
  → mean residual, std residual, MAE.

Output:
  - <out>.csv          per-checkpoint × per-formulation summary stats
  - <out>.png          two-panel plot:
      top: val_total vs epoch (black) with improvement-epoch markers
      bot: mean residual ± 1σ band per formulation, vs improvement epoch

Use this to answer: "are residuals improving smoothly across training, or do
they jump around?"

Usage:
    python eval_trajectory.py
    python eval_trajectory.py --run-dir runs/v3_long_100_seed1
    python eval_trajectory.py --formulation-ranks 0,3,11
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scilm.config import load_config
from scilm.data import Stats, load_split
from scilm.model import SciLM
from scilm.tokenizer import build_vocab_from_stoi

from eval_v2_figure import predict_ionic, short_formulation_label


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch, path), ...] sorted by epoch for files matching best_epNNN.pt."""
    pat = re.compile(r'best_ep(\d+)\.pt$')
    out = []
    for p in ckpt_dir.iterdir():
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='runs/v3_long_100_seed1')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--out', default=None,
                    help='output basename (default: trajectory_<run-name>)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--formulation-ranks', default='0,3,11',
                    help='comma-separated test-set ranks (by row count) to track')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    out_base = Path(args.out) if args.out else Path(f'trajectory_{run_dir.name}')

    cfg = load_config(run_dir / 'config.json')
    vocab_d = json.loads((run_dir / 'vocab.json').read_text())
    vocab = build_vocab_from_stoi(vocab_d['stoi'], vocab_d['L'])
    stats = Stats.load(run_dir / 'stats.json')
    history = json.loads((run_dir / 'history.json').read_text())

    data_root = args.data_root or cfg.data_root
    df_test = load_split(data_root, 'test')

    counts = df_test.groupby('smiles').size().sort_values(ascending=False)
    ranks = [int(r) for r in args.formulation_ranks.split(',')]
    chosen = []
    for r in ranks:
        smi = counts.index[r]
        sub = df_test[df_test['smiles'] == smi].sort_values('temperature')
        rows = [sub.iloc[i] for i in range(len(sub))]
        meas = sub['ionic'].to_numpy(dtype=np.float32)
        Ts = sub['temperature'].to_numpy(dtype=np.float32)
        chosen.append({
            'rank': r,
            'label': short_formulation_label(smi),
            'rows': rows, 'meas': meas, 'Ts': Ts,
        })
        print(f'  rank {r}: {chosen[-1]["label"]}  ({len(meas)} measurement pts)')

    ckpts = list_checkpoints(run_dir / 'checkpoints')
    if not ckpts:
        raise SystemExit(f'no checkpoints in {run_dir}/checkpoints/ — '
                         f'this run was trained before per-improvement '
                         f'checkpointing was added.')
    print(f'[trajectory] {len(ckpts)} improvement checkpoints found '
          f'(epochs {[ep for ep,_ in ckpts]})')

    # build a fresh model once, then reload its weights for each checkpoint
    model = SciLM(cfg, vocab).to(device)
    model.eval()

    rows_csv = ['epoch,rank,label,n,mean_resid,std_resid,mae']
    # per-formulation residuals over checkpoints: shape (len(ckpts), n_pts)
    resid_series = [np.zeros((len(ckpts), len(c['meas']))) for c in chosen]
    epochs_arr = np.array([ep for ep, _ in ckpts], dtype=int)

    for ci, (ep, path) in enumerate(ckpts):
        model.load_state_dict(torch.load(path, map_location=device))
        for fi, c in enumerate(chosen):
            pred = predict_ionic(model, c['rows'], vocab, stats, cfg, device)
            resid = pred - c['meas']
            resid_series[fi][ci] = resid
            rows_csv.append(
                f'{ep},{c["rank"]},"{c["label"]}",{len(resid)},'
                f'{resid.mean():.4f},{resid.std():.4f},{np.abs(resid).mean():.4f}'
            )
        print(f'  ep {ep:>3}: ' + '   '.join(
            f'{c["label"][:18]:<18} mean={resid_series[fi][ci].mean():+.3f} '
            f'mae={np.abs(resid_series[fi][ci]).mean():.3f}'
            for fi, c in enumerate(chosen)
        ))

    csv_path = out_base.with_suffix('.csv')
    csv_path.write_text('\n'.join(rows_csv) + '\n')
    print(f'[done] wrote {csv_path}')

    # --- plot ---
    fig, (axT, axR) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.6]})

    # top: val_total trajectory
    val_eps = np.arange(1, len(history['val_total']) + 1)
    axT.plot(val_eps, history['val_total'], '-', color='0.4', lw=1, label='val_total')
    axT.scatter(epochs_arr,
                [history['val_total'][e - 1] for e in epochs_arr],
                color='red', s=22, zorder=3, label='improvement epoch')
    axT.set_ylabel('val_total')
    axT.set_title(f'Run: {run_dir.name}  ({len(ckpts)} improvements over '
                  f'{len(history["val_total"])} epochs)')
    axT.grid(alpha=0.3)
    axT.legend(loc='upper right', fontsize=9)

    # bottom: residual evolution per formulation
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    axR.axhline(0, color='k', lw=0.6, alpha=0.5)
    for fi, c in enumerate(chosen):
        col = colors[fi % len(colors)]
        means = resid_series[fi].mean(axis=1)
        stds  = resid_series[fi].std(axis=1)
        axR.plot(epochs_arr, means, '-o', color=col, lw=1.6, ms=5, label=c['label'])
        axR.fill_between(epochs_arr, means - stds, means + stds,
                         color=col, alpha=0.18)
    axR.set_xlabel('epoch (improvement epochs only)')
    axR.set_ylabel('residual = predicted − measured  (log10 σ)')
    axR.legend(loc='best', fontsize=9)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    png_path = out_base.with_suffix('.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'[done] wrote {png_path}')


if __name__ == '__main__':
    main()
