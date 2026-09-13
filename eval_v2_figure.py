"""Produce the v2 model show-off figure: predict masked conductivity on the test set.

Loads runs/<run-name>/, masks ionic conductivity (slot 7) on every test row,
runs one forward pass, and writes a 2-panel PNG:
  Left  - predicted vs measured σ on the full test set, log-log, colored by T.
  Right - measured points + model T-sweep curve for a few chosen formulations.

Convention (see project memory `project_ionic_units`): the dataset's `ionic`
column is log10(σ) with σ in µS/cm. The paper reports σ in mS/cm, so for
display we convert: σ_mS_per_cm = 10^(ionic - 3).

Usage:
    python eval_v2_figure.py
    python eval_v2_figure.py --run-dir runs/v2_winner_full --out fig_v2_test.png
    python eval_v2_figure.py --formulation-ranks 0,3,11
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scilm.config import Config, load_config
from scilm.data import Stats, load_split
from scilm.model import SciLM
from scilm.tokenizer import Vocab, build_vocab_from_stoi


PAIR_COMPONENT, PAIR_TEMP, PAIR_ION = 0, 1, 2


def load_run(run_dir: Path, device: torch.device,
             checkpoint: str = 'model.pt') -> tuple[SciLM, Config, Vocab, Stats]:
    cfg = load_config(run_dir / 'config.json')
    vocab_d = json.loads((run_dir / 'vocab.json').read_text())
    vocab = build_vocab_from_stoi(vocab_d['stoi'], vocab_d['L'])
    stats = Stats.load(run_dir / 'stats.json')
    model = SciLM(cfg, vocab).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, vocab, stats


def build_inference_batch(rows, vocab: Vocab, stats: Stats, cfg: Config,
                          T_override: float | None = None):
    """Build a batch with ionic (slot 7) masked; everything else visible.
    rows: iterable of dicts/Series with keys 'pairs' and 'temperature'.
          (`ionic` is not read here — it's the prediction target.)
    T_override: if given, use this temperature for every row (for T sweeps).
    """
    rows = list(rows)
    B = len(rows)
    P, L = cfg.n_pairs, cfg.L
    tokens = torch.zeros(B, P, L, dtype=torch.long)
    scalars = torch.zeros(B, P, dtype=torch.float32)
    pair_kind = torch.zeros(B, P, dtype=torch.long)
    scalar_mask = torch.zeros(B, P, dtype=torch.bool)
    for i, row in enumerate(rows):
        pairs = row['pairs']
        T = float(T_override) if T_override is not None else float(row['temperature'])
        for j in range(6):
            smi, pct = pairs[j]
            tokens[i, j] = torch.tensor(vocab.encode_smiles_block(smi), dtype=torch.long)
            scalars[i, j] = stats.std_pct(pct)
            pair_kind[i, j] = PAIR_COMPONENT
        tokens[i, 6] = torch.tensor(vocab.encode_marker_block(vocab.TEMP_ID), dtype=torch.long)
        scalars[i, 6] = stats.std_temp(T)
        pair_kind[i, 6] = PAIR_TEMP
        tokens[i, 7] = torch.tensor(vocab.encode_marker_block(vocab.ION_ID), dtype=torch.long)
        # value at slot 7 is ignored: scalar_mask=True replaces it with E_mask_num inside the model
        scalars[i, 7] = 0.0
        pair_kind[i, 7] = PAIR_ION
        scalar_mask[i, 7] = True
    return {'tokens': tokens, 'scalars': scalars,
            'scalar_mask': scalar_mask, 'pair_kind': pair_kind}


@torch.no_grad()
def predict_ionic(model, rows, vocab, stats, cfg, device, batch_size=128,
                  T_override=None) -> np.ndarray:
    """Return predicted ionic (log10 σ in µS/cm) for each row."""
    out = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = build_inference_batch(chunk, vocab, stats, cfg, T_override=T_override)
        batch = {k: v.to(device) for k, v in batch.items()}
        _, scalar_pred = model(batch['tokens'], batch['scalars'],
                               batch['scalar_mask'], batch['pair_kind'])
        # slot 7 is the ionic prediction, in standardized units
        ion_z = scalar_pred[:, 7].detach().cpu().numpy()
        out[start:start + len(chunk)] = stats.unstd_ion(ion_z)
    return out


def short_formulation_label(smiles_field: str) -> str:
    """Compact label like 'LiPF6 + EC@46% + DMC@46%' from the raw smiles field.
    Falls back to truncated SMILES strings if no nicknames are known."""
    # very small map of common solvents/salts to keep the legend readable
    # known salts and solvents; salt SMILES appear in both orderings
    # ([Li+].anion and anion.[Li+]) so we list both.
    nicks = {
        # salts
        '[Li+].[O-]C#N': 'LiOCN',
        '[Li+].[O-][Cl+3]([O-])([O-])[O-]': 'LiClO4',
        '[Li+].F[B-](F)(F)F': 'LiBF4',
        'F[B-](F)(F)F.[Li+]': 'LiBF4',
        '[Li+].F[P-](F)(F)(F)(F)F': 'LiPF6',
        'F[P-](F)(F)(F)(F)F.[Li+]': 'LiPF6',
        '[Li+].O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': 'LiTFSI',
        'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]': 'LiTFSI',
        '[Li+].O=C1O[B-]2(OC(=O)C(=O)O2)OC1=O': 'LiBOB',
        'O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2.[Li+]': 'LiBOB',
        '[Li+].O=C1O[B-]2(OC(=O)C(F)(F)F)(F)OC1=O': 'LiDFOB',
        # solvents
        'CC1COC(=O)O1': 'PC',
        'O=C1OCCO1': 'EC',
        'COC(=O)OC': 'DMC',
        'CCOC(=O)OC': 'EMC',
        'CCOC(=O)OCC': 'DEC',
        'COCCOC': 'DME',
        'CS(C)=O': 'DMSO',
        'O': '',  # padding solvent
    }
    parts = smiles_field.split('<sep>')
    items = []
    for i in range(0, len(parts) - 1, 2):
        smi, pct = parts[i].strip(), parts[i + 1].strip()
        if smi == '' or float(pct) == 0.0:
            continue
        name = nicks.get(smi, smi[:8])
        if name == '':  # filler water/empty
            continue
        items.append(f'{name}@{float(pct):.0f}%')
    return ' + '.join(items) if items else '(empty)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='runs/v2_winner_full')
    ap.add_argument('--checkpoint', default='model.pt',
                    help='checkpoint filename inside run-dir (e.g. model_best.pt)')
    ap.add_argument('--data-root', default=None,
                    help='override data root; default: cfg.data_root from run')
    ap.add_argument('--out', default='fig_v2_test.png')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--formulation-ranks', default='0,3,11',
                    help='comma-separated test-set ranks (by row count) to feature')
    ap.add_argument('--n-sweep', type=int, default=80,
                    help='number of T points in the model curve per formulation')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    print(f'[load] run_dir={run_dir}  checkpoint={args.checkpoint}  device={device}')
    model, cfg, vocab, stats = load_run(run_dir, device, checkpoint=args.checkpoint)
    data_root = args.data_root or cfg.data_root
    df_test = load_split(data_root, 'test')
    print(f'[load] test rows: {len(df_test)}')

    # ----- panel A: full test scatter -----
    rows_all = [df_test.iloc[i] for i in range(len(df_test))]
    print('[predict] running test-set forward pass ...')
    pred_log = predict_ionic(model, rows_all, vocab, stats, cfg, device)
    meas_log = df_test['ionic'].to_numpy(dtype=np.float32)
    T_arr = df_test['temperature'].to_numpy(dtype=np.float32)

    err = pred_log - meas_log
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    median_fold = float(10 ** np.median(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((meas_log - meas_log.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f'[metrics] R²={r2:.3f}  MAE={mae:.4f} log10  RMSE={rmse:.4f} log10  '
          f'median fold-error={median_fold:.2f}x  (paper RMSE: 0.1087)')

    # convert to mS/cm for display
    pred_mS = 10 ** (pred_log - 3.0)
    meas_mS = 10 ** (meas_log - 3.0)

    # ----- panel B: T-sweep curves for chosen formulations -----
    counts = df_test.groupby('smiles').size().sort_values(ascending=False)
    ranks = [int(r) for r in args.formulation_ranks.split(',')]
    chosen_smiles = [counts.index[r] for r in ranks]
    print(f'[curves] formulations (ranks {ranks}):')

    curves = []
    for rank, smi in zip(ranks, chosen_smiles):
        sub = df_test[df_test['smiles'] == smi].copy()
        Ts_meas = sub['temperature'].to_numpy()
        Is_meas = sub['ionic'].to_numpy()
        T_lo, T_hi = Ts_meas.min(), Ts_meas.max()
        # extend slightly past the measured range
        pad = 0.1 * (T_hi - T_lo)
        T_sweep = np.linspace(T_lo - pad, T_hi + pad, args.n_sweep)

        # one fixed example row, T overridden for the sweep
        ref_row = sub.iloc[0]
        sweep_rows = [ref_row] * args.n_sweep
        pred_curve = np.array([
            predict_ionic(model, [ref_row], vocab, stats, cfg, device,
                          T_override=float(t))[0]
            for t in T_sweep
        ])
        label = short_formulation_label(smi)
        print(f'  rank {rank}: {label}  ({len(Ts_meas)} pts, T={T_lo:.1f}..{T_hi:.1f}°C)')
        curves.append({
            'rank': rank, 'label': label,
            'Ts_meas': Ts_meas, 'Is_meas': Is_meas,
            'T_sweep': T_sweep, 'pred_curve': pred_curve,
        })

    # ----- figure -----
    # Compact 2x1 layout for narrow embedding (~2" wide x ~4" tall column).
    # All font sizes are controllable via the FS_* knobs immediately below;
    # tweak these to taste rather than hunting through individual calls.
    FS_TITLE  = 7   # panel titles (set to None to skip titles entirely)
    FS_LABEL  = 7   # x/y axis labels and colorbar label
    FS_TICK   = 6   # x/y tick labels and colorbar tick labels
    FS_LEGEND = 5   # legend in panel B (panel A has no legend)
    FS_METRIC = 6   # the R²/RMSE annotation in panel A

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(2, 3.0),
                                   layout='constrained')

    # panel A: scatter (no legend; y=x is implied by parity-plot convention)
    sc = axA.scatter(meas_mS, pred_mS, c=T_arr, cmap='viridis',
                     s=3, alpha=0.7, edgecolors='none')
    lo = float(min(meas_mS.min(), pred_mS.min())) * 0.5
    hi = float(max(meas_mS.max(), pred_mS.max())) * 2.0
    axA.plot([lo, hi], [lo, hi], 'k--', lw=0.5, alpha=0.6)
    axA.set_xscale('log'); axA.set_yscale('log')
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_xlabel('measured σ (mS/cm)', fontsize=FS_LABEL)
    axA.set_ylabel('predicted σ (mS/cm)', fontsize=FS_LABEL)
    if FS_TITLE is not None:
        axA.set_title(f'Test set (n={len(meas_log)})', fontsize=FS_TITLE)
    axA.grid(True, which='major', alpha=0.3, lw=0.4)
    axA.tick_params(axis='both', which='major', labelsize=FS_TICK,
                    length=2.5, width=0.5, pad=2)
    cbar = fig.colorbar(sc, ax=axA, pad=0.02, shrink=0.85, aspect=15)
    cbar.set_label('T (°C)', fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=1)
    cbar.outline.set_linewidth(0.5)
    txt = (f'R² = {r2:.3f}\n'
           f'RMSE = {rmse:.3f}\n'
           f'fold = {median_fold:.2f}×')
    axA.text(0.97, 0.03, txt, transform=axA.transAxes,
             ha='right', va='bottom', fontsize=FS_METRIC,
             bbox=dict(boxstyle='round,pad=0.2', fc='white',
                       ec='gray', lw=0.4, alpha=0.85))

    # panel B: T-sweep curves with abbreviated legend labels
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        meas_y = 10 ** (c['Is_meas'] - 3.0)
        pred_y = 10 ** (c['pred_curve'] - 3.0)
        # drop the @XX% mole fractions in the label to save horizontal space:
        # 'LiClO4@8% + PC@92%'  ->  'LiClO4 / PC'
        short_label = ' / '.join(part.split('@')[0] for part in c['label'].split(' + '))
        axB.plot(c['T_sweep'], pred_y, color=col, lw=0.9, label=short_label)
        axB.scatter(c['Ts_meas'], meas_y, color=col, s=8,
                    edgecolors='black', linewidths=0.3, zorder=3)
    axB.set_yscale('log')
    axB.set_xlabel('T (°C)', fontsize=FS_LABEL)
    axB.set_ylabel('σ (mS/cm)', fontsize=FS_LABEL)
    if FS_TITLE is not None:
        axB.set_title('σ vs. T', fontsize=FS_TITLE)
    axB.grid(True, which='major', alpha=0.3, lw=0.4)
    axB.tick_params(axis='both', which='major', labelsize=FS_TICK,
                    length=2.5, width=0.5, pad=2)
    axB.legend(loc='lower right', fontsize=FS_LEGEND, frameon=False,
               handlelength=1.2, handletextpad=0.4, borderpad=0.2,
               labelspacing=0.25)

    # thin spines so they don't dominate at this small size
    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_linewidth(0.5)

    out = Path(args.out)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[done] wrote {out}')


if __name__ == '__main__':
    main()
