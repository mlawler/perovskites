"""Train a single SciLM model. Saves config, vocab, stats, history, weights, and a loss-curves PNG."""
from __future__ import annotations
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, save_config
from .data import (FormulationDataset, collate, collect_unique_smiles,
                   compute_stats, load_split)
from .model import SciLM
from .tokenizer import build_vocab


def pick_device() -> torch.device:
    """cuda > cpu. MPS is intentionally NOT auto-selected: in this project a
    sanity run on MPS produced val_mse ~1e33 (numerical blow-up) while the same
    code on CPU converged normally. Pass device='mps' explicitly only after
    testing that specific model + torch version."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def scilm_loss(tok_logits, scalar_pred, batch, lam: float):
    B, P, L, V_ = tok_logits.shape
    tok_mask  = batch['token_mask']
    tgt_tok   = batch['target_tokens']
    scal_mask = batch['scalar_mask']
    tgt_scal  = batch['target_scalars']

    if tok_mask.any():
        logits_flat = tok_logits.reshape(-1, V_)
        tgt_flat    = tgt_tok.reshape(-1)
        m           = tok_mask.reshape(-1)
        ce = F.cross_entropy(logits_flat[m], tgt_flat[m])
    else:
        ce = tok_logits.new_zeros(())

    if scal_mask.any():
        mse = F.mse_loss(scalar_pred[scal_mask], tgt_scal[scal_mask])
    else:
        mse = scalar_pred.new_zeros(())

    return ce + lam * mse, ce.detach(), mse.detach()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _save_loss_curves(history: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    steps = history['step']
    axes[0].plot(steps, history['train_total'], label='train total', alpha=0.6)
    if history['val_epoch']:
        axes[0].plot(history['val_epoch'], history['val_total'], 'o-', label='val total')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('total loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(steps, history['train_ce'],  label='train CE',  alpha=0.6)
    if history['val_epoch']:
        axes[1].plot(history['val_epoch'], history['val_ce'],  'o-', label='val CE')
    axes[1].set_xlabel('step'); axes[1].set_ylabel('token CE'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(steps, history['train_mse'], label='train MSE', alpha=0.6)
    if history['val_epoch']:
        axes[2].plot(history['val_epoch'], history['val_mse'], 'o-', label='val MSE')
    axes[2].set_xlabel('step'); axes[2].set_ylabel('scalar MSE (std)'); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_one(
    cfg: Config,
    save_dir: str | Path,
    log_cb: Optional[Callable[[dict], None]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Train SciLM with the given config; write artifacts to save_dir; return history dict.

    log_cb(payload) is called after every epoch with {'epoch', 'history', 'cfg'}.
    Use it for live plots in a notebook (e.g. clear_output + plt.show).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = device or pick_device()
    print(f'[train] device={device}  save_dir={save_dir}')

    df_train = load_split(cfg.data_root, 'train')
    df_valid = load_split(cfg.data_root, 'valid')
    df_test  = load_split(cfg.data_root, 'test')

    unique_smiles = collect_unique_smiles(df_train, df_valid, df_test)
    vocab = build_vocab(unique_smiles, cfg.L)
    stats = compute_stats(df_train)

    save_config(cfg, save_dir / 'config.json')
    (save_dir / 'vocab.json').write_text(json.dumps(vocab.to_dict(), indent=2))
    stats.save(save_dir / 'stats.json')

    ds_train = FormulationDataset(df_train, cfg, vocab, stats, augment=True,  rng_seed=None)
    ds_valid = FormulationDataset(df_valid, cfg, vocab, stats, augment=False, rng_seed=123)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0, drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0, drop_last=False)

    model = SciLM(cfg, vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.n_epochs * len(dl_train)
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {'step': [], 'train_total': [], 'train_ce': [], 'train_mse': [],
               'val_epoch': [], 'val_total': [], 'val_ce': [], 'val_mse': [],
               'epoch_seconds': [], 'samples_per_sec': [],
               'best_val_total': float('inf'), 'best_epoch': 0,
               'n_params': n_params, 'device': str(device)}

    @torch.no_grad()
    def run_val():
        model.eval()
        tot = ce_tot = mse_tot = n = 0
        for batch in dl_valid:
            batch = to_device(batch, device)
            tl, sp = model(batch['tokens'], batch['scalars'],
                           batch['scalar_mask'], batch['pair_kind'])
            loss, ce, mse = scilm_loss(tl, sp, batch, cfg.lambda_scalar)
            bs = batch['tokens'].size(0)
            tot += loss.item() * bs; ce_tot += ce.item() * bs; mse_tot += mse.item() * bs; n += bs
        model.train()
        return tot / n, ce_tot / n, mse_tot / n

    model.train()
    global_step = 0
    for epoch in range(cfg.n_epochs):
        epoch_t0 = time.perf_counter()
        epoch_samples = 0
        for batch in dl_train:
            batch = to_device(batch, device)
            tl, sp = model(batch['tokens'], batch['scalars'],
                           batch['scalar_mask'], batch['pair_kind'])
            loss, ce, mse = scilm_loss(tl, sp, batch, cfg.lambda_scalar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            history['step'].append(global_step)
            history['train_total'].append(loss.item())
            history['train_ce'].append(ce.item())
            history['train_mse'].append(mse.item())
            epoch_samples += batch['tokens'].size(0)
            global_step += 1
        epoch_seconds = time.perf_counter() - epoch_t0
        sps = epoch_samples / epoch_seconds if epoch_seconds > 0 else 0.0
        history['epoch_seconds'].append(epoch_seconds)
        history['samples_per_sec'].append(sps)

        vt, vce, vmse = run_val()
        history['val_epoch'].append(global_step)
        history['val_total'].append(vt); history['val_ce'].append(vce); history['val_mse'].append(vmse)
        improved = vt < history['best_val_total']
        if improved:
            history['best_val_total'] = vt
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), save_dir / 'model_best.pt')
            # also stash an epoch-tagged snapshot so we can replay the
            # trajectory of improvement (e.g. how per-formulation predictions
            # evolve across the run).
            torch.save(model.state_dict(),
                       ckpt_dir / f'best_ep{epoch+1:03d}.pt')
        marker = '  *best*' if improved else ''
        print(f'  epoch {epoch+1}/{cfg.n_epochs}  '
              f'{epoch_seconds:.1f}s  {sps:.1f} samples/s  '
              f'train_total={history["train_total"][-1]:.4f}  '
              f'val_total={vt:.4f}{marker}')
        if log_cb is not None:
            log_cb({'epoch': epoch + 1, 'history': history, 'cfg': cfg})

    torch.save(model.state_dict(), save_dir / 'model.pt')
    (save_dir / 'history.json').write_text(json.dumps(history, indent=2))
    _save_loss_curves(history, save_dir / 'loss_curves.png')

    return history
