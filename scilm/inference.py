"""Inference helpers: predict-conductivity and CE-nearest-SMILES decoding.

Usage:
    inf = Inference.from_run('runs/v1_first_full', device='mps')
    inf.predict_conductivity(pairs, T=25.0)
    inf.nearest_smiles(pairs, T, mask_pair=2)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .config import load_config
from .data import (PAIR_COMPONENT, PAIR_ION, PAIR_TEMP, collect_unique_smiles,
                   load_split)
from .data import Stats
from .model import SciLM
from .tokenizer import Vocab, build_vocab_from_stoi


class Inference:
    def __init__(self, model: SciLM, vocab: Vocab, stats: Stats,
                 inventory_smiles: List[str], device: torch.device):
        self.model = model.eval()
        self.vocab = vocab
        self.stats = stats
        self.device = device
        self.inventory = inventory_smiles
        self.inventory_ids = torch.tensor(
            [vocab.encode_smiles_block(s) for s in inventory_smiles], dtype=torch.long
        )

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str | torch.device | None = None):
        run_dir = Path(run_dir)
        cfg = load_config(run_dir / 'config.json')
        vocab_d = json.loads((run_dir / 'vocab.json').read_text())
        vocab = build_vocab_from_stoi(vocab_d['stoi'], vocab_d['L'])
        stats = Stats.load(run_dir / 'stats.json')

        if device is None:
            from .train import pick_device
            device = pick_device()
        device = torch.device(device)

        model = SciLM(cfg, vocab).to(device)
        model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device))

        df_train = load_split(cfg.data_root, 'train')
        df_valid = load_split(cfg.data_root, 'valid')
        df_test  = load_split(cfg.data_root, 'test')
        inventory = collect_unique_smiles(df_train, df_valid, df_test)
        return cls(model, vocab, stats, inventory, device)

    @torch.no_grad()
    def _run_single(self, pairs, T, mask_pairs):
        cfg = self.model.cfg
        v = self.vocab
        s = self.stats
        tokens = torch.zeros(cfg.n_pairs, cfg.L, dtype=torch.long)
        scalars = torch.zeros(cfg.n_pairs, dtype=torch.float32)
        pair_kind = torch.zeros(cfg.n_pairs, dtype=torch.long)
        for j in range(6):
            smi, pct = pairs[j]
            tokens[j] = torch.tensor(v.encode_smiles_block(smi), dtype=torch.long)
            scalars[j] = s.std_pct(pct)
            pair_kind[j] = PAIR_COMPONENT
        tokens[6] = torch.tensor(v.encode_marker_block(v.TEMP_ID), dtype=torch.long)
        scalars[6] = s.std_temp(T if T is not None else 25.0); pair_kind[6] = PAIR_TEMP
        tokens[7] = torch.tensor(v.encode_marker_block(v.ION_ID), dtype=torch.long)
        scalars[7] = 0.0; pair_kind[7] = PAIR_ION

        scalar_mask = torch.zeros(cfg.n_pairs, dtype=torch.bool)
        for p in mask_pairs:
            if p < 6:
                tokens[p] = v.MASK_ID
            scalar_mask[p] = True

        batch = {k: t.unsqueeze(0).to(self.device) for k, t in {
            'tokens': tokens, 'scalars': scalars,
            'scalar_mask': scalar_mask, 'pair_kind': pair_kind,
        }.items()}
        tl, sp = self.model(batch['tokens'], batch['scalars'],
                            batch['scalar_mask'], batch['pair_kind'])
        return tl[0].cpu(), sp[0].cpu()

    def predict_conductivity(self, pairs, T):
        _, sp = self._run_single(pairs, T, mask_pairs={7})
        z = sp[7].item()
        log10_uS = self.stats.unstd_ion(z)
        return {
            'log10_sigma_uS_per_cm': log10_uS,
            'sigma_mS_per_cm': 10 ** (log10_uS - 3),
        }

    def nearest_smiles_from_logits(self, logits_block: torch.Tensor) -> Tuple[str, float]:
        v = self.vocab
        logp = F.log_softmax(logits_block, dim=-1)
        ids = self.inventory_ids
        N, L = ids.shape
        gathered = logp.gather(-1, ids.T.unsqueeze(-1).reshape(L, N)).squeeze(-1)
        nonpad = (ids != v.PAD_ID).float().T
        nll = -(gathered * nonpad).sum(0) / nonpad.sum(0).clamp(min=1)
        best = int(torch.argmin(nll).item())
        return self.inventory[best], float(nll[best])

    def nearest_smiles(self, pairs, T, mask_pair: int) -> Tuple[str, float]:
        """Mask one component pair and decode it from the inventory."""
        assert 0 <= mask_pair < 6
        tl, _ = self._run_single(pairs, T, mask_pairs={mask_pair})
        return self.nearest_smiles_from_logits(tl[mask_pair])
