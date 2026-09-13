"""Data loading, scalar standardization, and the FormulationDataset."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config
from .tokenizer import Vocab


PAIR_COMPONENT = 0
PAIR_TEMP = 1
PAIR_ION = 2


def parse_row(smiles_field: str) -> List[Tuple[str, float]]:
    """Parse '<sep>'-delimited alternating (smiles, pct, ...). Returns 6 (smi, pct) pairs."""
    parts = smiles_field.split('<sep>')
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        smi = parts[i].strip()
        pct = parts[i + 1].strip()
        if smi == '':
            continue
        pairs.append((smi, float(pct)))
    assert len(pairs) == 6, f'expected 6 pairs, got {len(pairs)}: {smiles_field[:80]}'
    return pairs


def load_split(data_root: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_root) / f'{name}.csv')
    df['pairs'] = df['smiles'].apply(parse_row)
    return df


@dataclass
class Stats:
    pct_mean: float; pct_std: float
    temp_mean: float; temp_std: float
    ion_mean: float; ion_std: float

    def std_pct(self, x):  return (x - self.pct_mean)  / self.pct_std
    def std_temp(self, x): return (x - self.temp_mean) / self.temp_std
    def std_ion(self, x):  return (x - self.ion_mean)  / self.ion_std
    def unstd_pct(self, z):  return z * self.pct_std  + self.pct_mean
    def unstd_temp(self, z): return z * self.temp_std + self.temp_mean
    def unstd_ion(self, z):  return z * self.ion_std  + self.ion_mean

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def compute_stats(df_train: pd.DataFrame) -> Stats:
    all_pcts = np.array([p for pairs in df_train['pairs'] for _, p in pairs], dtype=np.float32)
    T_arr = df_train['temperature'].to_numpy(dtype=np.float32)
    I_arr = df_train['ionic'].to_numpy(dtype=np.float32)
    return Stats(
        pct_mean=float(all_pcts.mean()),  pct_std=float(all_pcts.std()),
        temp_mean=float(T_arr.mean()),    temp_std=float(T_arr.std()),
        ion_mean=float(I_arr.mean()),     ion_std=float(I_arr.std()),
    )


def collect_unique_smiles(*dfs: pd.DataFrame) -> List[str]:
    s = set()
    for df in dfs:
        for pairs in df['pairs']:
            for smi, _ in pairs:
                s.add(smi)
    return sorted(s)


class FormulationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, vocab: Vocab, stats: Stats,
                 augment: bool = True, rng_seed: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.vocab = vocab
        self.stats = stats
        self.augment = augment
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cfg = self.cfg
        v = self.vocab
        s = self.stats
        row = self.df.iloc[idx]
        pairs = list(row['pairs'])
        T = float(row['temperature'])
        I = float(row['ionic'])

        if self.augment and cfg.permute_solvent_slots:
            solvents = pairs[1:]
            self._rng.shuffle(solvents)
            pairs = [pairs[0]] + solvents

        tokens = torch.zeros(cfg.n_pairs, cfg.L, dtype=torch.long)
        scalars = torch.zeros(cfg.n_pairs, dtype=torch.float32)
        pair_kind = torch.zeros(cfg.n_pairs, dtype=torch.long)

        for j in range(6):
            smi, pct = pairs[j]
            tokens[j] = torch.tensor(v.encode_smiles_block(smi), dtype=torch.long)
            scalars[j] = s.std_pct(pct)
            pair_kind[j] = PAIR_COMPONENT
        tokens[6] = torch.tensor(v.encode_marker_block(v.TEMP_ID), dtype=torch.long)
        scalars[6] = s.std_temp(T); pair_kind[6] = PAIR_TEMP
        tokens[7] = torch.tensor(v.encode_marker_block(v.ION_ID), dtype=torch.long)
        scalars[7] = s.std_ion(I);  pair_kind[7] = PAIR_ION

        target_tokens = tokens.clone()
        target_scalars = scalars.clone()

        k = self._rng.randint(cfg.min_pairs_masked, cfg.max_pairs_masked)
        chosen = self._rng.sample(range(cfg.n_pairs), k)
        token_mask = torch.zeros(cfg.n_pairs, cfg.L, dtype=torch.bool)
        scalar_mask = torch.zeros(cfg.n_pairs, dtype=torch.bool)
        for p in chosen:
            if self._rng.random() < 0.5:
                tokens[p] = v.MASK_ID
                token_mask[p] = True
                scalar_mask[p] = True
            else:
                scalar_mask[p] = True
        return {
            'tokens': tokens,
            'scalars': scalars,
            'scalar_mask': scalar_mask,
            'token_mask': token_mask,
            'target_tokens': target_tokens,
            'target_scalars': target_scalars,
            'pair_kind': pair_kind,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}
