"""Config dataclass and run-mode factories.

CFG = make_config('full')
CFG = make_config('sanity')
CFG = make_config('full', lr=1e-3, dropout=0.2)   # overrides
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # tokenization / grid
    # L = max atom-token length in the dataset (40) + 1 for <eos_smiles>.
    # Longest SMILES has 40 atom tokens, so L must be >= 41.
    L: int = 41
    n_pairs: int = 8       # 6 components + temperature + conductivity
    # model
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    # loss
    lambda_scalar: float = 1.0
    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    n_epochs: int = 40
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    # masking
    min_pairs_masked: int = 1
    max_pairs_masked: int = 3
    permute_solvent_slots: bool = True
    # data
    data_root: str = 'data/IBM_SMI_TED_IC/data'
    # bookkeeping
    seed: int = 0


SANITY_OVERRIDES = dict(d_model=64, n_layers=2, n_heads=2, d_ff=128,
                        batch_size=32, n_epochs=2)


def make_config(run_mode: str = 'full', **overrides) -> Config:
    if run_mode == 'sanity':
        base = {**SANITY_OVERRIDES}
    elif run_mode == 'full':
        base = {}
    else:
        raise ValueError(f"Unknown run_mode={run_mode!r}; use 'sanity' or 'full'.")
    base.update(overrides)
    return Config(**base)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))


def load_config(path: str | Path) -> Config:
    return Config(**json.loads(Path(path).read_text()))
