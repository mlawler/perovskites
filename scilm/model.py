"""SciLM transformer model and ScalarEncoder."""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import Config
from .tokenizer import Vocab


class ScalarEncoder(nn.Module):
    """Per-kind MLP that embeds (scalar, kind) into a d_model vector."""
    def __init__(self, d_model: int, n_kinds: int = 3, hidden: int = 64):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d_model))
            for _ in range(n_kinds)
        ])

    def forward(self, x: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        B, P = x.shape
        D = self.mlps[0][-1].out_features
        out = x.new_zeros(B, P, D)
        x_unsq = x.unsqueeze(-1)
        for k, mlp in enumerate(self.mlps):
            m = (kind == k)
            if m.any():
                out[m] = mlp(x_unsq[m])
        return out


class SciLM(nn.Module):
    def __init__(self, cfg: Config, vocab: Vocab):
        super().__init__()
        self.cfg = cfg
        self.PAD_ID = vocab.PAD_ID
        V = vocab.V

        self.E_tok  = nn.Embedding(V, cfg.d_model)
        self.E_slot = nn.Embedding(cfg.n_pairs, cfg.d_model)
        self.E_pos  = nn.Embedding(cfg.L, cfg.d_model)
        nn.init.normal_(self.E_tok.weight,  mean=0.0, std=0.02)
        nn.init.normal_(self.E_slot.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.E_pos.weight,  mean=0.0, std=0.02)
        self.E_mask_num = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.E_mask_num, std=0.02)
        self.scalar_enc = ScalarEncoder(cfg.d_model, n_kinds=3)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is incompatible
        # with norm_first=True, so leaving it on just emits a warning each init.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(cfg.d_model)

        self.W_tok_bias = nn.Parameter(torch.zeros(V))
        self.W_num = nn.ModuleList([nn.Linear(cfg.d_model, 1) for _ in range(3)])

        slot_idx = torch.arange(cfg.n_pairs).unsqueeze(1).expand(cfg.n_pairs, cfg.L)
        pos_idx  = torch.arange(cfg.L).unsqueeze(0).expand(cfg.n_pairs, cfg.L)
        self.register_buffer('slot_idx', slot_idx, persistent=False)
        self.register_buffer('pos_idx',  pos_idx,  persistent=False)

    def forward(self, tokens, scalars, scalar_mask, pair_kind):
        """tokens: (B, P, L)  scalars: (B, P)  scalar_mask: (B, P) bool  pair_kind: (B, P) long"""
        B, P, L = tokens.shape
        D = self.cfg.d_model

        tok_e  = self.E_tok(tokens)
        slot_e = self.E_slot(self.slot_idx).unsqueeze(0).expand(B, P, L, D)
        pos_e  = self.E_pos(self.pos_idx).unsqueeze(0).expand(B, P, L, D)

        num_e_pair = self.scalar_enc(scalars, pair_kind)
        mask_vec = self.E_mask_num.view(1, 1, D).expand(B, P, D)
        num_e_pair = torch.where(scalar_mask.unsqueeze(-1), mask_vec, num_e_pair)
        num_e = num_e_pair.unsqueeze(2).expand(B, P, L, D)

        x = tok_e + slot_e + pos_e + num_e
        x = x.reshape(B, P * L, D)

        key_padding_mask = (tokens.reshape(B, P * L) == self.PAD_ID)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm_out(h)
        h = h.reshape(B, P, L, D)

        tok_logits = torch.einsum('bpld,vd->bplv', h, self.E_tok.weight) + self.W_tok_bias

        anchor = h[:, :, 0, :]
        scalar_pred = torch.zeros(B, P, device=h.device, dtype=h.dtype)
        for k, head in enumerate(self.W_num):
            m = (pair_kind == k)
            if m.any():
                scalar_pred[m] = head(anchor[m]).squeeze(-1)
        return tok_logits, scalar_pred
