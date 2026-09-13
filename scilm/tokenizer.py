"""Atom-level SMILES tokenizer + vocabulary.

Regex matches SMI-TED's. Vocab is built from atom tokens present in the dataset
plus structural specials.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

SPECIALS = ['<pad>', '<eos_smiles>', '<temp>', '<ionic>', '<mask_tok>']


def smiles_tokens(smi: str) -> List[str]:
    toks = SMILES_REGEX.findall(smi)
    assert ''.join(toks) == smi, f'tokenizer did not round-trip: {smi!r} -> {toks!r}'
    return toks


@dataclass
class Vocab:
    stoi: dict
    itos: dict
    V: int
    L: int
    PAD_ID: int
    EOS_ID: int
    TEMP_ID: int
    ION_ID: int
    MASK_ID: int

    def encode_smiles_block(self, smi: str) -> List[int]:
        """L token ids: [atom tokens..., <eos_smiles>, <pad> ...]."""
        toks = smiles_tokens(smi)
        assert len(toks) <= self.L - 1, f'SMILES too long ({len(toks)} > {self.L - 1}): {smi}'
        ids = [self.stoi[t] for t in toks]
        ids.append(self.EOS_ID)
        while len(ids) < self.L:
            ids.append(self.PAD_ID)
        return ids

    def encode_marker_block(self, marker_id: int) -> List[int]:
        return [marker_id] + [self.PAD_ID] * (self.L - 1)

    def to_dict(self):
        return {'stoi': self.stoi, 'L': self.L}

    @classmethod
    def from_dict(cls, d):
        return build_vocab_from_stoi(d['stoi'], d['L'])


def build_vocab(unique_smiles: Iterable[str], L: int) -> Vocab:
    """Build vocab from the union of atom tokens in the given SMILES set."""
    atom_vocab = set()
    max_len = 0
    for smi in unique_smiles:
        toks = smiles_tokens(smi)
        atom_vocab.update(toks)
        max_len = max(max_len, len(toks))
    assert L >= max_len + 1, (
        f'L={L} too small: need at least {max_len + 1} '
        f'(longest SMILES has {max_len} atom tokens, plus 1 for <eos_smiles>)'
    )
    vocab_list = SPECIALS + sorted(atom_vocab)
    stoi = {t: i for i, t in enumerate(vocab_list)}
    return build_vocab_from_stoi(stoi, L)


def build_vocab_from_stoi(stoi: dict, L: int) -> Vocab:
    itos = {i: t for t, i in stoi.items()}
    return Vocab(
        stoi=stoi, itos=itos, V=len(stoi), L=L,
        PAD_ID=stoi['<pad>'], EOS_ID=stoi['<eos_smiles>'],
        TEMP_ID=stoi['<temp>'], ION_ID=stoi['<ionic>'],
        MASK_ID=stoi['<mask_tok>'],
    )
