"""Byte-level tokenization for Sparrow-1M.

Each byte 0x00-0xFF maps to token ID 0-255 directly. No BPE, no merges,
no learned vocabulary. Sparrow's vocab_size is exactly 256.

This module is shared by train_local.py and eval_vs_1b.py.
"""
import torch


PAD_ID = 0  # byte 0x00 (NUL)
BOS_ID = 0  # same as PAD
EOS_ID = 10  # byte 0x0a (newline) — natural end-of-example for our line-per-problem format


def encode(text: str) -> list:
    """Return a list of int token IDs (0-255) for the input string."""
    return list(text.encode('utf-8'))


def encode_batch(texts: list) -> list:
    return [encode(t) for t in texts]


def decode(ids) -> str:
    """Decode a list/tensor of int token IDs back to a string."""
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    # Filter out any ids >=256 (shouldn't happen, but defensive)
    safe = [i for i in ids if 0 <= i < 256]
    return bytes(safe).decode('utf-8', errors='replace')


def encode_tensor(text: str) -> torch.Tensor:
    """Convenience: encode + wrap as 1D long tensor."""
    return torch.tensor(encode(text), dtype=torch.long)
