"""Local training loop for Sparrow-1M.

CPU-or-GPU friendly (auto-detects CUDA). No Liger Kernel, no Muon —
vanilla AdamW + cosine schedule. The model is small enough that
the simplicity is the point.

Reads a line-delimited text file produced by gen_arith.py. Concatenates
lines with a newline (already present), tokenizes via raw bytes, and
slices into seq_len chunks for next-token prediction.

Usage:
    python train_local.py \\
        --resume   E:/sparrow/iter1/init \\
        --output   E:/sparrow/iter1/trained \\
        --data     E:/sparrow/iter1_2digit_add.txt \\
        --steps    5000 --batch-size 64 --seq-len 128 \\
        --peak-lr  3e-3 --min-lr 3e-4
"""
import argparse
import math
import os
import random
import sys
import time
from typing import Iterator

import torch
import torch.nn.functional as F

# Local import (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes_tok import encode, EOS_ID, PAD_ID  # noqa: E402


# ----- Data loader ---------------------------------------------------------

def load_corpus_bytes(path: str) -> list:
    """Read the line-delimited text file as one big list of byte token IDs.
    Lines already end in \\n, which is our natural EOS."""
    print(f'  reading {path} ...')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    ids = list(text.encode('utf-8'))
    print(f'    {len(ids):,} tokens ({len(ids)/1e6:.1f} M bytes)')
    return ids


def packed_batch_iter(ids: list, batch_size: int, seq_len: int,
                     rng: random.Random) -> Iterator:
    """Yield (input_ids, labels) tensors of shape [batch_size, seq_len].

    CRITICAL: labels = input_ids (NOT pre-shifted). HuggingFace's
    Qwen3ForCausalLM.forward(labels=...) does its own shift internally
    (shift_labels = labels[..., 1:]). Passing pre-shifted labels here
    causes an off-by-one (model trained to predict 2 positions ahead),
    which silently produces low loss but a model that emits the wrong
    token at inference.
    """
    n = len(ids)
    if n < seq_len + 1:
        raise RuntimeError(f'corpus too small: {n} tokens, need at least {seq_len + 1}')
    while True:
        starts = [rng.randint(0, n - seq_len - 1) for _ in range(batch_size)]
        inp = torch.tensor([ids[s:s + seq_len] for s in starts], dtype=torch.long)
        tgt = inp.clone()  # HF does its own shift; passing same as input is correct
        yield inp, tgt


# ----- Optimizer schedule --------------------------------------------------

def cosine_lr(step, total, peak, mn, warmup_steps=200):
    if step < warmup_steps:
        return mn + (peak - mn) * (step / max(warmup_steps, 1))
    p = (step - warmup_steps) / max(total - warmup_steps, 1)
    p = min(max(p, 0.0), 1.0)
    return mn + (peak - mn) * 0.5 * (1 + math.cos(math.pi * p))


# ----- Train ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--resume', required=True, help='Sparrow init or prior checkpoint dir')
    p.add_argument('--output', required=True)
    p.add_argument('--data', required=True, help='line-delimited text file from gen_arith.py')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--peak-lr', type=float, default=3e-3)
    p.add_argument('--min-lr', type=float, default=3e-4)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260504)
    p.add_argument('--device', default=None,
                   help='cuda | cpu (default: auto)')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'  device: {device}')

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # ----- Load model
    from transformers import Qwen3ForCausalLM
    print(f'  loading from {args.resume}')
    model = Qwen3ForCausalLM.from_pretrained(args.resume, torch_dtype=torch.float32)
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Sparrow-1M loaded: {n_params/1e6:.3f}M params')

    # ----- Load data
    ids = load_corpus_bytes(args.data)
    bi = packed_batch_iter(ids, args.batch_size, args.seq_len, rng)

    # ----- Optimizer (vanilla AdamW)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # ----- Train loop
    eff_batch = args.batch_size  # no grad accum for tiny models
    print(f'\n=== Sparrow-1M training begins ===')
    print(f'    steps={args.steps}  batch={args.batch_size}  seq_len={args.seq_len}')
    print(f'    LR: warmup {args.warmup_steps} steps to {args.peak_lr}, '
          f'cosine decay to {args.min_lr}')
    print(f'    Effective tokens/step: {eff_batch * args.seq_len:,}')
    print(f'    Total tokens: {args.steps * eff_batch * args.seq_len:,}\n')

    log_path = os.path.join(args.output, 'train.log')
    log_f = open(log_path, 'a', buffering=1)

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.peak_lr, args.min_lr, args.warmup_steps)
        for pg in optim.param_groups:
            pg['lr'] = lr

        inp, tgt = next(bi)
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(input_ids=inp, labels=tgt)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f'  step {step}: NaN/Inf -- abort')
            return

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        losses.append(loss.item())
        if len(losses) > 100:
            losses.pop(0)

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses) / len(losses)
            el = time.time() - t_start
            tps = (step * eff_batch * args.seq_len) / el
            line = (f'  step {step:6d}/{args.steps}  loss={loss.item():.4f}  '
                    f'avg100={avg:.4f}  lr={lr:.2e}  gn={gn.item():.3f}  '
                    f'{tps:,.0f} tok/s  elapsed={el/60:.1f}m')
            print(line)
            log_f.write(line + '\n')

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt = os.path.join(args.output, f'step_{step:06d}')
            model.save_pretrained(ckpt, safe_serialization=True)
            print(f'    [ckpt saved: {ckpt}]')

    final = os.path.join(args.output, 'final')
    model.save_pretrained(final, safe_serialization=True)
    # Also copy sparrow_tokenizer.json from the resume dir if present
    tok_meta_src = os.path.join(args.resume, 'sparrow_tokenizer.json')
    if os.path.exists(tok_meta_src):
        import shutil
        shutil.copy(tok_meta_src, os.path.join(final, 'sparrow_tokenizer.json'))
    print(f'\n=== Sparrow-1M training done ===')
    print(f'    final: {final}')
    print(f'    train log: {log_path}')

    log_f.close()


if __name__ == '__main__':
    main()
