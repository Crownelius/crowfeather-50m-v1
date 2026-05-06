"""Phase E Phase 1 trainer — Position-Coupled Sparrow.

Differences from `train_local.py`:
  1. Batches are PER-PROBLEM (line-aligned), not packed-stream chunks. Each
     problem in the corpus produces its own row in the batch (padded to the
     batch's max length).
  2. Per-sample random offset: sample `start ~ U[1, max_pos - len(line)]`
     for each problem in each batch, then compute coupled position_ids via
     `gen_arith.compute_pc_position_ids(line, start)`.
  3. Pass `position_ids` to `Qwen3ForCausalLM.forward(...)`. The HF API
     supports this natively — no model subclassing.
  4. Loss masking: only compute CE loss on the bytes AT or AFTER the ' = '
     boundary. Operand + operator + first-space bytes are -100. The deep-read
     extract from arxiv:2405.20671 says: "We only care about the result of
     the next-token prediction for the '=' token and the tokens in the
     response (except its last token)." We follow that exactly.

Usage:
    python train_local_pc.py \\
        --resume   E:/sparrow/iter37/init \\
        --output   E:/sparrow/iter37/trained \\
        --data     E:/sparrow/iter37/data.txt \\
        --steps    25000 --batch-size 32 --max-pos 200 \\
        --peak-lr  1.5e-3 --min-lr 1e-5 --warmup-steps 500
"""
import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes_tok import encode, EOS_ID  # noqa: E402
from gen_arith import compute_pc_position_ids  # noqa: E402


def load_lines(path: str) -> list:
    """Read training corpus as a list of strings, one problem per line."""
    print(f'  reading {path} ...')
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n').rstrip('\r') for ln in f if ln.strip()]
    print(f'    {len(lines):,} problems')
    return lines


def find_eq_byte_index(line: str) -> int:
    """Return the byte index of the '=' character in the line."""
    return line.index('=')


def build_batch(lines: list, indices: list, max_pos: int, max_seq_len: int,
                rng: random.Random, device: str):
    """Build (input_ids, position_ids, labels, attention_mask) tensors for
    a batch of problem indices. Each row:
      - input_ids: bytes of the line, padded with PAD (byte 0) to batch max length
      - position_ids: coupled IDs from compute_pc_position_ids(line, start)
                      with start sampled per-line; PAD positions get 0
      - labels: same as input_ids, but bytes BEFORE '=' are masked to -100
      - attention_mask: 1 for real bytes, 0 for PAD
    """
    batch_lines = [lines[i] for i in indices]
    # Truncate any line longer than max_seq_len (shouldn't happen at 3-digit mul)
    batch_lines = [ln[:max_seq_len] for ln in batch_lines]

    # Sample per-line random offset, ensuring start + len(line) <= max_pos.
    # If line is too long for any start>=1, clamp start = 1.
    pos_ids_per_line = []
    for ln in batch_lines:
        max_start = max_pos - len(ln)
        if max_start < 1:
            start = 1
        else:
            start = rng.randint(1, max_start)
        try:
            pos_ids = compute_pc_position_ids(ln, start=start)
        except (ValueError, AssertionError):
            # Malformed line — fall back to sequential IDs starting at 1
            pos_ids = list(range(1, 1 + len(ln)))
        pos_ids_per_line.append(pos_ids)

    # Encode bytes
    input_lists = [encode(ln) for ln in batch_lines]

    # Pad to batch max length
    seq_len = max(len(x) for x in input_lists)
    B = len(batch_lines)

    input_ids = torch.zeros(B, seq_len, dtype=torch.long)
    position_ids = torch.zeros(B, seq_len, dtype=torch.long)
    labels = torch.full((B, seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(B, seq_len, dtype=torch.long)

    for r, (ids, pos_ids, ln) in enumerate(zip(input_lists, pos_ids_per_line, batch_lines)):
        L = len(ids)
        input_ids[r, :L] = torch.tensor(ids, dtype=torch.long)
        position_ids[r, :L] = torch.tensor(pos_ids, dtype=torch.long)
        attention_mask[r, :L] = 1
        # Loss mask: only AT-or-AFTER the '=' byte.
        try:
            eq_byte_idx = find_eq_byte_index(ln)
        except ValueError:
            continue
        # Predict tokens at eq_byte_idx onward (HF does its own shift, so we
        # set labels = input_ids for those positions). Bytes before eq_byte_idx
        # stay at -100 (ignored by CE).
        labels[r, eq_byte_idx:L] = input_ids[r, eq_byte_idx:L]

    return (input_ids.to(device),
            position_ids.to(device),
            labels.to(device),
            attention_mask.to(device))


def cosine_lr(step, total, peak, mn, warmup_steps=500):
    if step < warmup_steps:
        return mn + (peak - mn) * (step / max(warmup_steps, 1))
    p = (step - warmup_steps) / max(total - warmup_steps, 1)
    p = min(max(p, 0.0), 1.0)
    return mn + (peak - mn) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--resume', required=True, help='Sparrow init or prior ckpt dir')
    p.add_argument('--output', required=True)
    p.add_argument('--data', required=True, help='line-delimited PC text from gen_arith.py --pc')
    p.add_argument('--steps', type=int, default=25000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-seq-len', type=int, default=128,
                   help='per-row hard cap on bytes; longer lines truncated')
    p.add_argument('--max-pos', type=int, default=200,
                   help='max position ID; samples start ~ U[1, max_pos - line_len]')
    p.add_argument('--peak-lr', type=float, default=1.5e-3)
    p.add_argument('--min-lr',  type=float, default=1e-5)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--ckpt-every', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20260506)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device: {device}')

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    from transformers import Qwen3ForCausalLM
    print(f'  loading from {args.resume}')
    model = Qwen3ForCausalLM.from_pretrained(args.resume, torch_dtype=torch.float32)
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Sparrow-PC loaded: {n_params/1e6:.3f}M params')

    lines = load_lines(args.data)
    if len(lines) < args.batch_size:
        raise RuntimeError(f'data has only {len(lines)} lines; need >= batch_size {args.batch_size}')

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    print(f'\n=== Sparrow-PC training begins ===')
    print(f'    steps={args.steps}  batch={args.batch_size}  max_pos={args.max_pos}')
    print(f'    LR: warmup {args.warmup_steps} steps to {args.peak_lr}, '
          f'cosine decay to {args.min_lr}\n')

    log_path = os.path.join(args.output, 'train.log')
    log_f = open(log_path, 'a', buffering=1)

    losses = []
    t_start = time.time()

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.peak_lr, args.min_lr, args.warmup_steps)
        for pg in optim.param_groups:
            pg['lr'] = lr

        # Sample batch indices
        idxs = [rng.randint(0, len(lines) - 1) for _ in range(args.batch_size)]
        input_ids, position_ids, labels, attention_mask = build_batch(
            lines, idxs, args.max_pos, args.max_seq_len, rng, device
        )

        out = model(input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=labels)
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
            tps = step * args.batch_size * input_ids.size(1) / el
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
    tok_meta_src = os.path.join(args.resume, 'sparrow_tokenizer.json')
    if os.path.exists(tok_meta_src):
        import shutil
        shutil.copy(tok_meta_src, os.path.join(final, 'sparrow_tokenizer.json'))
    print(f'\n=== Sparrow-PC training done ===')
    print(f'    final: {final}')
    print(f'    train log: {log_path}')

    log_f.close()


if __name__ == '__main__':
    main()
