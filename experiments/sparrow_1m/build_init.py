"""Build a fresh 1M Qwen3 dense model for Sparrow.

Architecture (from README.md):
    vocab_size       = 256   (raw bytes; no BPE training needed)
    hidden_size      = 128
    num_hidden_layers = 5
    num_attention_heads = 4 (Q)
    num_key_value_heads = 2 (KV, GQA 2:1)
    head_dim         = 32
    intermediate_size = 512 (4x hidden, SwiGLU)
    max_position_embeddings = 512
    tied embeddings  = True

Total ~1.078M parameters.

Tokenization: bytes 0-255 are token IDs 0-255 directly. No tokenizer.json
is saved — Sparrow's encode/decode lives in scripts/bytes_tok.py and is
imported by train_local.py and eval_vs_1b.py.
"""
import argparse
import json
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--vocab-size', type=int, default=256)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--kv-heads', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=32)
    p.add_argument('--intermediate', type=int, default=512)
    p.add_argument('--max-pos', type=int, default=512)
    p.add_argument('--rope-theta', type=float, default=10_000.0)
    p.add_argument('--seed', type=int, default=20260504)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate,
        max_position_embeddings=args.max_pos,
        rope_theta=args.rope_theta,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_cache=True,  # eval uses model.generate, which wants the KV cache
        bos_token_id=0,  # we use byte 0 as BOS / PAD interchangeably
        eos_token_id=10, # newline byte ('\n') as natural end-of-example
        pad_token_id=0,
        torch_dtype='float32',  # train in fp32 (model is tiny, no need for bf16)
    )

    model = Qwen3ForCausalLM(cfg)

    n_total = sum(p.numel() for p in model.parameters())
    n_embed = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n.lower())
    n_layers_p = n_total - n_embed
    print(f'  Sparrow-1M init')
    print(f'    Total params:   {n_total/1e6:.3f} M  ({n_total:,})')
    print(f'    Embedding:      {n_embed/1e6:.3f} M  ({100*n_embed/n_total:.1f}%)')
    print(f'    Transformer:    {n_layers_p/1e6:.3f} M  ({100*n_layers_p/n_total:.1f}%)')
    print(f'    Layers:         {args.layers}')
    print(f'    Hidden:         {args.hidden}')
    print(f'    Heads (Q/KV):   {args.heads}/{args.kv_heads}  head_dim={args.head_dim}')
    print(f'    FFN intermed:   {args.intermediate}')
    print(f'    Max position:   {args.max_pos}')

    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save a small sparrow_tokenizer.json that documents the byte mapping. This
    # is purely informational — the trainer/eval scripts use direct byte encoding.
    tok_meta = {
        'tokenizer_type': 'bytes',
        'vocab_size': args.vocab_size,
        'note': 'Each byte 0x00-0xFF maps to token ID 0-255 directly. '
                'Use bytes_tok.encode(s) / bytes_tok.decode(ids).',
        'special_tokens': {
            'pad_token_id': 0,    # byte 0x00 (NUL); never appears in our text data
            'bos_token_id': 0,
            'eos_token_id': 10,   # byte 0x0a (newline); natural example boundary
        },
    }
    with open(os.path.join(args.output_dir, 'sparrow_tokenizer.json'), 'w', encoding='utf-8') as f:
        json.dump(tok_meta, f, indent=2)

    print(f'  saved init to: {args.output_dir}')
    print(f'  next: train_local.py --resume {args.output_dir} ...')


if __name__ == '__main__':
    main()
