"""Build a fresh dense Qwen3 50M init from a trained tokenizer.

Path A spec:
    vocab_size      = 32_768  (taken from tokenizer)
    hidden_size     = 448
    num_layers      = 12
    num_heads       = 8 query, 4 KV  (GQA 2:1)
    head_dim        = 56  (448 / 8)
    intermediate    = 1792  (~4x hidden, SwiGLU 3-matrix)
    max_position    = 16384
    tie_embeddings  = True
    rope_theta      = 1_000_000

Total ~50.8M parameters.
"""
import argparse, os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--hidden', type=int, default=448)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--head-dim', type=int, default=56)
    p.add_argument('--intermediate', type=int, default=1792)
    p.add_argument('--max-pos', type=int, default=16384)
    p.add_argument('--rope-theta', type=float, default=1_000_000.0)
    p.add_argument('--seed', type=int, default=20260504)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    vocab_size = len(tok)

    cfg = Qwen3Config(
        vocab_size=vocab_size,
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
        use_cache=False,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        torch_dtype='bfloat16',
    )
    print(f'  Config: vocab={vocab_size} hidden={args.hidden} layers={args.layers} '
          f'q={args.heads} kv={args.kv_heads} head_dim={args.head_dim} '
          f'inter={args.intermediate} max_pos={args.max_pos}')

    model = Qwen3ForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    n_total = sum(p.numel() for p in model.parameters())
    n_embed = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n.lower())
    n_layers_p = n_total - n_embed
    print(f'  Total params:  {n_total/1e6:.2f}M')
    print(f'    Embedding:   {n_embed/1e6:.2f}M ({100*n_embed/n_total:.1f}%)')
    print(f'    Transformer: {n_layers_p/1e6:.2f}M ({100*n_layers_p/n_total:.1f}%)')

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)
    print(f'  Saved init to {args.output_dir}')


if __name__ == '__main__':
    main()
