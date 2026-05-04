"""Train 32K Byte-Level BPE on the distillation corpus.

Per-digit wrap is applied to all training text so the BPE merge table
never produces multi-digit number tokens. Special tokens (FIM, chat
roles, think tags) get the lowest IDs and bypass BPE merging.

Usage:
    python train_bpe.py --cache-dir /content/distill_data \\
                        --output-dir /content/tokenizer
"""
import argparse, json, os, re

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers


_DIGIT_RUN = re.compile(r'\d{2,}')
def per_digit_wrap(text):
    return _DIGIT_RUN.sub(lambda m: ' '.join(m.group()), text)


# Special tokens get IDs 0..len-1 in this exact order.
SPECIAL_TOKENS = [
    "<|pad|>",             # 0  padding
    "<|bos|>",             # 1  beginning of sequence
    "<|eos|>",             # 2  end of sequence
    "<|unk|>",             # 3  reserved (byte fallback handles unseen text)
    "<|fim_prefix|>",      # 4  FIM: marks start of prefix
    "<|fim_suffix|>",      # 5  FIM: marks start of suffix
    "<|fim_middle|>",      # 6  FIM: marks start of middle (target)
    "<|fim_pad|>",         # 7  FIM: padding inside infill regions
    "<|user|>",            # 8  chat role
    "<|assistant|>",       # 9  chat role
    "<|system|>",          # 10 chat role
    "<|tool|>",            # 11 chat role
    "<|think|>",           # 12 chain-of-thought open
    "</|think|>",          # 13 chain-of-thought close
    "<|tool_call|>",       # 14
    "</|tool_call|>",      # 15
    "<|tool_response|>",   # 16
    "</|tool_response|>",  # 17
]


def stream_corpus(jsonl_paths, max_chars):
    chars = 0
    for p in jsonl_paths:
        if not os.path.exists(p):
            print(f'  WARN: {p} not found, skipping')
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get('text', '')
                except json.JSONDecodeError:
                    continue
                if not text or not text.strip():
                    continue
                yield per_digit_wrap(text)
                chars += len(text)
                if chars >= max_chars:
                    print(f'  reached corpus cap of {max_chars/1e9:.2f} GB at {os.path.basename(p)}')
                    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', required=True, help='dir of *.jsonl files')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--vocab-size', type=int, default=32_768)
    p.add_argument('--max-corpus-chars', type=int, default=int(2e9), help='2 GB cap default')
    p.add_argument('--min-frequency', type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    jsonl_paths = sorted([
        os.path.join(args.cache_dir, fn)
        for fn in os.listdir(args.cache_dir)
        if fn.endswith('.jsonl')
        and os.path.getsize(os.path.join(args.cache_dir, fn)) > 1000
    ])
    if not jsonl_paths:
        raise RuntimeError(f'no .jsonl files in {args.cache_dir}')
    print(f'  Training BPE on {len(jsonl_paths)} files, max corpus = {args.max_corpus_chars/1e9:.1f} GB')
    for jp in jsonl_paths:
        print(f'    {os.path.basename(jp):25s} {os.path.getsize(jp)/1e6:8.1f} MB')

    tokenizer.train_from_iterator(
        stream_corpus(jsonl_paths, args.max_corpus_chars),
        trainer=trainer,
    )

    raw_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer.save(raw_path)
    print(f'  Saved tokenizer.json to {raw_path}')
    print(f'  Vocab size: {tokenizer.get_vocab_size()}')

    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        bos_token='<|bos|>',
        eos_token='<|eos|>',
        pad_token='<|pad|>',
        unk_token='<|unk|>',
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )
    hf_tok.save_pretrained(args.output_dir)
    print(f'  Saved HF tokenizer files to {args.output_dir}')

    test_strings = [
        "The quick brown fox jumps over 1 2 3 4 5 6 lazy dogs.",
        per_digit_wrap("The cost was $12,345.67 for 200 units."),
        "<|fim_prefix|>def add(a, b):<|fim_suffix|>    return result<|fim_middle|>    result = a + b",
        "<|user|>" + chr(10) + "What is 2+2?" + chr(10) + "<|assistant|>" + chr(10) + "<|think|>" + chr(10) + "Simple addition: 2+2=4" + chr(10) + "</|think|>" + chr(10) + "The answer is 4." + chr(10) + "<|eos|>",
    ]
    print('\n  === Sanity check ===')
    for s in test_strings:
        ids = hf_tok.encode(s, add_special_tokens=False)
        decoded = hf_tok.decode(ids, skip_special_tokens=False)
        roundtrip = (decoded == s)
        snippet = s[:60].replace(chr(10), '\\n')
        print(f'    in : {snippet!r}')
        print(f'    ids: {len(ids):4d} tokens   roundtrip: {"OK" if roundtrip else "MISMATCH"}')
    print('\n  done.')


if __name__ == '__main__':
    main()
