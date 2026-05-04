"""Convert distillation JSONL rows into the optimal Crowfeather-50M format
using Anthropic Sonnet 4.6.

For each row, sends the original `text` field to Sonnet with a strict system
prompt (see docs/OPTIMAL_FORMAT.md). Sonnet returns either:
  - cleaned text in the optimal format, or
  - the literal string SKIP (untrainable record)

The script then validates conformance, writes valid rows to --output, and
sidecars invalid rows to {output}.invalid.jsonl for manual review.

WARNING: Sonnet 4.6 pricing is $3/M input tokens, $15/M output tokens.
At ~3K input + 1K output per row, cost is ~$0.014/row. Doing 5M rows is
~$70K. Always start with --sample 1000 (~$14) to gauge quality before
scaling. The script prints a cost estimate and prompts for confirmation
on jobs >$100.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    # 1K sample (default — ~$14, 5-10 min)
    python scripts/sonnet_convert.py \\
        --input  E:/crowfeather_data/lang.jsonl \\
        --output E:/crowfeather_data/lang.sonnet.jsonl

    # Larger sample
    python scripts/sonnet_convert.py \\
        --input  E:/crowfeather_data/lang.jsonl \\
        --output E:/crowfeather_data/lang.sonnet.jsonl \\
        --sample 50000

    # Cost estimate only (no API calls)
    python scripts/sonnet_convert.py \\
        --input  E:/crowfeather_data/lang.jsonl \\
        --cost-estimate-only
"""
import argparse, asyncio, json, os, random, sys, time
from typing import Optional


MODEL_ID = 'claude-sonnet-4-6'

# Pricing per 1M tokens (Sonnet 4.6, USD, May 2026)
PRICE_INPUT_PER_M = 3.0
PRICE_OUTPUT_PER_M = 15.0
# Prompt-cache reads/writes are slightly cheaper but not modeled here for
# conservatism. Real cost is typically 5-15% lower than the printed estimate.

SYSTEM_PROMPT = """You are converting raw training records into the optimal format for Crowfeather-50M-v1, a 50M-parameter language model. The model has 18 reserved special tokens; only these may appear as role markers:

  <|user|>  <|assistant|>  <|system|>  <|tool|>
  <|think|>  </|think|>
  <|tool_call|>  </|tool_call|>  <|tool_response|>  </|tool_response|>
  <|fim_prefix|>  <|fim_suffix|>  <|fim_middle|>  <|fim_pad|>
  <|pad|>  <|bos|>  <|eos|>  <|unk|>

Output exactly the cleaned text, no JSON wrapper, no commentary.

If the record's domain is "web", output ONLY the cleaned document text -- no special tokens at all.

If the record is a chat with reasoning, output:
<|user|>
{user prompt}
<|assistant|>
<|think|>
{concise reasoning, 100-2000 tokens, preserves key derivation steps}
</|think|>
{final answer}
<|eos|>

If the record is chat without reasoning, output:
<|user|>
{user prompt}
<|assistant|>
{response}
<|eos|>

Cleanup mandate: strip HTML/markdown clutter, fix mojibake, remove ads/SEO/banners, collapse repetitive whitespace, truncate to ~4K tokens preserving the highest-value section. Drop the record entirely (output the literal token SKIP and nothing else) if the input is gibberish, near-duplicate boilerplate, untrainable noise, or has no coherent reasoning when reasoning is required.

Never invent content. If you cannot extract a clean version that preserves the original meaning, output SKIP."""


# ----- Format validation ----------------------------------------------------

RESERVED_TOKENS = [
    '<|user|>', '<|assistant|>', '<|system|>', '<|tool|>',
    '<|think|>', '</|think|>',
    '<|tool_call|>', '</|tool_call|>', '<|tool_response|>', '</|tool_response|>',
    '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>', '<|fim_pad|>',
    '<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>',
]


def validate(text: str, fmt: str) -> tuple:
    """Return (is_valid, reason)."""
    if fmt == 'raw':
        for tok in RESERVED_TOKENS:
            if tok in text:
                return False, f'web record contains reserved token {tok!r}'
        if len(text) < 100:
            return False, f'web record too short ({len(text)} chars)'
        return True, 'ok'

    if fmt == 'chat_with_thinking':
        if not text.startswith('<|user|>'):
            return False, 'must start with <|user|>'
        if '<|assistant|>' not in text:
            return False, 'missing <|assistant|>'
        if '<|think|>' not in text or '</|think|>' not in text:
            return False, 'missing thinking block'
        if not text.rstrip().endswith('<|eos|>'):
            return False, 'missing trailing <|eos|>'
        i_user, i_asst = text.find('<|user|>'), text.find('<|assistant|>')
        i_think_o = text.find('<|think|>')
        i_think_c = text.find('</|think|>')
        if not (i_user < i_asst < i_think_o < i_think_c):
            return False, 'token ordering wrong'
        return True, 'ok'

    if fmt in ('chat', 'qa'):
        if not text.startswith('<|user|>'):
            return False, 'must start with <|user|>'
        if '<|assistant|>' not in text:
            return False, 'missing <|assistant|>'
        if not text.rstrip().endswith('<|eos|>'):
            return False, 'missing trailing <|eos|>'
        if '<|think|>' in text:
            return False, 'chat fmt should not have <|think|> (use chat_with_thinking)'
        return True, 'ok'

    return False, f'unknown format {fmt!r}'


# ----- Conversion -----------------------------------------------------------

async def convert_row(client, row, semaphore, retries: int = 3):
    """Convert a single row via Sonnet. Returns ('ok', new_row) | ('skip', None) | ('error', err)."""
    async with semaphore:
        original = row.get('text', '')
        domain = row.get('domain', '?')
        fmt = row.get('format', '?')

        prompt = (
            f'Domain: {domain}\n'
            f'Source format: {fmt}\n'
            f'has_thinking_in_source: {row.get("has_thinking", False)}\n'
            f'\n'
            f'Original text:\n{original}'
        )

        for attempt in range(retries):
            try:
                resp = await client.messages.create(
                    model=MODEL_ID,
                    max_tokens=8192,
                    system=[{
                        'type': 'text',
                        'text': SYSTEM_PROMPT,
                        'cache_control': {'type': 'ephemeral'},
                    }],
                    messages=[{'role': 'user', 'content': prompt}],
                )
                cleaned = resp.content[0].text.strip()
                usage_in = resp.usage.input_tokens + getattr(resp.usage, 'cache_read_input_tokens', 0) + getattr(resp.usage, 'cache_creation_input_tokens', 0)
                usage_out = resp.usage.output_tokens

                if cleaned == 'SKIP':
                    return ('skip', None, usage_in, usage_out)

                # Determine output format from content (Sonnet may downgrade
                # chat_with_thinking -> chat if it dropped the thinking block).
                if not cleaned.startswith('<|user|>'):
                    out_fmt = 'raw'
                elif '<|think|>' in cleaned:
                    out_fmt = 'chat_with_thinking'
                else:
                    out_fmt = 'chat'

                ok, reason = validate(cleaned, out_fmt)
                new_row = dict(row)
                new_row['text'] = cleaned
                new_row['format'] = out_fmt
                new_row['has_thinking'] = (out_fmt == 'chat_with_thinking')
                new_row['tokens_est'] = len(cleaned) // 4
                meta = dict(new_row.get('metadata', {}))
                meta['converted_by'] = MODEL_ID
                meta['original_tokens_est'] = len(original) // 4
                meta['format_validated'] = ok
                if not ok:
                    meta['format_validation_reason'] = reason
                new_row['metadata'] = meta

                return ('ok' if ok else 'invalid', new_row, usage_in, usage_out)

            except Exception as e:
                if attempt == retries - 1:
                    return ('error', str(e), 0, 0)
                await asyncio.sleep(2 ** attempt)


# ----- Driver ---------------------------------------------------------------

def estimate_cost(rows, sample_size: Optional[int] = None):
    if sample_size and sample_size < len(rows):
        target = sample_size
    else:
        target = len(rows)
    avg_chars = sum(len(r.get('text', '')) for r in rows[:1000]) / max(1, len(rows[:1000]))
    in_tokens_per_row = avg_chars / 4 + 200  # +SYSTEM_PROMPT amortized via cache
    out_tokens_per_row = avg_chars / 4 * 0.7  # output usually ~70% of input
    in_total = target * in_tokens_per_row
    out_total = target * out_tokens_per_row
    cost = (in_total * PRICE_INPUT_PER_M + out_total * PRICE_OUTPUT_PER_M) / 1e6
    return cost, in_total, out_total


def already_processed(output_path: str) -> set:
    """Return set of original-row indices already in --output (resume support)."""
    seen = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                idx = r.get('metadata', {}).get('idx')
                if idx is not None:
                    seen.add(idx)
            except json.JSONDecodeError:
                continue
    return seen


async def main_async(args):
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print('ERROR: anthropic SDK not installed. Run: pip install anthropic')
        sys.exit(1)

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Pass --api-key or export it.')
        sys.exit(1)

    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f'Loaded {len(rows):,} rows from {args.input}')

    if args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)
        print(f'Sampled {args.sample:,} rows (seed={args.seed})')

    cost, in_tok, out_tok = estimate_cost(rows)
    print(f'\nCOST ESTIMATE')
    print(f'  rows to process:   {len(rows):,}')
    print(f'  input tokens est:  {in_tok/1e6:>8.1f} M  @ ${PRICE_INPUT_PER_M}/M')
    print(f'  output tokens est: {out_tok/1e6:>8.1f} M  @ ${PRICE_OUTPUT_PER_M}/M')
    print(f'  TOTAL estimated:   ${cost:>8.2f}')
    print(f'  (real cost typically 5-15% lower from system-prompt caching)')

    if args.cost_estimate_only:
        return

    if cost > 100 and not args.yes:
        print(f'\nCost exceeds $100. Re-run with --yes to confirm.')
        sys.exit(1)

    # Resume support
    already = already_processed(args.output) if args.resume else set()
    if already:
        print(f'\nResume: {len(already):,} rows already in {args.output}; skipping')
        rows = [r for r in rows if r.get('metadata', {}).get('idx') not in already]
        if not rows:
            print('Nothing new to convert.')
            return

    invalid_path = args.output + '.invalid.jsonl'
    print(f'\nWriting valid rows to:   {args.output}')
    print(f'Writing invalid rows to: {invalid_path}')
    print(f'Concurrency: {args.concurrency}')
    print()

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    n_ok = n_skip = n_invalid = n_error = 0
    total_in = total_out = 0
    t_start = time.time()

    out_f = open(args.output, 'a', encoding='utf-8')
    inv_f = open(invalid_path, 'a', encoding='utf-8')
    try:
        CHUNK = 50
        for ci in range(0, len(rows), CHUNK):
            chunk = rows[ci:ci + CHUNK]
            results = await asyncio.gather(*[
                convert_row(client, r, semaphore) for r in chunk
            ])
            for status, payload, ut_in, ut_out in results:
                total_in += ut_in
                total_out += ut_out
                if status == 'ok':
                    out_f.write(json.dumps(payload, ensure_ascii=True) + '\n')
                    n_ok += 1
                elif status == 'invalid':
                    inv_f.write(json.dumps(payload, ensure_ascii=True) + '\n')
                    n_invalid += 1
                elif status == 'skip':
                    n_skip += 1
                else:  # error
                    n_error += 1
            done = ci + len(chunk)
            elapsed = time.time() - t_start
            rate = done / max(elapsed, 1)
            eta_min = (len(rows) - done) / max(rate, 0.01) / 60
            cost_so_far = (total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1e6
            print(f'  {done:>6,}/{len(rows):,}  ok={n_ok} invalid={n_invalid} skip={n_skip} err={n_error}  '
                  f'rate={rate:.1f}/s  cost=${cost_so_far:.2f}  ETA={eta_min:.1f}m', flush=True)
    finally:
        out_f.close()
        inv_f.close()

    print()
    print('=' * 70)
    print('DONE')
    print('=' * 70)
    print(f'  valid:   {n_ok:,}    -> {args.output}')
    print(f'  invalid: {n_invalid:,}    -> {invalid_path}')
    print(f'  skipped: {n_skip:,}    (Sonnet judged untrainable)')
    print(f'  errors:  {n_error:,}    (re-run to retry these)')
    print(f'  total cost: ${(total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1e6:.2f}')
    print(f'  wall time:  {(time.time() - t_start)/60:.1f} minutes')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='input JSONL (per-source or per-domain)')
    p.add_argument('--output', help='output JSONL (default: {input}.sonnet.jsonl)')
    p.add_argument('--sample', type=int, default=1000,
                   help='process N random rows (default 1000; use 0 for all)')
    p.add_argument('--concurrency', type=int, default=10,
                   help='parallel API calls (default 10)')
    p.add_argument('--api-key', default=None,
                   help='Anthropic API key (default: $ANTHROPIC_API_KEY)')
    p.add_argument('--seed', type=int, default=20260504, help='random seed for sampling')
    p.add_argument('--cost-estimate-only', action='store_true',
                   help='print cost estimate without making API calls')
    p.add_argument('--yes', action='store_true',
                   help='skip the >$100 confirmation prompt')
    p.add_argument('--resume', action='store_true',
                   help='skip rows whose metadata.idx already appears in --output')
    args = p.parse_args()

    if not args.output:
        args.output = args.input.rsplit('.jsonl', 1)[0] + '.sonnet.jsonl'
    if args.sample == 0:
        args.sample = None  # all rows

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
