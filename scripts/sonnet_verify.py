"""Sonnet-based quality verifier for Crowfeather-50M-v1 training records.

Workflow: deterministic Python (precache_distill.py) does the conversion;
Sonnet 4.6 verifies a stratified sample. Much cheaper than full conversion
(~$0.005/row vs ~$0.014/row) because verification output is a short JSON
verdict, not reformatted text.

Per-row Sonnet output (JSON):
    {
      "verdict": "PASS" | "FAIL" | "BORDERLINE",
      "format_ok": true | false,
      "content_ok": true | false,
      "issues": ["mojibake", "empty_thinking", "ad_text", ...],
      "notes": "<one sentence>"
    }

Aggregate output (printed at end):
    - Overall pass/fail/borderline rate, with 95% Wilson CI
    - Per-domain breakdown
    - Top-10 issue tags by frequency
    - Up to 3 example failures per issue type (reason + 200-char excerpt)

Recommendation: pass rate >=90% means data is ready for training. Below 90%
means iterate on the deterministic conversion script (e.g. add mojibake
fix, tighten HTML strip, drop empty-thinking records before they ship).

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   # bash
    $env:ANTHROPIC_API_KEY = "sk-ant-..." # PowerShell

Usage:
    # Verify 1000 rows total, stratified across all *.jsonl in a directory
    python scripts/sonnet_verify.py --input-dir E:/crowfeather_data --sample 1000

    # Verify a single file
    python scripts/sonnet_verify.py --input E:/crowfeather_data/lang.jsonl --sample 500

    # Cost estimate only
    python scripts/sonnet_verify.py --input-dir E:/crowfeather_data --sample 5000 --cost-estimate-only
"""
import argparse, asyncio, json, math, os, random, sys, time
from collections import Counter, defaultdict
from typing import Optional


MODEL_ID = 'claude-sonnet-4-6'

# Sonnet 4.6 pricing per 1M tokens (USD, May 2026)
PRICE_INPUT_PER_M = 3.0
PRICE_OUTPUT_PER_M = 15.0

# Output is short JSON; budget 256 tokens hard.
MAX_OUTPUT_TOKENS = 256


SYSTEM_PROMPT = """You verify training records for Crowfeather-50M-v1, a 50M-parameter language model.

The model has 18 reserved special tokens; only these may appear as role markers:
  <|user|>  <|assistant|>  <|system|>  <|tool|>
  <|think|>  </|think|>
  <|tool_call|>  </|tool_call|>  <|tool_response|>  </|tool_response|>
  <|fim_prefix|>  <|fim_suffix|>  <|fim_middle|>  <|fim_pad|>
  <|pad|>  <|bos|>  <|eos|>  <|unk|>

Format rules per domain:
- domain=web    -> raw text, NO special tokens at all
- domain=math/lang/code with format=chat_with_thinking ->
    "<|user|>\\n{q}\\n<|assistant|>\\n<|think|>\\n{R}\\n</|think|>\\n{A}\\n<|eos|>"
    Both {q}, {R}, {A} must be non-empty and substantive.
- domain=math/lang/code with format=chat or qa ->
    "<|user|>\\n{q}\\n<|assistant|>\\n{a}\\n<|eos|>"
    No <|think|> block.

PASS verdict (all must hold):
- Text is coherent (English prose, math notation, or syntactically plausible code)
- Format matches the domain rules above
- Reasoning records have substantive thinking content (not just restating the question)
- No mojibake, no broken HTML, no obvious ads/SEO/banners
- No truncated mid-sentence garbage

FAIL verdict (any one is enough):
- Gibberish or untrainable noise
- Wrong format for domain (e.g. web record contains <|user|>)
- Missing or empty <|think|>...</|think|> block when domain says chat_with_thinking
- Severe encoding artifacts (mojibake unfixable, double-encoded UTF-8)
- Pure boilerplate / nav / footer with no real content

BORDERLINE: trainable but flawed. Note the specific issue.

Issue tags (use these — pick 0 or more from this fixed set):
  "mojibake", "broken_html", "empty_thinking", "wrong_format",
  "untrainable_noise", "ad_text", "boilerplate", "near_duplicate",
  "truncated_midsentence", "ascii_art", "non_english", "code_only_no_explanation",
  "thinking_just_repeats_question", "unsubstantive", "encoding_artifact"

Output ONLY a single JSON object, nothing else:
{
  "verdict": "PASS" | "FAIL" | "BORDERLINE",
  "format_ok": true | false,
  "content_ok": true | false,
  "issues": [<tag>, ...],
  "notes": "<one short sentence>"
}"""


# ----- Stratified sampling --------------------------------------------------

def stratified_sample(by_file: dict, total_n: int, rng: random.Random) -> list:
    """Sample total_n rows distributed proportionally across files but with
    a minimum 50 per file (so a small file isn't completely missed)."""
    files = list(by_file.keys())
    if not files:
        return []
    sizes = [len(by_file[f]) for f in files]
    total_rows = sum(sizes)

    per_file = {}
    for f, sz in zip(files, sizes):
        prop = sz / total_rows
        n_target = max(50, int(total_n * prop))
        per_file[f] = min(n_target, sz)

    # If we over-allocated, scale down proportionally
    total_alloc = sum(per_file.values())
    if total_alloc > total_n * 1.2:
        scale = total_n / total_alloc
        for f in per_file:
            per_file[f] = max(50, min(int(per_file[f] * scale), len(by_file[f])))

    sampled = []
    for f, n in per_file.items():
        sampled.extend(rng.sample(by_file[f], min(n, len(by_file[f]))))
    return sampled


# ----- Sonnet verification --------------------------------------------------

async def verify_row(client, row, semaphore, retries: int = 3):
    """Send a single row to Sonnet, return (verdict_dict, in_tokens, out_tokens)."""
    async with semaphore:
        text = row.get('text', '')
        # Clip extremely long records — verification doesn't need the whole 8K
        if len(text) > 12000:
            text = text[:6000] + '\n[...TRUNCATED IN VERIFY ONLY...]\n' + text[-2000:]

        prompt = (
            f'domain: {row.get("domain", "?")}\n'
            f'format: {row.get("format", "?")}\n'
            f'has_thinking: {row.get("has_thinking", False)}\n'
            f'source_dataset: {row.get("source_dataset", "?")}\n'
            f'\n'
            f'TEXT:\n{text}'
        )

        for attempt in range(retries):
            try:
                resp = await client.messages.create(
                    model=MODEL_ID,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=[{
                        'type': 'text',
                        'text': SYSTEM_PROMPT,
                        'cache_control': {'type': 'ephemeral'},
                    }],
                    messages=[{'role': 'user', 'content': prompt}],
                )
                raw = resp.content[0].text.strip()
                # Strip ```json fences if Sonnet adds them
                if raw.startswith('```'):
                    raw = raw.strip('`').lstrip('json').strip()
                verdict = json.loads(raw)
                in_tok = resp.usage.input_tokens \
                    + getattr(resp.usage, 'cache_read_input_tokens', 0) \
                    + getattr(resp.usage, 'cache_creation_input_tokens', 0)
                out_tok = resp.usage.output_tokens
                return (verdict, in_tok, out_tok)
            except json.JSONDecodeError:
                if attempt == retries - 1:
                    return ({'verdict': 'BORDERLINE', 'format_ok': False,
                             'content_ok': False,
                             'issues': ['verifier_json_parse_fail'],
                             'notes': 'Sonnet output was not valid JSON'},
                            0, 0)
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == retries - 1:
                    return ({'verdict': 'BORDERLINE', 'format_ok': False,
                             'content_ok': False,
                             'issues': ['verifier_api_error'],
                             'notes': f'API error: {e}'},
                            0, 0)
                await asyncio.sleep(2 ** attempt)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ----- Driver ---------------------------------------------------------------

def gather_inputs(args) -> dict:
    """Returns {filename: [row, ...]}."""
    by_file = {}
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            rows = []
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        by_file[os.path.basename(args.input)] = rows
    if args.input_dir:
        for fn in sorted(os.listdir(args.input_dir)):
            # Verify only the per-domain combined files (web/math/lang/code)
            if not fn.endswith('.jsonl'):
                continue
            if fn.endswith('.invalid.jsonl') or fn.endswith('.verify.jsonl'):
                continue
            base = fn.rsplit('.', 1)[0]
            if base not in ('web', 'math', 'lang', 'code'):
                continue
            path = os.path.join(args.input_dir, fn)
            rows = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            by_file[fn] = rows
    return by_file


def estimate_cost(by_file: dict, total_sample: int) -> tuple:
    """Estimate cost. Assumes ~3000 input + 100 output tokens per verify."""
    in_tokens = total_sample * 3000  # text + system prompt amortized via cache
    out_tokens = total_sample * 100  # short JSON
    cost = (in_tokens * PRICE_INPUT_PER_M + out_tokens * PRICE_OUTPUT_PER_M) / 1e6
    return cost, in_tokens, out_tokens


async def main_async(args):
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print('ERROR: anthropic SDK not installed. Run: pip install anthropic')
        sys.exit(1)

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not args.cost_estimate_only and not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Pass --api-key or export it.')
        sys.exit(1)

    by_file = gather_inputs(args)
    if not by_file:
        print('ERROR: no input rows found.')
        sys.exit(1)
    total_rows = sum(len(rs) for rs in by_file.values())
    print(f'Loaded {total_rows:,} rows across {len(by_file)} files:')
    for f, rs in by_file.items():
        print(f'  {f:25s} {len(rs):>10,} rows')

    rng = random.Random(args.seed)
    sample = stratified_sample(by_file, args.sample, rng)
    print(f'\nStratified sample size: {len(sample):,} rows (seed={args.seed})')

    cost, in_tok, out_tok = estimate_cost(by_file, len(sample))
    print(f'\nCOST ESTIMATE')
    print(f'  rows to verify:   {len(sample):,}')
    print(f'  input tokens:    ~{in_tok/1e6:>6.2f} M @ ${PRICE_INPUT_PER_M}/M')
    print(f'  output tokens:   ~{out_tok/1e6:>6.2f} M @ ${PRICE_OUTPUT_PER_M}/M')
    print(f'  TOTAL estimated:  ${cost:.2f}')
    print(f'  (real cost typically 10-15% lower from system-prompt caching)')

    if args.cost_estimate_only:
        return

    if cost > 50 and not args.yes:
        print(f'\nCost exceeds $50. Re-run with --yes to confirm.')
        sys.exit(1)

    # Run verification
    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f'\nVerifying with {MODEL_ID}, concurrency={args.concurrency}...')

    verdicts = []
    total_in = total_out = 0
    t_start = time.time()

    CHUNK = 50
    for ci in range(0, len(sample), CHUNK):
        chunk = sample[ci:ci + CHUNK]
        results = await asyncio.gather(*[
            verify_row(client, r, semaphore) for r in chunk
        ])
        for original, (verdict, in_t, out_t) in zip(chunk, results):
            verdicts.append((original, verdict))
            total_in += in_t
            total_out += out_t
        done = ci + len(chunk)
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1)
        eta_min = (len(sample) - done) / max(rate, 0.01) / 60
        cost_so_far = (total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1e6
        print(f'  {done:>5}/{len(sample):,}  rate={rate:.1f}/s  '
              f'cost=${cost_so_far:.2f}  ETA={eta_min:.1f}m', flush=True)

    # Persist verdicts
    out_path = args.output or 'sonnet_verify_report.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for original, verdict in verdicts:
            f.write(json.dumps({
                'original_idx': original.get('metadata', {}).get('idx'),
                'source_dataset': original.get('source_dataset'),
                'domain': original.get('domain'),
                'format': original.get('format'),
                'verdict': verdict,
                'text_excerpt': original.get('text', '')[:200],
            }, ensure_ascii=True) + '\n')

    # ----- Aggregate report -----
    print()
    print('=' * 78)
    print(f'VERIFICATION REPORT  (n={len(verdicts):,})')
    print('=' * 78)

    overall_counts = Counter(v['verdict'] for _, v in verdicts)
    total = sum(overall_counts.values())
    pass_n = overall_counts.get('PASS', 0)
    fail_n = overall_counts.get('FAIL', 0)
    border_n = overall_counts.get('BORDERLINE', 0)
    pass_lo, pass_hi = wilson_ci(pass_n, total)
    fail_lo, fail_hi = wilson_ci(fail_n, total)
    print(f'  PASS:       {pass_n:>5} / {total:<5} = {100*pass_n/total:.1f}%  '
          f'95% CI [{100*pass_lo:.1f}%, {100*pass_hi:.1f}%]')
    print(f'  FAIL:       {fail_n:>5} / {total:<5} = {100*fail_n/total:.1f}%  '
          f'95% CI [{100*fail_lo:.1f}%, {100*fail_hi:.1f}%]')
    print(f'  BORDERLINE: {border_n:>5} / {total:<5} = {100*border_n/total:.1f}%')

    print(f'\nPER-DOMAIN BREAKDOWN')
    by_domain = defaultdict(lambda: Counter())
    for original, verdict in verdicts:
        by_domain[original.get('domain', '?')][verdict['verdict']] += 1
    for d in sorted(by_domain):
        c = by_domain[d]
        n = sum(c.values())
        p = c.get('PASS', 0)
        print(f'  {d:6s}  n={n:>5}  pass={100*p/n:.1f}%  '
              f'fail={100*c.get("FAIL", 0)/n:.1f}%  '
              f'borderline={100*c.get("BORDERLINE", 0)/n:.1f}%')

    print(f'\nTOP-10 ISSUE TAGS')
    issue_counter = Counter()
    for _, v in verdicts:
        for issue in v.get('issues', []):
            issue_counter[issue] += 1
    for issue, count in issue_counter.most_common(10):
        pct = 100 * count / total
        print(f'  {issue:35s} {count:>5} ({pct:.1f}% of sampled rows)')

    # Failure examples — first 3 per top issue
    print(f'\nFAILURE EXAMPLES (first 3 per top issue)')
    for issue, _ in issue_counter.most_common(5):
        examples = [(o, v) for o, v in verdicts if issue in v.get('issues', [])][:3]
        if not examples:
            continue
        print(f'\n  [{issue}]')
        for original, verdict in examples:
            print(f'    src={original.get("source_dataset")} domain={original.get("domain")}')
            print(f'    notes: {verdict.get("notes", "")[:150]}')
            print(f'    text:  {original.get("text", "")[:150].replace(chr(10), " / ")!r}')
            print()

    # Recommendation
    print('=' * 78)
    pass_rate = pass_n / total
    if pass_rate >= 0.90:
        print(f'RECOMMENDATION: data is training-ready (pass rate {100*pass_rate:.1f}% >= 90%)')
    elif pass_rate >= 0.75:
        print(f'RECOMMENDATION: data is borderline (pass rate {100*pass_rate:.1f}%).')
        print('  Top issues above are actionable — patch the precache adapters and re-verify.')
    else:
        print(f'RECOMMENDATION: do NOT train yet (pass rate {100*pass_rate:.1f}% < 75%).')
        print('  Critical issues need fixing in the deterministic precache. See top tags above.')

    print(f'\nTotal cost: ${(total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1e6:.2f}')
    print(f'Total time: {(time.time() - t_start)/60:.1f} minutes')
    print(f'Per-row verdicts written to: {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='single JSONL file to verify')
    src.add_argument('--input-dir',
                     help='directory containing web/math/lang/code .jsonl files')
    p.add_argument('--output', default=None,
                   help='per-row verdict JSONL (default: sonnet_verify_report.jsonl)')
    p.add_argument('--sample', type=int, default=1000,
                   help='total stratified sample size (default 1000)')
    p.add_argument('--concurrency', type=int, default=10)
    p.add_argument('--api-key', default=None)
    p.add_argument('--seed', type=int, default=20260504)
    p.add_argument('--cost-estimate-only', action='store_true')
    p.add_argument('--yes', action='store_true', help='skip >$50 confirmation')
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
