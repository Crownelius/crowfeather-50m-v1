"""Per-dataset distillation precache with unified output schema.

Each source dataset is downloaded SEPARATELY and converted to a unified JSONL
format. The trainer reads only the `text` field; the other fields enable
filtering, analysis, and reproducibility.

Output record schema (every line of every JSONL file):

    {
      "text":           "<chat-formatted with reserved special tokens>",
      "source_dataset": "AI-MO/NuminaMath-CoT" | "open-r1/Mixture-of-Thoughts:math" | ...,
      "domain":         "math" | "lang" | "code",
      "format":         "chat_with_thinking" | "chat" | "qa" | "raw",
      "has_thinking":   true | false,
      "tokens_est":     <int — len(text) // 4>,
      "metadata":       {<dataset-specific>}
    }

The `text` field always uses our 18 reserved special tokens:

    <|user|>
    {user content}
    <|assistant|>
    <|think|>
    {reasoning if present}
    </|think|>
    {final answer}
    <|eos|>

Robustness: every adapter prints the first record's schema, tries multiple
known field names, and emits a loud warning at the end if any source
yielded zero documents.
"""
import argparse, json, os, sys, traceback
from typing import Iterator, Optional


# ----- Format helpers -------------------------------------------------------

NL = '\n'

USER_TAG       = '<|user|>'
ASSISTANT_TAG  = '<|assistant|>'
THINK_OPEN     = '<|think|>'
THINK_CLOSE    = '</|think|>'
EOS_TAG        = '<|eos|>'


def chat_format(user: str, assistant_response: str, thinking: Optional[str] = None) -> str:
    """Build chat-format text using our reserved special tokens."""
    parts = [USER_TAG, user.strip(), ASSISTANT_TAG]
    if thinking and thinking.strip():
        parts.extend([THINK_OPEN, thinking.strip(), THINK_CLOSE])
    parts.append(assistant_response.strip())
    parts.append(EOS_TAG)
    return NL.join(parts)


def assistant_only_format(text: str) -> str:
    """For raw assistant-side data without an explicit user prompt."""
    return f'{ASSISTANT_TAG}{NL}{text.strip()}{NL}{EOS_TAG}'


def split_thinking(asst_msg: str) -> tuple:
    """If the assistant message contains <think>...</think>, split it out.
    Returns (thinking, final_response). thinking may be None."""
    if '<think>' in asst_msg and '</think>' in asst_msg:
        s = asst_msg.find('<think>')
        e = asst_msg.find('</think>')
        if 0 <= s < e:
            thinking = asst_msg[s + len('<think>'):e].strip()
            final = (asst_msg[:s] + asst_msg[e + len('</think>'):]).strip()
            return thinking, final or asst_msg.strip()
    return None, asst_msg.strip()


def make_record(text: str, source: str, domain: str, fmt: str,
                has_thinking: bool, metadata: dict = None) -> dict:
    return {
        'text': text,
        'source_dataset': source,
        'domain': domain,
        'format': fmt,
        'has_thinking': has_thinking,
        'tokens_est': len(text) // 4,
        'metadata': metadata or {},
    }


def _log_schema(name: str, sample: dict):
    keys = list(sample.keys())
    print(f'  [{name}] schema: keys={keys}')


# ----- Per-dataset adapters -------------------------------------------------

def stream_fineweb_edu(subset: str = 'sample-10BT') -> Iterator[dict]:
    """HuggingFaceFW/fineweb-edu — quality-filtered educational English web text.

    The foundational language source. Unlike the reasoning-heavy sources (Kimi,
    R1, NuminaMath), this is raw web text with no chat structure — just documents
    the model learns to continue fluently. Provides the English-fluency baseline
    that chat/reasoning data builds on. ODC-By 1.0 license.

    Configs: 'sample-10BT' (~30 GB, 10B tokens — sweet spot for sub-1B models),
             'sample-100BT' (~300 GB), 'sample-350BT' (~1 TB), 'default' (1.3T).
    """
    from datasets import load_dataset
    try:
        ds = load_dataset('HuggingFaceFW/fineweb-edu', subset, split='train', streaming=True)
    except Exception as e:
        print(f'  ERROR loading HuggingFaceFW/fineweb-edu:{subset}: {e}')
        print(f'  -> the dataset is gated; check your HF_TOKEN has read access')
        return
    schema_logged = False
    src = f'HuggingFaceFW/fineweb-edu:{subset}'
    for i, r in enumerate(ds):
        if not schema_logged:
            _log_schema('fineweb_edu', r); schema_logged = True
        text = r.get('text', '')
        if not text or not isinstance(text, str) or len(text) < 100:
            continue
        # Raw text — no special token wrapping. The trainer adds EOS at packing
        # time. Mixing raw web text with chat-formatted text is standard practice
        # (Phi, MiniCPM, OLMo all do this).
        yield make_record(
            text=text,
            source=src,
            domain='web',
            fmt='raw',
            has_thinking=False,
            metadata={
                'idx': i,
                'url': r.get('url', ''),
                'language_score': r.get('language_score', 0.0),
                'edu_score': r.get('score', 0.0),
                'token_count': r.get('token_count', 0),
                'dump': r.get('dump', ''),
            },
        )


def stream_numinamath() -> Iterator[dict]:
    """AI-MO/NuminaMath-CoT — fields: problem, solution, source, messages."""
    from datasets import load_dataset
    ds = load_dataset('AI-MO/NuminaMath-CoT', split='train', streaming=True)
    schema_logged = False
    for i, r in enumerate(ds):
        if not schema_logged:
            _log_schema('numinamath', r); schema_logged = True
        prob = r.get('problem', '')
        sol = r.get('solution', '')
        if not prob or not sol:
            continue
        thinking, final = split_thinking(sol)
        text = chat_format(prob, final, thinking)
        yield make_record(
            text=text,
            source='AI-MO/NuminaMath-CoT',
            domain='math',
            fmt='chat_with_thinking' if thinking else 'qa',
            has_thinking=bool(thinking),
            metadata={'idx': i, 'source_field': r.get('source', '')},
        )


def stream_metamathqa() -> Iterator[dict]:
    """meta-math/MetaMathQA — fields: query, response, type."""
    from datasets import load_dataset
    ds = load_dataset('meta-math/MetaMathQA', split='train', streaming=True)
    schema_logged = False
    for i, r in enumerate(ds):
        if not schema_logged:
            _log_schema('metamathqa', r); schema_logged = True
        q = r.get('query', '')
        a = r.get('response', '')
        if not q or not a:
            continue
        thinking, final = split_thinking(a)
        text = chat_format(q, final, thinking)
        yield make_record(
            text=text,
            source='meta-math/MetaMathQA',
            domain='math',
            fmt='chat_with_thinking' if thinking else 'qa',
            has_thinking=bool(thinking),
            metadata={'idx': i, 'qa_type': r.get('type', '')},
        )


def stream_r1_subset(subset: str, domain: str) -> Iterator[dict]:
    """open-r1/Mixture-of-Thoughts — robust to schema drift across configs.
    Tries: messages -> prompt/completion -> question/response -> raw text."""
    from datasets import load_dataset
    try:
        ds = load_dataset('open-r1/Mixture-of-Thoughts', subset, split='train', streaming=True)
    except Exception as e:
        print(f'  ERROR loading open-r1/Mixture-of-Thoughts:{subset}: {e}')
        return
    schema_logged = False
    src = f'open-r1/Mixture-of-Thoughts:{subset}'
    for i, r in enumerate(ds):
        if not schema_logged:
            _log_schema(f'r1_{subset}', r); schema_logged = True

        # Path 1: messages-format (math/science use this in current dataset rev)
        msgs = r.get('messages')
        if msgs and isinstance(msgs, list):
            user_msg = next((m['content'] for m in msgs if m.get('role') == 'user'), '')
            asst_msg = next((m['content'] for m in msgs if m.get('role') == 'assistant'), '')
            sys_msg  = next((m['content'] for m in msgs if m.get('role') == 'system'), '')
            if user_msg and asst_msg:
                if sys_msg:
                    user_msg = f'[System]: {sys_msg}{NL}{NL}{user_msg}'
                thinking, final = split_thinking(asst_msg)
                text = chat_format(user_msg, final, thinking)
                yield make_record(
                    text=text, source=src, domain=domain,
                    fmt='chat_with_thinking' if thinking else 'chat',
                    has_thinking=bool(thinking),
                    metadata={'idx': i, 'subset': subset, 'n_messages': len(msgs)},
                )
                continue

        # Path 2: prompt/completion (some R1 dumps use this)
        prompt = r.get('prompt') or r.get('question') or r.get('instruction')
        completion = r.get('completion') or r.get('response') or r.get('answer') or r.get('output')
        if prompt and completion:
            thinking, final = split_thinking(str(completion))
            text = chat_format(str(prompt), final, thinking)
            yield make_record(
                text=text, source=src, domain=domain,
                fmt='chat_with_thinking' if thinking else 'qa',
                has_thinking=bool(thinking),
                metadata={'idx': i, 'subset': subset, 'fields': list(r.keys())},
            )
            continue

        # Path 3: raw text
        raw = r.get('text') or r.get('content')
        if raw and isinstance(raw, str) and len(raw) > 100:
            thinking, final = split_thinking(raw)
            if thinking:
                text = chat_format('Continue:', final, thinking)
                fmt = 'chat_with_thinking'
            else:
                text = assistant_only_format(raw)
                fmt = 'raw'
            yield make_record(
                text=text, source=src, domain=domain,
                fmt=fmt,
                has_thinking=bool(thinking),
                metadata={'idx': i, 'subset': subset, 'raw_field': True},
            )


def stream_kimi() -> Iterator[dict]:
    """ianncity/KIMI-K2.5-1000000x — Kimi K2.5 reasoning traces (replaces Sonnet).

    K2.5 emits thinking in the assistant content, often with a <think> wrapper or
    similar marker. The adapter splits these into our <|think|>/</|think|> tokens
    so the trained model learns the same thinking-then-answering structure.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset('ianncity/KIMI-K2.5-1000000x', split='train', streaming=True)
    except Exception as e:
        print(f'  ERROR loading Kimi K2.5 dataset: {e}')
        return
    schema_logged = False
    src = 'ianncity/KIMI-K2.5-1000000x'
    for i, r in enumerate(ds):
        if not schema_logged:
            _log_schema('kimi', r); schema_logged = True

        # Messages-format first (most likely)
        msgs = r.get('messages') or r.get('conversation')
        if msgs and isinstance(msgs, list):
            user_msg = next((m.get('content', '') for m in msgs if m.get('role') == 'user'), '')
            asst_msg = next((m.get('content', '') for m in msgs if m.get('role') == 'assistant'), '')
            sys_msg  = next((m.get('content', '') for m in msgs if m.get('role') == 'system'), '')
            if user_msg and asst_msg:
                if sys_msg:
                    user_msg = f'[System]: {sys_msg}{NL}{NL}{user_msg}'
                thinking, final = split_thinking(asst_msg)
                text = chat_format(user_msg, final, thinking)
                yield make_record(
                    text=text, source=src, domain='lang',
                    fmt='chat_with_thinking' if thinking else 'chat',
                    has_thinking=bool(thinking),
                    metadata={'idx': i, 'n_messages': len(msgs)},
                )
                continue

        # Some Kimi-style dumps emit thinking in a separate field
        prompt = r.get('prompt') or r.get('input') or r.get('instruction') or r.get('question')
        output = r.get('output') or r.get('response') or r.get('completion') or r.get('answer')
        thinking_field = r.get('thinking') or r.get('reasoning') or r.get('cot') or r.get('chain_of_thought')
        if prompt and output:
            if thinking_field and isinstance(thinking_field, str) and thinking_field.strip():
                # Explicit separate thinking field — use it directly
                text = chat_format(str(prompt), str(output), str(thinking_field))
                yield make_record(
                    text=text, source=src, domain='lang',
                    fmt='chat_with_thinking', has_thinking=True,
                    metadata={'idx': i, 'fields': list(r.keys()), 'thinking_field': True},
                )
                continue
            # Otherwise try to split <think>...</think> from output
            thinking, final = split_thinking(str(output))
            text = chat_format(str(prompt), final, thinking)
            yield make_record(
                text=text, source=src, domain='lang',
                fmt='chat_with_thinking' if thinking else 'chat',
                has_thinking=bool(thinking),
                metadata={'idx': i, 'fields': list(r.keys())},
            )
            continue

        # Raw text fallback
        raw = r.get('text') or r.get('content') or r.get('body')
        if isinstance(raw, list):
            raw = NL.join(str(t) for t in raw)
        if raw and isinstance(raw, str) and len(raw) > 100:
            thinking, final = split_thinking(raw)
            if thinking:
                text = chat_format('Continue:', final, thinking)
                yield make_record(
                    text=text, source=src, domain='lang',
                    fmt='chat_with_thinking', has_thinking=True,
                    metadata={'idx': i, 'fields': list(r.keys())},
                )
            else:
                text = assistant_only_format(raw)
                yield make_record(
                    text=text, source=src, domain='lang',
                    fmt='raw', has_thinking=False,
                    metadata={'idx': i, 'fields': list(r.keys())},
                )


def stream_drive_jsonl(path: str, source_label: str, domain: str) -> Iterator[dict]:
    """User-curated JSONL on Drive (e.g. Opus 4.6 traces). Tries messages -> text."""
    if not os.path.exists(path):
        print(f'  Drive cache file not found: {path}')
        print(f'  -> upload your Opus traces to that path or skip this source')
        return
    sz_mb = os.path.getsize(path) / 1e6
    print(f'  Drive cache found: {path} ({sz_mb:.1f} MB)')
    schema_logged = False
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not schema_logged:
                _log_schema(f'drive:{source_label}', r); schema_logged = True

            msgs = r.get('messages') or r.get('conversation')
            if msgs and isinstance(msgs, list):
                user_msg = next((m.get('content', '') for m in msgs if m.get('role') == 'user'), '')
                asst_msg = next((m.get('content', '') for m in msgs if m.get('role') == 'assistant'), '')
                if user_msg and asst_msg:
                    thinking, final = split_thinking(asst_msg)
                    text = chat_format(user_msg, final, thinking)
                    yield make_record(
                        text=text, source=source_label, domain=domain,
                        fmt='chat_with_thinking' if thinking else 'chat',
                        has_thinking=bool(thinking),
                        metadata={'idx': i, 'source_path': path},
                    )
                    continue

            raw = r.get('text', '')
            if isinstance(raw, list):
                raw = NL.join(str(t) for t in raw)
            if raw and len(raw) > 100:
                # If it already has our special tokens, pass through as-is
                if USER_TAG in raw or ASSISTANT_TAG in raw:
                    text = raw if raw.endswith(EOS_TAG) else raw + NL + EOS_TAG
                    fmt = 'chat'
                    has_th = THINK_OPEN in raw
                else:
                    thinking, final = split_thinking(raw)
                    if thinking:
                        text = chat_format('Continue:', final, thinking)
                        fmt = 'chat_with_thinking'
                        has_th = True
                    else:
                        text = assistant_only_format(raw)
                        fmt = 'raw'
                        has_th = False
                yield make_record(
                    text=text, source=source_label, domain=domain,
                    fmt=fmt, has_thinking=has_th,
                    metadata={'idx': i, 'source_path': path},
                )


# ----- Driver ---------------------------------------------------------------

def write_dataset(target_dir: str, name: str, stream: Iterator[dict], max_chars: int) -> dict:
    """Write a single dataset's records to {target_dir}/{name}.jsonl with dedup."""
    out_path = os.path.join(target_dir, f'{name}.jsonl')

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        sz_mb = os.path.getsize(out_path) / 1e6
        print(f'  {name}: SKIP ({sz_mb:.1f} MB already cached)')
        return {'name': name, 'status': 'skip', 'cached_mb': sz_mb}

    if os.path.exists(out_path):
        os.remove(out_path)

    seen, n_docs, n_chars = set(), 0, 0
    has_thinking_count = 0
    fmt_counts = {}
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for record in stream:
                key = record['text'][:200]
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(record, ensure_ascii=True) + NL)
                n_docs += 1
                n_chars += len(record['text'])
                if record['has_thinking']:
                    has_thinking_count += 1
                fmt_counts[record['format']] = fmt_counts.get(record['format'], 0) + 1
                if n_chars >= max_chars:
                    break
    except Exception as e:
        print(f'  {name}: ERROR mid-stream: {type(e).__name__}: {e}')
        traceback.print_exc(limit=2)
        return {'name': name, 'status': 'error', 'n_docs': n_docs, 'error': str(e)}

    sz_mb = n_chars / 1e6
    pct_thinking = (100 * has_thinking_count / n_docs) if n_docs else 0
    fmt_summary = ', '.join(f'{k}={v}' for k, v in sorted(fmt_counts.items()))
    print(f'  {name}: {n_docs:,} docs ({sz_mb:.1f} MB)  '
          f'thinking={pct_thinking:.0f}%  formats=[{fmt_summary}]')
    return {
        'name': name, 'status': 'ok' if n_docs > 0 else 'empty',
        'n_docs': n_docs, 'n_chars': n_chars,
        'pct_thinking': pct_thinking, 'fmt_counts': fmt_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-dir', default='/content/distill_data')
    p.add_argument('--drive-cache', default='/content/drive/MyDrive/crowfeather_50m_v1/distill_data')
    p.add_argument('--budget-mb', type=int, default=8000,
                   help='total cap (math/lang/code split 30/40/30); ignored if --unlimited')
    p.add_argument('--unlimited', action='store_true',
                   help='download every source in full; ignores --budget-mb')
    p.add_argument('--force-refresh', action='store_true',
                   help='delete existing per-source JSONLs first (always recombines per-domain)')
    args = p.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)

    if args.unlimited:
        # Effectively infinite per-source budget; user manages Drive space
        BUDGET_WEB = BUDGET_MATH = BUDGET_LANG = BUDGET_CODE = int(1e15)
        print('UNLIMITED mode: every source downloaded in full.')
        print('  Make sure you have at least 60-100 GB free on Drive.')
    else:
        # Mix: web 40 / math 25 / lang 20 / code 15. Web is the foundational
        # English-fluency source; reasoning sources are the specialization layers.
        BUDGET_WEB  = int(args.budget_mb * 0.40 * 1e6)
        BUDGET_MATH = int(args.budget_mb * 0.25 * 1e6)
        BUDGET_LANG = int(args.budget_mb * 0.20 * 1e6)
        BUDGET_CODE = int(args.budget_mb * 0.15 * 1e6)
        print(f'Total budget:   {args.budget_mb} MB')
        print(f'  web  (40%): {BUDGET_WEB/1e6:.0f} MB  (FineWeb-Edu)')
        print(f'  math (25%): {BUDGET_MATH/1e6:.0f} MB')
        print(f'  lang (20%): {BUDGET_LANG/1e6:.0f} MB')
        print(f'  code (15%): {BUDGET_CODE/1e6:.0f} MB')
    print(f'Output dir:     {args.target_dir}')
    print(f'Drive cache:    {args.drive_cache}')

    # Per-source char budget. In --unlimited, all are int(1e15) so no source caps out.
    # Within a domain, budgeted weights set rough proportions when bounded.
    SOURCES = [
        # Web foundation — FIRST so it lands first and provides the English baseline
        ('fineweb_edu','web',  stream_fineweb_edu,                         BUDGET_WEB),
        # Math reasoning
        ('numinamath', 'math', stream_numinamath,                          int(BUDGET_MATH * 0.40)),
        ('metamathqa', 'math', stream_metamathqa,                          int(BUDGET_MATH * 0.30)),
        ('r1_math',    'math', lambda: stream_r1_subset('math', 'math'),   int(BUDGET_MATH * 0.30)),
        # Lang reasoning + chat (Kimi K2.5 provides long-form thinking traces)
        ('kimi',       'lang', stream_kimi,                                int(BUDGET_LANG * 0.55)),
        ('r1_science', 'lang', lambda: stream_r1_subset('science', 'lang'),int(BUDGET_LANG * 0.30)),
        ('opus',       'lang', lambda: stream_drive_jsonl(
                                  f'{args.drive_cache}/opus_4_6.jsonl',
                                  'Anthropic/opus-4.6-traces',
                                  'lang'),                                 int(BUDGET_LANG * 0.15)),
        # Code reasoning
        ('r1_code',    'code', lambda: stream_r1_subset('code', 'code'),   BUDGET_CODE),
    ]

    if args.force_refresh:
        print('\n--force-refresh: deleting existing per-source JSONLs')
        for name, _, _, _ in SOURCES:
            path = os.path.join(args.target_dir, f'{name}.jsonl')
            if os.path.exists(path):
                os.remove(path); print(f'  removed {path}')

    print('\n' + '=' * 70)
    print('PER-DATASET DOWNLOAD (separately)')
    print('=' * 70)

    results = []
    for name, domain, factory, budget in SOURCES:
        print(f'\n[{name}] domain={domain} budget={budget/1e6:.0f} MB')
        try:
            stream = factory()
            stats = write_dataset(args.target_dir, name, stream, budget)
        except Exception as e:
            print(f'  FATAL: {type(e).__name__}: {e}')
            traceback.print_exc(limit=2)
            stats = {'name': name, 'status': 'error', 'error': str(e)}
        results.append(stats)

    print('\n' + '=' * 70)
    print('COMBINING PER-DOMAIN')
    print('=' * 70)

    DOMAINS = {
        'web':  ['fineweb_edu'],
        'math': ['numinamath', 'metamathqa', 'r1_math'],
        'lang': ['kimi', 'r1_science', 'opus'],
        'code': ['r1_code'],
    }
    for d, srcs in DOMAINS.items():
        out = os.path.join(args.target_dir, f'{d}.jsonl')
        if os.path.exists(out):
            os.remove(out)
        sz, n = 0, 0
        with open(out, 'w', encoding='utf-8') as out_f:
            for s in srcs:
                p = os.path.join(args.target_dir, f'{s}.jsonl')
                if not os.path.exists(p):
                    continue
                with open(p, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line); sz += len(line); n += 1
        print(f'  {d}: {n:,} docs ({sz/1e6:.1f} MB)')

    print('\n' + '=' * 70)
    print('PER-DATASET SUMMARY')
    print('=' * 70)
    for s in results:
        name = s['name']
        if s['status'] == 'ok':
            print(f'  {name:18s} OK     {s["n_docs"]:>8,} docs  '
                  f'{s["n_chars"]/1e6:>7.1f} MB  thinking={s["pct_thinking"]:.0f}%')
        elif s['status'] == 'skip':
            print(f'  {name:18s} SKIP   ({s["cached_mb"]:>7.1f} MB cached -- pre-existing, unified format not guaranteed)')
        elif s['status'] == 'empty':
            print(f'  {name:18s} EMPTY  -- 0 docs, schema mismatch likely')
        else:
            print(f'  {name:18s} ERROR  -- {s.get("error", "unknown")}')

    failures = [s['name'] for s in results
                if s['status'] in ('empty', 'error')]
    if failures:
        print(f'\nDATA-LOSS WARNING: {failures}')
        print(f'  These sources produced 0 docs or errored. Investigate before training.')
        print(f'  Tip: run scripts/diagnose_datasets.py to inspect their schemas.')
    else:
        print(f'\nAll sources OK.')


if __name__ == '__main__':
    main()
