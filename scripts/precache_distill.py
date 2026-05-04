"""Precache distillation traces split by domain into math/lang/code JSONLs."""
import argparse, json, os, traceback


def append_to(name, target_dir, text_iter, max_chars):
    out = os.path.join(target_dir, f'{name}.jsonl')
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f'  {name}: SKIP ({os.path.getsize(out)/1e6:.1f} MB)'); return
    seen, chars, docs = set(), 0, 0
    with open(out, 'w', encoding='utf-8') as f:
        for text in text_iter:
            if not text or not text.strip(): continue
            key = text[:200]
            if key in seen: continue
            seen.add(key)
            f.write(json.dumps({'text': text}) + '\n')
            chars += len(text); docs += 1
            if chars >= max_chars: break
    print(f'  {name}: {docs:,} docs ({chars/1e6:.1f} MB)')


def safe_run(label, fn):
    print(f'\n{label}')
    try: fn()
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}')
        traceback.print_exc(limit=2)


def stream_r1_subset(subset, max_docs=300_000):
    from datasets import load_dataset
    yielded = 0
    try:
        ds = load_dataset('open-r1/Mixture-of-Thoughts', subset, split='train', streaming=True)
        BS = chr(92); NL = BS + 'n'
        for r in ds:
            if yielded >= max_docs: return
            msgs = r.get('messages', [])
            if not msgs: continue
            parts = [f'<|{m.get("role","?")}|>' + NL + str(m.get('content', '')) for m in msgs]
            text = NL.join(parts)
            if len(text) > 100:
                yield text; yielded += 1
    except Exception as e:
        print(f'  open-r1 subset {subset!r} unavailable: {e}')


def stream_sonnet(max_docs=200_000):
    from datasets import load_dataset
    ds = load_dataset('Roman1111111/claude-sonnet-4.6-120000x', split='train', streaming=True)
    for i, r in enumerate(ds):
        if i >= max_docs: return
        text = r.get('text') or r.get('output') or r.get('response') or ''
        if isinstance(text, list): text = chr(10).join(str(t) for t in text)
        if text and len(text) > 100: yield text


def stream_numinamath():
    from datasets import load_dataset
    ds = load_dataset('AI-MO/NuminaMath-CoT', split='train', streaming=True)
    for r in ds:
        prob, sol = r.get('problem', ''), r.get('solution', '')
        if prob and sol:
            yield f'Problem: {prob}' + chr(10) + chr(10) + f'Solution: {sol}'


def stream_metamathqa():
    from datasets import load_dataset
    ds = load_dataset('meta-math/MetaMathQA', split='train', streaming=True)
    for r in ds:
        q, a = r.get('query', ''), r.get('response', '')
        if q and a:
            yield f'Question: {q}' + chr(10) + chr(10) + f'Answer: {a}'


def stream_drive_jsonl(path):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                yield r.get('text') or json.dumps(r)
            except json.JSONDecodeError: continue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-dir', default='/content/distill_data')
    p.add_argument('--drive-cache', default='/content/drive/MyDrive/crowfeather_412m_3e_natural/distill_data')
    p.add_argument('--budget-mb', type=int, default=8000)
    args = p.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)

    BUDGET_MATH = int(args.budget_mb * 0.30 * 1e6)
    BUDGET_LANG = int(args.budget_mb * 0.40 * 1e6)
    BUDGET_CODE = int(args.budget_mb * 0.30 * 1e6)

    safe_run('[MATH 1/3] NuminaMath-CoT', lambda: append_to(
        'numinamath', args.target_dir, stream_numinamath(), int(BUDGET_MATH * 0.40)))
    safe_run('[MATH 2/3] MetaMathQA', lambda: append_to(
        'metamathqa', args.target_dir, stream_metamathqa(), int(BUDGET_MATH * 0.30)))
    safe_run('[MATH 3/3] R1 math', lambda: append_to(
        'r1_math', args.target_dir, stream_r1_subset('math'), int(BUDGET_MATH * 0.30)))

    safe_run('[LANG 1/3] Sonnet 4.6', lambda: append_to(
        'sonnet', args.target_dir, stream_sonnet(), int(BUDGET_LANG * 0.55)))
    safe_run('[LANG 2/3] R1 science', lambda: append_to(
        'r1_science', args.target_dir, stream_r1_subset('science'), int(BUDGET_LANG * 0.30)))
    opus_path = f'{args.drive_cache}/opus_4_6.jsonl'
    safe_run(f'[LANG 3/3] Opus 4.6 (Drive)', lambda: append_to(
        'opus', args.target_dir, stream_drive_jsonl(opus_path), int(BUDGET_LANG * 0.15)))

    safe_run('[CODE 1/1] R1 code', lambda: append_to(
        'r1_code', args.target_dir, stream_r1_subset('code'), BUDGET_CODE))

    print('\n=== Combining per-domain files ===')
    DOMAINS = {
        'math': ['numinamath', 'metamathqa', 'r1_math'],
        'lang': ['sonnet', 'r1_science', 'opus'],
        'code': ['r1_code'],
    }
    for d, srcs in DOMAINS.items():
        out = os.path.join(args.target_dir, f'{d}.jsonl')
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f'  {d}: SKIP'); continue
        sz = 0
        with open(out, 'w', encoding='utf-8') as out_f:
            for s in srcs:
                p = os.path.join(args.target_dir, f'{s}.jsonl')
                if not os.path.exists(p): continue
                with open(p, 'r', encoding='utf-8') as in_f:
                    for line in in_f: out_f.write(line); sz += len(line)
        print(f'  {d}: {sz/1e6:.1f} MB combined')

    print('\n=== Final ===')
    total = 0
    for fn in sorted(os.listdir(args.target_dir)):
        if fn.endswith('.jsonl'):
            sz = os.path.getsize(os.path.join(args.target_dir, fn))
            total += sz
            print(f'  {fn:25s} {sz/1e6:8.2f} MB')
    print(f'  TOTAL: {total/1e6:.2f} MB')


if __name__ == '__main__':
    main()
