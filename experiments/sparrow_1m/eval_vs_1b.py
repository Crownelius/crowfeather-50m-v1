"""Head-to-head eval: Sparrow-1M vs a 1B reference.

For each problem in the held-out test set:
- Sparrow-1M: prompt with "{a} {op} {b} = ", greedy decode until newline,
  exact-match the expected answer string.
- Baseline 1B: 5-shot prompted with our format, greedy decode, exact-match.

Both models are scored on the same problems with the same prompt format.
The win condition is rigorous: same task, same prompt, same metric.

THREE BASELINE MODES (choose via --baseline-provider):

  1. openrouter (default if OPENROUTER_API_KEY is set)
       Cheapest path. Free tier model: 'meta-llama/llama-3.2-1b-instruct:free'
       (rate-limited ~20 req/min on free; 1000 problems = ~50 min).
       Paid same model: ~$0.03 total for 1000 problems with no rate limit.
       Setup:
         - sign up at openrouter.ai, get an API key
         - set OPENROUTER_API_KEY in your env
         - (free tier requires $1+ balance; topping up enables everything)

  2. local (HF transformers)
       Loads weights to your machine. Default model is gated; use:
         meta-llama/Llama-3.2-1B-Instruct       (gated, accept license first)
         HuggingFaceTB/SmolLM2-1.35B-Instruct   (open, no gate)
         Qwen/Qwen2.5-1.5B-Instruct             (open, slightly larger)

  3. anthropic (for completeness, e.g. compare Sparrow-1M vs Claude Haiku 4.5)

Usage:
    # OpenRouter free tier (recommended; auto-detected if key is set)
    python eval_vs_1b.py \\
        --sparrow  E:/sparrow/iter1/trained/final \\
        --baseline meta-llama/llama-3.2-1b-instruct:free \\
        --task add --digits 2 --n 1000

    # Local HF model (slow first time due to download)
    python eval_vs_1b.py \\
        --sparrow  E:/sparrow/iter1/trained/final \\
        --baseline HuggingFaceTB/SmolLM2-1.35B-Instruct \\
        --baseline-provider local \\
        --task add --digits 2 --n 1000
"""
import argparse
import json
import os
import random
import sys
import time

import torch

# Local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes_tok import encode, decode, EOS_ID  # noqa: E402
from gen_arith import gen_problem, per_digit  # noqa: E402


# ----- Test set generation --------------------------------------------------

def build_test_set(task: str, digits: int, n: int, seed: int = 12345):
    """Returns list of (prompt, expected_answer_str, problem_str)."""
    rng = random.Random(seed)
    problems = []
    if task == 'mixed':
        from gen_arith import gen_mixed
        for _ in range(n):
            # Default: 3 ops, +/-/* (no /), digits passed in
            line = gen_mixed(digits, ['+', '-', '*'], 3, rng)
            eq, answer_with_nl = line.split(' = ')
            answer = answer_with_nl.rstrip('\n')
            prompt = eq + ' = '
            problems.append((prompt, answer, line.rstrip('\n')))
        return problems
    if task == 'algebra':
        from gen_arith import gen_algebra
        for _ in range(n):
            line = gen_algebra(rng).rstrip('\n')
            # Format: "a x ± b = c x = solution"
            # Split at the LAST " x = " to separate equation from answer
            eq_part, answer = line.rsplit(' x = ', 1)
            prompt = eq_part + ' x = '
            problems.append((prompt, answer, line))
        return problems
    op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[task]
    for _ in range(n):
        line = gen_problem(digits, op, rng)
        # line is e.g. "1 2 3 + 4 5 6 = 5 7 9\n"
        eq, answer_with_nl = line.split(' = ')
        answer = answer_with_nl.rstrip('\n')
        prompt = eq + ' = '  # "1 2 3 + 4 5 6 = "
        problems.append((prompt, answer, line.rstrip('\n')))
    return problems


# ----- Sparrow eval ---------------------------------------------------------

def eval_sparrow(model_dir: str, problems: list, device: str, max_new: int = 32):
    from transformers import Qwen3ForCausalLM
    print(f'  loading Sparrow-1M from {model_dir}')
    model = Qwen3ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device); model.eval()

    n_correct = 0
    t_start = time.time()
    n_out_tokens = 0
    sample_outputs = []

    with torch.no_grad():
        for prompt, expected, _ in problems:
            ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            out = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=EOS_ID,
                use_cache=True,
            )
            generated = decode(out[0, ids.shape[1]:].tolist())
            # Trim at first newline if present
            generated = generated.split('\n')[0].strip()
            n_out_tokens += out.shape[1] - ids.shape[1]
            if generated == expected:
                n_correct += 1
            if len(sample_outputs) < 5:
                sample_outputs.append((prompt, expected, generated))

    elapsed = time.time() - t_start
    accuracy = n_correct / len(problems)
    tps = n_out_tokens / elapsed
    return {
        'name': 'Sparrow-1M',
        'model_dir': model_dir,
        'n_correct': n_correct,
        'n_total': len(problems),
        'accuracy': accuracy,
        'elapsed_sec': elapsed,
        'tokens_per_sec': tps,
        'p50_latency_ms': 1000 * elapsed / len(problems),
        'sample_outputs': sample_outputs,
    }


# ----- Baseline eval --------------------------------------------------------

FEW_SHOT_TEMPLATE = """The following are arithmetic problems written in per-digit format. Each digit is space-separated, and the answer comes after the equals sign.

Examples:
{examples}

Now solve this:
{prompt}"""


def build_few_shot_prompt(prompt: str, k: int, digits: int, op: str, seed: int = 9999) -> str:
    rng = random.Random(seed)
    examples = []
    if op == 'mixed':
        from gen_arith import gen_mixed
        for _ in range(k):
            examples.append(gen_mixed(digits, ['+', '-', '*'], 3, rng).rstrip('\n'))
    elif op == 'algebra':
        from gen_arith import gen_algebra
        for _ in range(k):
            examples.append(gen_algebra(rng).rstrip('\n'))
    else:
        for _ in range(k):
            examples.append(gen_problem(digits, op, rng).rstrip('\n'))
    return FEW_SHOT_TEMPLATE.format(
        examples='\n'.join(examples),
        prompt=prompt,
    )


def eval_baseline_local(model_id: str, problems: list, device: str,
                        k_shots: int, digits: int, op: str, max_new: int = 32):
    """Local HF transformers baseline (downloads weights on first run)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f'  loading baseline {model_id} via HF transformers (downloads on first run)')
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map=device,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  baseline loaded: {n_params/1e9:.2f}B params')

    n_correct = 0
    t_start = time.time()
    n_out_tokens = 0
    sample_outputs = []

    with torch.no_grad():
        for prompt, expected, _ in problems:
            full_prompt = build_few_shot_prompt(prompt, k_shots, digits, op)
            ids = tok(full_prompt, return_tensors='pt').input_ids.to(device)
            out = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
            generated_text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            first_line = generated_text.strip().split('\n')[0].strip()
            n_out_tokens += out.shape[1] - ids.shape[1]
            if first_line == expected:
                n_correct += 1
            if len(sample_outputs) < 5:
                sample_outputs.append((prompt, expected, first_line))

    elapsed = time.time() - t_start
    accuracy = n_correct / len(problems)
    tps = n_out_tokens / elapsed
    return {
        'name': model_id,
        'provider': 'local',
        'n_correct': n_correct,
        'n_total': len(problems),
        'accuracy': accuracy,
        'elapsed_sec': elapsed,
        'tokens_per_sec': tps,
        'p50_latency_ms': 1000 * elapsed / len(problems),
        'sample_outputs': sample_outputs,
        'n_params': n_params,
    }


def eval_baseline_openrouter(model_id: str, problems: list,
                             k_shots: int, digits: int, op: str,
                             max_new: int = 32, rps: float = 5.0,
                             api_key: str = None):
    """OpenRouter API baseline. Auto-throttles to `rps` requests/sec; retries
    with exponential backoff on 429 rate-limit responses.

    For free-tier models like 'meta-llama/llama-3.2-1b-instruct:free',
    keep rps low (1-3). For paid, push to 10+."""
    import requests
    api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY not set. '
                           'Get one at https://openrouter.ai/keys')

    print(f'  using OpenRouter for {model_id}  (rps={rps})')

    n_correct = 0
    n_out_tokens = 0
    n_in_tokens = 0
    sample_outputs = []
    failed = 0
    t_start = time.time()
    interval = 1.0 / max(rps, 0.01)
    last_call_t = 0.0

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/Crownelius/crowfeather-50m-v1',
        'X-Title': 'Sparrow-1M arithmetic eval',
    }

    for i, (prompt, expected, _) in enumerate(problems):
        full_prompt = build_few_shot_prompt(prompt, k_shots, digits, op)

        # Throttle
        wait = interval - (time.time() - last_call_t)
        if wait > 0:
            time.sleep(wait)

        # Retry loop with exponential backoff on 429s
        backoff = 1.0
        for attempt in range(5):
            last_call_t = time.time()
            try:
                resp = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json={
                        'model': model_id,
                        'messages': [{'role': 'user', 'content': full_prompt}],
                        'max_tokens': max_new,
                        'temperature': 0.0,
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    if attempt == 4:
                        print(f'    rate-limited 5x in a row; giving up on this row')
                        failed += 1
                        break
                    print(f'    [{i+1}/{len(problems)}] 429 rate-limited; backoff {backoff:.1f}s')
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if resp.status_code != 200:
                    print(f'    [{i+1}/{len(problems)}] HTTP {resp.status_code}: {resp.text[:200]}')
                    failed += 1
                    break

                data = resp.json()
                content = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                n_in_tokens += usage.get('prompt_tokens', 0)
                n_out_tokens += usage.get('completion_tokens', 0)

                first_line = content.strip().split('\n')[0].strip()
                if first_line == expected:
                    n_correct += 1
                if len(sample_outputs) < 5:
                    sample_outputs.append((prompt, expected, first_line))
                break  # success
            except (requests.RequestException, KeyError, ValueError) as e:
                if attempt == 4:
                    print(f'    [{i+1}/{len(problems)}] error: {e}; giving up')
                    failed += 1
                    break
                time.sleep(backoff)
                backoff *= 2

        # Progress log every 50
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(problems) - i - 1) / max(rate, 0.01)
            cur_acc = n_correct / max(i + 1 - failed, 1)
            print(f'    {i+1}/{len(problems)}  acc_so_far={100*cur_acc:.1f}%  '
                  f'rate={rate:.2f}/s  ETA={eta/60:.1f}m  failed={failed}')

    elapsed = time.time() - t_start
    n_scored = len(problems) - failed
    accuracy = n_correct / max(n_scored, 1)
    tps = n_out_tokens / elapsed if elapsed > 0 else 0

    return {
        'name': model_id,
        'provider': 'openrouter',
        'n_correct': n_correct,
        'n_total': len(problems),
        'n_scored': n_scored,
        'n_failed': failed,
        'accuracy': accuracy,
        'elapsed_sec': elapsed,
        'tokens_per_sec': tps,
        'p50_latency_ms': 1000 * elapsed / max(len(problems), 1),
        'sample_outputs': sample_outputs,
        'usage': {'input_tokens': n_in_tokens, 'output_tokens': n_out_tokens},
        'n_params': None,  # unknown from API
    }


def eval_baseline(model_id: str, problems: list, device: str,
                  k_shots: int, digits: int, op: str,
                  provider: str = 'auto', rps: float = 5.0, max_new: int = 32):
    """Dispatch to local or OpenRouter baseline."""
    if provider == 'auto':
        if os.environ.get('OPENROUTER_API_KEY') and ':free' in model_id or '/' in model_id:
            # Heuristic: if user has OpenRouter key set and model_id looks like
            # an OpenRouter slug (provider/model[:tag]), use OpenRouter.
            # User can override with --baseline-provider local.
            provider = 'openrouter' if os.environ.get('OPENROUTER_API_KEY') else 'local'
        else:
            provider = 'local'
    if provider == 'openrouter':
        return eval_baseline_openrouter(model_id, problems, k_shots, digits, op,
                                        max_new=max_new, rps=rps)
    return eval_baseline_local(model_id, problems, device, k_shots, digits, op,
                               max_new=max_new)


# ----- Driver ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sparrow', required=True, help='Sparrow-1M trained final/ dir')
    p.add_argument('--baseline', default='meta-llama/llama-3.2-1b-instruct:free',
                   help='model id for the 1B reference. OpenRouter slug like '
                        '"meta-llama/llama-3.2-1b-instruct:free" or local HF id like '
                        '"HuggingFaceTB/SmolLM2-1.35B-Instruct".')
    p.add_argument('--baseline-provider', default='auto',
                   choices=['auto', 'openrouter', 'local'],
                   help='how to call the baseline: openrouter API or load locally. '
                        '"auto" picks openrouter if OPENROUTER_API_KEY is set, else local.')
    p.add_argument('--rps', type=float, default=3.0,
                   help='OpenRouter rate limit (requests/sec). Free tier: keep <=3. '
                        'Paid: bump to 10+.')
    p.add_argument('--task', default='add', choices=['add', 'sub', 'mul', 'div', 'mixed', 'algebra'])
    p.add_argument('--digits', type=int, default=2)
    p.add_argument('--n', type=int, default=1000, help='test problems')
    p.add_argument('--k-shots', type=int, default=5, help='few-shot examples for baseline')
    p.add_argument('--device', default=None)
    p.add_argument('--report', default=None,
                   help='write JSON report (default: <sparrow>/eval_<task>_<digits>d.json)')
    p.add_argument('--skip-baseline', action='store_true',
                   help='only eval Sparrow (faster smoke test)')
    args = p.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'  device: {device}')
    print(f'  task: {args.task} ({args.digits}-digit)  n={args.n}')

    op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'mixed': 'mixed', 'algebra': 'algebra'}[args.task]
    problems = build_test_set(args.task, args.digits, args.n)

    print('\n=== Sparrow-1M eval ===')
    sparrow_results = eval_sparrow(args.sparrow, problems, device)

    baseline_results = None
    if not args.skip_baseline:
        print(f'\n=== Baseline eval ({args.baseline}) ===')
        baseline_results = eval_baseline(args.baseline, problems, device,
                                         args.k_shots, args.digits, op,
                                         provider=args.baseline_provider,
                                         rps=args.rps)

    # ----- Report
    print()
    print('=' * 78)
    print(f'HEAD-TO-HEAD: {args.task} ({args.digits}-digit)  n={args.n}')
    print('=' * 78)
    print(f'  {"Model":35s} {"Acc":>7s} {"tok/s":>10s} {"p50_latency":>12s}')
    print(f'  {"-"*35} {"-"*7} {"-"*10} {"-"*12}')
    s = sparrow_results
    print(f'  {s["name"]:35s} {100*s["accuracy"]:>6.1f}%  {s["tokens_per_sec"]:>9.0f} {s["p50_latency_ms"]:>10.1f}ms')
    if baseline_results:
        b = baseline_results
        print(f'  {b["name"]:35s} {100*b["accuracy"]:>6.1f}%  {b["tokens_per_sec"]:>9.0f} {b["p50_latency_ms"]:>10.1f}ms')

        delta = 100 * (s['accuracy'] - b['accuracy'])
        if b.get('n_params'):
            size_str = f'{b["n_params"]/1.078e6:.0f}x smaller'
        else:
            # Provider doesn't expose param count (OpenRouter); use the slug as a hint
            size_str = '~1000x smaller (1M vs ~1B)'
        print()
        if delta > 0:
            print(f'  Sparrow-1M WINS by {delta:.1f}pp  ({size_str})')
        else:
            print(f'  Sparrow-1M loses by {-delta:.1f}pp  ({size_str})')

        # OpenRouter usage stats if applicable
        if b.get('provider') == 'openrouter' and b.get('usage'):
            print(f'  OpenRouter usage: {b["usage"]["input_tokens"]:,} in + '
                  f'{b["usage"]["output_tokens"]:,} out tokens '
                  f'({b.get("n_failed", 0)} failed)')

    print()
    print('  Sparrow-1M sample outputs:')
    for prompt, expected, generated in sparrow_results['sample_outputs']:
        match = 'OK' if generated == expected else 'WRONG'
        print(f'    [{match:5s}] prompt={prompt!r}  exp={expected!r}  got={generated!r}')

    if baseline_results:
        print(f'\n  {args.baseline} sample outputs:')
        for prompt, expected, generated in baseline_results['sample_outputs']:
            match = 'OK' if generated == expected else 'WRONG'
            print(f'    [{match:5s}] prompt={prompt!r}  exp={expected!r}  got={generated!r}')

    # JSON report
    report_path = args.report or os.path.join(args.sparrow, f'eval_{args.task}_{args.digits}d.json')
    report = {
        'task': args.task, 'digits': args.digits, 'n': args.n,
        'sparrow': sparrow_results,
        'baseline': baseline_results,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=str)
    print(f'\n  report saved: {report_path}')


if __name__ == '__main__':
    main()
