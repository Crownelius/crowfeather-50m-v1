"""Head-to-head eval: Sparrow-1M vs a 1B reference (default Llama-3.2-1B-Instruct).

For each problem in the held-out test set:
- Sparrow-1M: prompt with "{a} {op} {b} = ", greedy decode until newline,
  exact-match the expected answer string.
- Baseline 1B: 5-shot prompted with our format, greedy decode, exact-match.

Both models are scored on the same problems with the same prompt format.
The win condition is rigorous: same task, same hardware, same metric.

Default baseline: meta-llama/Llama-3.2-1B-Instruct (gated model — accept
license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct first
and ensure your HF_TOKEN has access). Open alternatives below if you don't
have access:
    HuggingFaceTB/SmolLM2-1.35B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct  (1.54B, but the closest open ~1B)

Usage:
    python eval_vs_1b.py \\
        --sparrow  E:/sparrow/iter1/trained/final \\
        --baseline meta-llama/Llama-3.2-1B-Instruct \\
        --task add --digits 2 --n 1000

    # Open-license baseline (no HF gate)
    python eval_vs_1b.py \\
        --sparrow  E:/sparrow/iter1/trained/final \\
        --baseline HuggingFaceTB/SmolLM2-1.35B-Instruct \\
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
    for _ in range(k):
        line = gen_problem(digits, op, rng).rstrip('\n')
        examples.append(line)
    return FEW_SHOT_TEMPLATE.format(
        examples='\n'.join(examples),
        prompt=prompt,
    )


def eval_baseline(model_id: str, problems: list, device: str,
                  k_shots: int, digits: int, op: str, max_new: int = 32):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f'  loading baseline {model_id} (downloads on first run)')
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
            # Take first non-empty line as the answer
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
        'n_correct': n_correct,
        'n_total': len(problems),
        'accuracy': accuracy,
        'elapsed_sec': elapsed,
        'tokens_per_sec': tps,
        'p50_latency_ms': 1000 * elapsed / len(problems),
        'sample_outputs': sample_outputs,
        'n_params': n_params,
    }


# ----- Driver ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sparrow', required=True, help='Sparrow-1M trained final/ dir')
    p.add_argument('--baseline', default='meta-llama/Llama-3.2-1B-Instruct',
                   help='HF model id for the 1B reference')
    p.add_argument('--task', default='add', choices=['add', 'sub', 'mul', 'div'])
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

    op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[args.task]
    problems = build_test_set(args.task, args.digits, args.n)

    print('\n=== Sparrow-1M eval ===')
    sparrow_results = eval_sparrow(args.sparrow, problems, device)

    baseline_results = None
    if not args.skip_baseline:
        print(f'\n=== Baseline eval ({args.baseline}) ===')
        baseline_results = eval_baseline(args.baseline, problems, device,
                                         args.k_shots, args.digits, op)

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
        size_ratio = b.get('n_params', 1) / 1.078e6
        print()
        if delta > 0:
            print(f'  Sparrow-1M WINS by {delta:.1f}pp  ({size_ratio:.0f}x smaller)')
        else:
            print(f'  Sparrow-1M loses by {-delta:.1f}pp  ({size_ratio:.0f}x smaller)')

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
