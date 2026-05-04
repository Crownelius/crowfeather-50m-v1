"""Synthetic arithmetic data generator for Sparrow-1M.

Emits per-digit-formatted arithmetic problems, one per line:

    1 2 3 + 4 5 6 = 5 7 9

The format is the SAME for training and eval — the model learns to predict
the digits after "=". At inference, prompt with "{a} {op} {b} = " and decode
greedily until newline.

Usage:
    # 200K 2-digit addition problems for iter 1
    python gen_arith.py --out E:/sparrow/iter1.txt --n 200000 --digits 2 --ops +

    # 500K problems mixing 1-4 digit addition
    python gen_arith.py --out E:/sparrow/iter3.txt --n 500000 --max-digits 4 --ops +

    # Mixed-operation curriculum across 1-3 digits
    python gen_arith.py --out E:/sparrow/iter6.txt --n 1500000 \\
        --max-digits 3 --ops + - "*"
"""
import argparse
import os
import random


def per_digit(n: int) -> str:
    """123 -> '1 2 3'. Negative numbers get '- 1 2 3'."""
    if n < 0:
        return '- ' + ' '.join(str(-n))
    return ' '.join(str(n))


def gen_problem(digits: int, op: str, rng: random.Random) -> str:
    """Generate one arithmetic problem string ending in '\\n'."""
    lo, hi = 10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)

    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    elif op == '/':
        # Division: ensure b != 0 and result is integer for exact-match-ability.
        # We sample b in 1..min(9, max_b), then a = b * q where q is a random
        # quotient that keeps the dividend within `digits` digit count.
        b = rng.randint(1, min(9, hi))
        q_max = max(1, hi // b)
        q = rng.randint(1, q_max)
        a = b * q
        result = q
    else:
        raise ValueError(f'unknown op: {op!r}')

    return f'{per_digit(a)} {op} {per_digit(b)} = {per_digit(result)}\n'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output text file (one problem per line)')
    p.add_argument('--n', type=int, default=200_000, help='number of problems')
    p.add_argument('--digits', type=int, default=None,
                   help='exact digit count for both operands; mutually exclusive with --max-digits')
    p.add_argument('--max-digits', type=int, default=None,
                   help='sample digit count uniformly from {1..max_digits}')
    p.add_argument('--ops', nargs='+', default=['+'],
                   choices=['+', '-', '*', '/'],
                   help='operators to mix uniformly (default: +)')
    p.add_argument('--seed', type=int, default=20260504)
    p.add_argument('--shuffle', action='store_true', default=True,
                   help='shuffle problems before writing (default on)')
    args = p.parse_args()

    if (args.digits is None) == (args.max_digits is None):
        p.error('exactly one of --digits or --max-digits must be set')

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    problems = []
    for _ in range(args.n):
        d = args.digits if args.digits else rng.randint(1, args.max_digits)
        op = rng.choice(args.ops)
        problems.append(gen_problem(d, op, rng))

    if args.shuffle:
        rng.shuffle(problems)

    n_written = 0
    n_bytes = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for line in problems:
            f.write(line)
            n_written += 1
            n_bytes += len(line)

    print(f'  wrote {n_written:,} problems to {args.out}')
    print(f'  {n_bytes/1e6:.1f} MB ({n_bytes:,} bytes)')
    print(f'  avg problem length: {n_bytes/max(n_written, 1):.1f} bytes')
    print(f'  ops mix: {args.ops}')
    print(f'  digits: {"variable 1.." + str(args.max_digits) if args.max_digits else f"exactly {args.digits}"}')

    # Print a few samples for sanity
    print('\n  samples:')
    for line in problems[:5]:
        print(f'    {line.rstrip()!r}')


if __name__ == '__main__':
    main()
