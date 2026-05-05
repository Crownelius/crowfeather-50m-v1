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


def gen_algebra(rng: random.Random, digits: int = 1) -> str:
    """Generate one linear 1-variable algebra problem.

    Format: "a x + b = c x = solution\\n"

    `digits` controls the magnitude of a, b, and x (the solution). c = a*x + b
    is emitted per-digit; a and b are also per-digit when multi-digit (so the
    model sees the same per-digit format end-to-end).

    digits=1: a in [1,9],   b in [-9,9],   x in [-9,9]\\{0}
    digits=2: a in [10,99], b in [-99,99], x in [-99,99]\\{0}
    digits=3: a in [100,999], b in [-999,999], x in [-99,99]\\{0}  (kept x small to bound c)

    For digits >= 2, a and b are emitted per-digit too: "1 5 x + 2 3 = ...".

    Examples (digits=1):
        "3 x + 4 = 1 0 x = 2"     (3*2 + 4 = 10)
        "2 x + 1 = - 9 x = - 5"
    Examples (digits=2):
        "1 5 x + 2 3 = 1 7 3 x = 1 0"     (15*10 + 23 = 173)
    """
    if digits == 1:
        x_lo, x_hi = -9, 9
        a_lo, a_hi = 1, 9
        b_lo, b_hi = -9, 9
    elif digits == 2:
        x_lo, x_hi = -99, 99
        a_lo, a_hi = 10, 99
        b_lo, b_hi = -99, 99
    elif digits == 3:
        # Cap x to [-99, 99] to keep c bounded; a in [100, 999]
        x_lo, x_hi = -99, 99
        a_lo, a_hi = 100, 999
        b_lo, b_hi = -999, 999
    else:
        raise ValueError(f'algebra: unsupported digits={digits} (try 1, 2, or 3)')

    x = rng.randint(x_lo, x_hi)
    while x == 0:
        x = rng.randint(x_lo, x_hi)
    a = rng.randint(a_lo, a_hi)
    b = rng.randint(b_lo, b_hi)
    c = a * x + b

    a_str = per_digit(a)
    if b >= 0:
        eq = f'{a_str} x + {per_digit(b)} = {per_digit(c)}'
    else:
        eq = f'{a_str} x - {per_digit(-b)} = {per_digit(c)}'
    return f'{eq} x = {per_digit(x)}\n'


def gen_distribute(rng: random.Random, digits: int = 1) -> str:
    """Generate one distributive expansion problem.

    Format: "a ( x + b ) = a x + ab\\n"  (or with - for negative b/ab)

    The model must (1) copy `a` to the RHS coefficient and (2) compute a*b
    as the constant. Both signs handled per-digit.

    digits=1: a, b in [-9, 9]\\{0 for a}
    digits=2: a, b in [-99, 99]\\{0 for a}
    """
    if digits == 1:
        lo, hi = -9, 9
    elif digits == 2:
        lo, hi = -99, 99
    else:
        raise ValueError(f'distribute: unsupported digits={digits} (try 1 or 2)')

    a = rng.randint(lo, hi)
    while a == 0:
        a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    ab = a * b

    a_str = per_digit(a)
    if b >= 0:
        prompt = f'{a_str} ( x + {per_digit(b)} )'
    else:
        prompt = f'{a_str} ( x - {per_digit(-b)} )'

    if ab >= 0:
        answer = f'{a_str} x + {per_digit(ab)}'
    else:
        answer = f'{a_str} x - {per_digit(-ab)}'

    return f'{prompt} = {answer}\n'


def gen_mixed(digits: int, ops_list: list, n_ops: int, rng: random.Random) -> str:
    """Generate one left-to-right mixed expression.

    Format: 'a op b op c op d = result' where the expression is evaluated
    LEFT-TO-RIGHT, ignoring standard operator precedence (this keeps the
    model's learning target simple — it doesn't need to learn order-of-ops).

    Division is skipped in mixed mode (it requires divisibility constraints
    that compose poorly with chained ops). Iter 8 covers division separately.
    """
    if '/' in ops_list:
        ops_list = [o for o in ops_list if o != '/']
    if not ops_list:
        raise ValueError('mixed mode needs at least one of +, -, *')

    lo = 10 ** (digits - 1) if digits > 1 else 0
    hi = 10 ** digits - 1
    operands = [rng.randint(lo, hi) for _ in range(n_ops + 1)]
    chosen_ops = [rng.choice(ops_list) for _ in range(n_ops)]

    # Left-to-right evaluation
    result = operands[0]
    for op, b in zip(chosen_ops, operands[1:]):
        if op == '+':
            result = result + b
        elif op == '-':
            result = result - b
        elif op == '*':
            result = result * b

    # Format
    parts = [per_digit(operands[0])]
    for op, b in zip(chosen_ops, operands[1:]):
        parts.append(op)
        parts.append(per_digit(b))
    parts.append('=')
    parts.append(per_digit(result))
    return ' '.join(parts) + '\n'


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
    p.add_argument('--mixed', action='store_true',
                   help='generate mixed left-to-right expressions (a op1 b op2 c op3 d = result)')
    p.add_argument('--n-ops', type=int, default=3,
                   help='number of operations per mixed expression (default 3 -> 4 operands)')
    p.add_argument('--algebra', action='store_true',
                   help='generate linear 1-variable algebra problems: a x + b = c -> x = solution')
    p.add_argument('--distribute', action='store_true',
                   help='generate distributive expansion: a ( x + b ) = a x + ab')
    p.add_argument('--seed', type=int, default=20260504)
    p.add_argument('--shuffle', action='store_true', default=True,
                   help='shuffle problems before writing (default on)')
    args = p.parse_args()

    if not args.algebra and not args.distribute and (args.digits is None) == (args.max_digits is None):
        p.error('exactly one of --digits or --max-digits must be set (or use --algebra/--distribute)')
    if (args.algebra or args.distribute) and args.digits is None:
        args.digits = 1

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    problems = []
    for _ in range(args.n):
        if args.algebra:
            problems.append(gen_algebra(rng, args.digits))
            continue
        if args.distribute:
            problems.append(gen_distribute(rng, args.digits))
            continue
        d = args.digits if args.digits else rng.randint(1, args.max_digits)
        if args.mixed:
            problems.append(gen_mixed(d, args.ops, args.n_ops, rng))
        else:
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
