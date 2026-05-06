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


# ===== Phase D: tool-augmented (calc-tag) data generation =====
# Sentinel bytes — never appear in printable training data:
CALC_OPEN  = '\x01'
CALC_CLOSE = '\x02'
# Phase D extension 2: sympy-augmented (sym-tag) data generation.
# Distinct sentinels so the inference dispatcher can pick the right wrapper.
SYM_OPEN   = '\x03'
SYM_CLOSE  = '\x04'

# ===== Phase E Phase 1: Position Coupling (Cho et al. 2024, arxiv:2405.20671) =====
# Per the deep-read at D:/FANT_TRAINING_D_Drive/sparrow_deep_read_position_coupling.md:
# - Result emitted LSB-first (REVERSED relative to standard Sparrow per-digit format).
# - Each digit byte gets a position-ID equal to its decimal-significance index,
#   shared across operands and result. LSB -> start, MSB -> start + n_digits - 1.
# - Operator + '=' get position start + max(n_a, n_b).
# - Random offset start ~ U[1, max_pos - len(result)] per-sample at training.
# - Space bytes inherit the position-ID of their immediate LEFT digit neighbor.
# - max_pos defaults to 200 (fits up to ~99-digit results with random offset headroom).
PC_MAX_POS = 200


def _parse_pc_line(line: str) -> dict:
    """Parse a Position-Coupling-formatted line of the form:
        '<a per-digit> <op> <b per-digit> = <result REVERSED per-digit>'
    Returns a dict with byte spans needed to build position IDs.
    """
    # Find ' = ' boundary. LHS = 'a op b', RHS = result-LSB-first.
    if ' = ' not in line:
        raise ValueError(f"PC line missing ' = ': {line!r}")
    lhs, rhs = line.split(' = ', 1)
    # Strip trailing newline from rhs
    rhs = rhs.rstrip('\n')
    # LHS structure: 'd1 d2 d3 op d4 d5 d6'  (single-byte op among + - * /)
    parts = lhs.split(' ')
    # Find operator index — first token in {'+','-','*','/'}
    op_idx = None
    for i, t in enumerate(parts):
        if t in ('+', '-', '*', '/'):
            op_idx = i
            break
    if op_idx is None:
        raise ValueError(f"PC line missing operator: {line!r}")
    a_tokens = parts[:op_idx]
    op = parts[op_idx]
    b_tokens = parts[op_idx + 1:]
    r_tokens = rhs.split(' ')
    return {'a': a_tokens, 'op': op, 'b': b_tokens, 'r': r_tokens}


def compute_pc_position_ids(line: str, start: int = 1) -> list:
    """Given a PC-formatted line + a start offset, return a list of position
    IDs of the same byte-length as the line (excluding any trailing newline).

    Rule (matches Cho et al. 2024 + the deep-read in
    D:/FANT_TRAINING_D_Drive/sparrow_deep_read_position_coupling.md):
    - Each digit byte: position_id = start + (its decimal-significance index)
      where significance is counted FROM THE LSB of its number (LSB = 0).
      Result is in LSB-first order, so position_ids of result digits are
      [start, start+1, start+2, ...] reading left-to-right.
    - Operator and '=' bytes: position_id = start + max(len(a), len(b)).
    - Space bytes: inherit position_id of their immediate LEFT digit neighbor
      (or the left non-space byte). For the leading space inside the LHS
      between digits this is well-defined.
    - For the very-first byte (which is a digit) there's no left neighbor;
      that's fine because the first byte IS a digit (no leading space).
    """
    line = line.rstrip('\n')
    parts = _parse_pc_line(line)
    n_a = len(parts['a'])
    n_b = len(parts['b'])
    n_r = len(parts['r'])
    op_pos = start + max(n_a, n_b)

    # Walk the line byte by byte, deciding position-id by which segment we're in.
    pos_ids = []
    cursor = 0   # byte index into line

    def assign_token(token: str, base_position_for_lsb: int, is_reversed: bool):
        """Append position IDs for the bytes spanned by `token` (a digit)
        treating `base_position_for_lsb` as the LSB anchor.

        `is_reversed = False` (operands): token's left-most byte is MSB,
            so its position_id = base + (n - 1 - i) for byte index i in token.
        `is_reversed = True` (result, already LSB-first in text): token's
            left-most byte is LSB, position_id = base + i.

        For Sparrow's per-digit format each digit token is ONE byte long
        ('5', '7' etc.), so n=1 and the formula simplifies — but we keep
        the general form for safety.
        """
        # Sparrow tokens are single chars in '0'-'9' or a leading '-'. Per-digit
        # format uses spaces between digits, so a "token" here is a single byte.
        if len(token) != 1:
            # Negative numbers in our format come as separate '-' tokens, not
            # multi-byte tokens — so this branch shouldn't fire for arithmetic.
            for j, _ in enumerate(token):
                pos_ids.append(op_pos)  # treat unknown as operator-adjacent
            return
        pos_ids.append(base_position_for_lsb)

    # Walk segments: a_tokens [space] op [space] b_tokens [space] = [space] r_tokens
    # Each token is one byte wide; spaces are single bytes between tokens.
    # Operand A — MSB-first: i=0 -> position start + n_a - 1, i=n-1 -> start.
    for i, tok in enumerate(parts['a']):
        if i > 0:
            # space byte before this digit — inherit LEFT digit's position
            pos_ids.append(pos_ids[-1])
        # this digit's position
        pos_ids.append(start + (n_a - 1 - i))
    # space before operator
    pos_ids.append(pos_ids[-1])
    # operator byte
    pos_ids.append(op_pos)
    # space before B
    pos_ids.append(op_pos)
    # Operand B — MSB-first
    for i, tok in enumerate(parts['b']):
        if i > 0:
            pos_ids.append(pos_ids[-1])
        pos_ids.append(start + (n_b - 1 - i))
    # space before '='
    pos_ids.append(pos_ids[-1])
    # '=' byte
    pos_ids.append(op_pos)
    # space before result
    pos_ids.append(op_pos)
    # Result tokens — already LSB-first in text. i=0 -> position start, i=n-1 -> start + n_r - 1.
    for i, tok in enumerate(parts['r']):
        if i > 0:
            pos_ids.append(pos_ids[-1])
        pos_ids.append(start + i)

    # Sanity: pos_ids length should match raw line length in bytes.
    assert len(pos_ids) == len(line), (
        f"position-ids length mismatch: line is {len(line)} bytes but produced "
        f"{len(pos_ids)} ids. Line: {line!r}"
    )
    return pos_ids


def gen_problem_pc(digits: int, op: str, rng: random.Random,
                   max_pos: int = PC_MAX_POS) -> str:
    """Position-Coupling-formatted arithmetic problem. Same operand format as
    `gen_problem` but the result is emitted LSB-first (reversed). Position-IDs
    are NOT stored alongside — they are deterministic from the line content
    and a per-sample random `start`, computed by `compute_pc_position_ids`
    at training time.

    Text format (single line):
        '<a per-digit MSB-first> <op> <b per-digit MSB-first> = <result LSB-first per-digit>\\n'
    Example for 526 * 850 = 447100:
        '5 2 6 * 8 5 0 = 0 0 1 7 4 4'
    """
    lo = 10 ** (digits - 1) if digits > 1 else 0
    hi = 10 ** digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)

    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    elif op == '/':
        b = rng.randint(1, min(9, hi))
        q_max = max(1, hi // b)
        q = rng.randint(1, q_max)
        a = b * q
        result = q
    else:
        raise ValueError(f'unknown op: {op!r}')

    a_str = per_digit(a)
    b_str = per_digit(b)
    # Result LSB-first: take the absolute value's digits, reverse them.
    if result < 0:
        r_digits = '- ' + ' '.join(str(-result)[::-1])
    else:
        r_digits = ' '.join(str(result)[::-1])
    return f'{a_str} {op} {b_str} = {r_digits}\n'


def gen_problem_calc(digits: int, op: str, rng: random.Random) -> str:
    """Calc-tag-wrapped variant of gen_problem.

    Output format: '<lhs per-digit> = <\\x01><lhs compact><\\x02>\\n'
    where the model emits ONLY the calc tag (no answer); the inference wrapper
    runs python eval on the tag content, formats the result per-digit, and
    that's the answer for scoring.

    Example: '1 2 3 * 4 5 6 = \\x01123 * 456\\x02\\n'
    The inference wrapper extracts '123 * 456', evaluates to 56088, formats
    as '5 6 0 8 8' for comparison against the expected per-digit answer.
    """
    lo, hi = 10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)

    if op == '/':
        b = rng.randint(1, min(9, hi))
        q_max = max(1, hi // b)
        q = rng.randint(1, q_max)
        a = b * q
    if op not in {'+', '-', '*', '/'}:
        raise ValueError(f'unknown op: {op!r}')

    # Use Python operators inside the calc tag — eval() handles them.
    # Note: '/' for integer division would give float; use '//' for exact int.
    py_op = '//' if op == '/' else op
    return f'{per_digit(a)} {op} {per_digit(b)} = {CALC_OPEN}{a} {py_op} {b}{CALC_CLOSE}\n'


def gen_factor_sym(rng: random.Random, digits: int = 1) -> str:
    """Phase D ext-2: sympy-augmented quadratic factoring.

    Input format (per-digit, matches iter15 prompt):
        'x ^ 2 [+/-] |p| x [+/-] |q|'
    Output format (sym-tag, machine-evaluable by sympy.factor):
        '<\\x03>factor(x**2 [+/-] |p|*x [+/-] |q|)<\\x04>'

    The roots r1 < r2 are sampled from {-r_hi..r_hi} excluding equal-magnitude
    cases that produce a 0 linear term (skip r1 = -r2).
    """
    if digits == 1:
        r_lo, r_hi = -9, 9
    elif digits == 2:
        r_lo, r_hi = -49, 49
    else:
        raise ValueError(f'factor_sym: unsupported digits={digits}')

    while True:
        r1 = rng.randint(r_lo, r_hi)
        r2 = rng.randint(r_lo, r_hi)
        if r1 == r2 or r1 == -r2:
            continue
        if r1 > r2:
            r1, r2 = r2, r1
        break

    p = -(r1 + r2)
    q = r1 * r2

    # Per-digit prompt (LHS)
    p_sign = '+' if p >= 0 else '-'
    q_sign = '+' if q >= 0 else '-'
    p_per = per_digit(abs(p))
    q_per = per_digit(abs(q))
    prompt = f'x ^ 2 {p_sign} {p_per} x {q_sign} {q_per}'

    # Sym-tag inner (sympy-syntax)
    p_py = f'+ {abs(p)}*x' if p >= 0 else f'- {abs(p)}*x'
    q_py = f'+ {abs(q)}'   if q >= 0 else f'- {abs(q)}'
    sym_inner = f'factor(x**2 {p_py} {q_py})'

    return f'{prompt} = {SYM_OPEN}{sym_inner}{SYM_CLOSE}\n'


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


def gen_polymul(rng: random.Random, digits: int = 1) -> str:
    """Polynomial multiplication: (x + a)(x + b) = x^2 + (a+b)x + ab.

    Two simultaneous computations: sum (coefficient of x) and product (constant
    term). Introduces 'x ^ 2' token sequence.

    digits=1: a, b in [-9, 9]\\{0}.  s = a+b can be -18..18. p = a*b can be -81..81.
    digits=2: a, b in [-99, 99]\\{0}.

    Skip problems where s == 0 (i.e., b == -a) to keep format uniform —
    no special-case "x^2 + 0 x + p".

    Example: "( x + 3 ) ( x - 5 ) = x ^ 2 - 2 x - 1 5"   (3 + (-5) = -2, 3 * -5 = -15)
    """
    if digits == 1:
        lo, hi = -9, 9
    elif digits == 2:
        lo, hi = -99, 99
    else:
        raise ValueError(f'polymul: unsupported digits={digits} (try 1 or 2)')

    while True:
        a = rng.randint(lo, hi)
        while a == 0:
            a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        while b == 0:
            b = rng.randint(lo, hi)
        s = a + b
        if s != 0:
            break  # avoid the s=0 edge case

    p = a * b

    a_sign = '+' if a >= 0 else '-'
    b_sign = '+' if b >= 0 else '-'
    s_sign = '+' if s >= 0 else '-'
    p_sign = '+' if p >= 0 else '-'

    prompt = f'( x {a_sign} {per_digit(abs(a))} ) ( x {b_sign} {per_digit(abs(b))} )'
    answer = f'x ^ 2 {s_sign} {per_digit(abs(s))} x {p_sign} {per_digit(abs(p))}'
    return f'{prompt} = {answer}\n'


def gen_factor(rng: random.Random, digits: int = 1) -> str:
    """Quadratic factoring: x^2 + p x + q = (x + a)(x + b) where a+b=p, a*b=q.

    Reverse of gen_polymul. Tests whether the model can SEARCH for factors.
    For uniqueness of target output, the factors are canonicalized so a <= b.

    Skip p=0 cases (i.e. b == -a, factoring as (x+a)(x-a) = x^2 - a^2 with no x term).
    """
    if digits == 1:
        lo, hi = -9, 9
    elif digits == 2:
        # Match the range used by gen_factor_sym(digits=2) so iter36's
        # eval test set aligns with iter36's training distribution.
        lo, hi = -49, 49
    else:
        raise ValueError(f'factor: unsupported digits={digits}')

    while True:
        a = rng.randint(lo, hi)
        while a == 0:
            a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        while b == 0:
            b = rng.randint(lo, hi)
        if a + b != 0:
            break

    # Canonicalize: a <= b for deterministic target
    if a > b:
        a, b = b, a

    p = a + b
    q = a * b

    p_sign = '+' if p >= 0 else '-'
    q_sign = '+' if q >= 0 else '-'
    a_sign = '+' if a >= 0 else '-'
    b_sign = '+' if b >= 0 else '-'

    prompt = f'x ^ 2 {p_sign} {per_digit(abs(p))} x {q_sign} {per_digit(abs(q))}'
    answer = f'( x {a_sign} {per_digit(abs(a))} ) ( x {b_sign} {per_digit(abs(b))} )'
    return f'{prompt} = {answer}\n'


def gen_differentiate(rng: random.Random, digits: int = 1) -> str:
    """Quadratic differentiation: d/dx[a x^2 + b x + c] = 2a x + b.

    First calculus task. Tests:
    - Recognize d/dx[...] notation
    - Compute 2a (multiply leading coefficient by 2)
    - Drop the constant c
    - Preserve b and its sign

    Restricted to a > 0 for format simplicity (so 2a is also > 0).
    b and c can be in [-9, 9] for digits=1.
    """
    if digits == 1:
        a_lo, a_hi = 1, 9
        bc_lo, bc_hi = -9, 9
    elif digits == 2:
        a_lo, a_hi = 1, 99
        bc_lo, bc_hi = -99, 99
    else:
        raise ValueError(f'differentiate: unsupported digits={digits}')

    a = rng.randint(a_lo, a_hi)
    b = rng.randint(bc_lo, bc_hi)
    c = rng.randint(bc_lo, bc_hi)
    da = 2 * a

    a_str = per_digit(a)
    da_str = per_digit(da)
    b_sign = '+' if b >= 0 else '-'
    b_str = per_digit(abs(b))
    c_sign = '+' if c >= 0 else '-'
    c_str = per_digit(abs(c))

    prompt = f'd / d x [ {a_str} x ^ 2 {b_sign} {b_str} x {c_sign} {c_str} ]'
    answer = f'{da_str} x {b_sign} {b_str}'
    return f'{prompt} = {answer}\n'


def _is_prime(n: int) -> bool:
    """Trial division primality test. Plenty fast for n < 10^4."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def gen_collatz_step(rng: random.Random, digits: int = 1) -> str:
    """One step of the Collatz / 3n+1 sequence: 3n+1 if odd, n/2 if even.

    Format: 'c o l [ n ] = next'.
    Conjecture (Collatz, 1937): every positive n eventually reaches 1.
    """
    if digits == 1:
        n_lo, n_hi = 1, 9
    elif digits == 2:
        n_lo, n_hi = 10, 99
    elif digits == 3:
        n_lo, n_hi = 100, 999
    else:
        raise ValueError(f'collatz_step: unsupported digits={digits}')

    n = rng.randint(n_lo, n_hi)
    next_n = (3 * n + 1) if (n % 2) else (n // 2)

    prompt = f'c o l [ {per_digit(n)} ]'
    answer = per_digit(next_n)
    return f'{prompt} = {answer}\n'


def gen_collatz_time(rng: random.Random, digits: int = 3) -> str:
    """Number of Collatz steps until n reaches 1 (the 'total stopping time').

    Format: 'c t l [ n ] = steps'.
    For n in [1, 999] all trajectories terminate (verified far beyond);
    max steps for n <= 999 is 178 (at n = 871).
    """
    if digits == 1:
        n_lo, n_hi = 1, 9
    elif digits == 2:
        n_lo, n_hi = 10, 99
    elif digits == 3:
        n_lo, n_hi = 100, 999
    else:
        raise ValueError(f'collatz_time: unsupported digits={digits}')

    n = rng.randint(n_lo, n_hi)
    steps = 0
    cur = n
    while cur != 1:
        cur = (3 * cur + 1) if (cur % 2) else (cur // 2)
        steps += 1

    prompt = f'c t l [ {per_digit(n)} ]'
    answer = per_digit(steps)
    return f'{prompt} = {answer}\n'


def gen_goldbach(rng: random.Random, digits: int = 3) -> str:
    """Goldbach decomposition: even n = p + q with p <= q both prime.

    Outputs the CANONICAL pair: smallest p such that (p, n-p) are both prime.

    Format: 'g d c [ n ] = p + q'.
    Conjecture (Goldbach, 1742): every even n >= 4 is the sum of two primes.
    Verified up to ~4 * 10^18; for n <= 998 a decomposition always exists.
    """
    if digits == 2:
        n_lo, n_hi = 4, 98
    elif digits == 3:
        n_lo, n_hi = 100, 998
    else:
        raise ValueError(f'goldbach: unsupported digits={digits} (use 2 or 3)')

    # Resample odd n; we only emit even decompositions.
    for _ in range(1000):
        n = rng.randint(n_lo, n_hi)
        if n % 2 != 0:
            continue
        for p in range(2, n // 2 + 1):
            if _is_prime(p) and _is_prime(n - p):
                q = n - p
                prompt = f'g d c [ {per_digit(n)} ]'
                answer = f'{per_digit(p)} + {per_digit(q)}'
                return f'{prompt} = {answer}\n'
    raise RuntimeError('gen_goldbach: failed to sample a decomposable n in 1000 tries')


def gen_fermat_little(rng: random.Random, digits: int = 2) -> str:
    """Fermat's little theorem checker: compute a^(p-1) mod p.

    Format: 'f l t [ a ; p ] = result'.
    By Fermat (1640): if p is prime and gcd(a, p) = 1 then a^(p-1) = 1 (mod p).
    Sampling p uniformly from {p_lo..p_hi}, a uniformly from {2..p-1}.
    p may be composite; in that case result usually != 1, occasionally 1
    (Carmichael / pseudoprime cases). Either way the model learns modular
    exponentiation; the conjecture-flavored framing is the eval point.
    """
    if digits == 1:
        p_lo, p_hi = 3, 9
    elif digits == 2:
        p_lo, p_hi = 11, 99
    elif digits == 3:
        p_lo, p_hi = 100, 999
    else:
        raise ValueError(f'fermat_little: unsupported digits={digits}')

    p = rng.randint(p_lo, p_hi)
    a = rng.randint(2, p - 1)
    result = pow(a, p - 1, p)  # builtin fast modular exponentiation

    prompt = f'f l t [ {per_digit(a)} ; {per_digit(p)} ]'
    answer = per_digit(result)
    return f'{prompt} = {answer}\n'


# ===== Phase C: cross-discipline generators (number theory, combinatorics, =====
# ===== linear algebra, geometry, probability, real analysis, complex)        =====


def gen_gcd(rng: random.Random, digits: int = 2) -> str:
    """Greatest common divisor: gcd[a; b] = result.
    a, b independent in the digit range."""
    if digits == 1: lo, hi = 1, 9
    elif digits == 2: lo, hi = 1, 99
    elif digits == 3: lo, hi = 1, 999
    else: raise ValueError(f'gcd: unsupported digits={digits}')
    a = rng.randint(lo, hi); b = rng.randint(lo, hi)
    import math
    g = math.gcd(a, b)
    return f'g c d [ {per_digit(a)} ; {per_digit(b)} ] = {per_digit(g)}\n'


def gen_lcm(rng: random.Random, digits: int = 2) -> str:
    """Least common multiple: lcm[a; b] = result. Restricted so output fits within
    digits+1 — ensures a*b/gcd doesn't overflow Sparrow's max length."""
    if digits == 1: lo, hi = 1, 9
    elif digits == 2: lo, hi = 1, 99
    else: raise ValueError(f'lcm: unsupported digits={digits} (use 1 or 2)')
    import math
    while True:
        a = rng.randint(lo, hi); b = rng.randint(lo, hi)
        l = a * b // math.gcd(a, b)
        if l <= 99999:
            return f'l c m [ {per_digit(a)} ; {per_digit(b)} ] = {per_digit(l)}\n'


def gen_totient(rng: random.Random, digits: int = 2) -> str:
    """Euler's totient phi(n): count of k in [1,n] with gcd(k,n)=1.
    Format: 'p h i [ n ] = result'."""
    if digits == 1: lo, hi = 2, 9
    elif digits == 2: lo, hi = 2, 99
    elif digits == 3: lo, hi = 2, 999
    else: raise ValueError(f'totient: unsupported digits={digits}')
    import math
    n = rng.randint(lo, hi)
    phi = sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1)
    return f'p h i [ {per_digit(n)} ] = {per_digit(phi)}\n'


def gen_modinv(rng: random.Random, digits: int = 2) -> str:
    """Modular multiplicative inverse: inv[a; m] = a^-1 mod m, where gcd(a,m)=1.
    Resamples until coprime. Format: 'i n v [ a ; m ] = result'."""
    if digits == 1: m_lo, m_hi = 3, 9
    elif digits == 2: m_lo, m_hi = 3, 99
    else: raise ValueError(f'modinv: unsupported digits={digits}')
    import math
    for _ in range(1000):
        m = rng.randint(m_lo, m_hi)
        a = rng.randint(1, m - 1)
        if math.gcd(a, m) != 1:
            continue
        inv = pow(a, -1, m)
        return f'i n v [ {per_digit(a)} ; {per_digit(m)} ] = {per_digit(inv)}\n'
    raise RuntimeError('modinv: failed to sample coprime pair in 1000 tries')


def gen_choose(rng: random.Random, digits: int = 2) -> str:
    """Binomial coefficient nCk. Restricted to result <= 99999 to fit Sparrow's
    output budget. Format: 'c h s [ n ; k ] = result'."""
    if digits == 1: n_lo, n_hi = 2, 9
    elif digits == 2: n_lo, n_hi = 2, 20  # 20C10 = 184,756 already; limit harder
    else: raise ValueError(f'choose: unsupported digits={digits}')
    import math
    for _ in range(1000):
        n = rng.randint(n_lo, n_hi)
        k = rng.randint(0, n)
        c = math.comb(n, k)
        if c <= 99999:
            return f'c h s [ {per_digit(n)} ; {per_digit(k)} ] = {per_digit(c)}\n'
    raise RuntimeError('choose: failed to sample bounded n,k in 1000 tries')


def gen_fib(rng: random.Random, digits: int = 1) -> str:
    """nth Fibonacci number. F(0)=0, F(1)=1. Restricted to n<=20 (F(20)=6765).
    Format: 'f i b [ n ] = result'."""
    if digits == 1: n_lo, n_hi = 0, 9
    elif digits == 2: n_lo, n_hi = 0, 20
    else: raise ValueError(f'fib: unsupported digits={digits}')
    n = rng.randint(n_lo, n_hi)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return f'f i b [ {per_digit(n)} ] = {per_digit(a)}\n'


def gen_det2(rng: random.Random, digits: int = 1) -> str:
    """Determinant of 2x2 integer matrix [[a,b],[c,d]] = ad - bc.
    Format: 'd e t [ a ; b ; c ; d ] = result'."""
    if digits == 1: lo, hi = -9, 9
    elif digits == 2: lo, hi = -99, 99
    else: raise ValueError(f'det2: unsupported digits={digits}')
    a = rng.randint(lo, hi); b = rng.randint(lo, hi)
    c = rng.randint(lo, hi); d = rng.randint(lo, hi)
    det = a * d - b * c
    return f'd e t [ {per_digit(a)} ; {per_digit(b)} ; {per_digit(c)} ; {per_digit(d)} ] = {per_digit(det)}\n'


def gen_matmul2(rng: random.Random, digits: int = 1) -> str:
    """2x2 matrix multiplication. Output 4 entries as 'e ; f ; g ; h'.
    Format: 'm m [ a;b;c;d * e;f;g;h ] = e2;f2;g2;h2' simplified to use commas."""
    if digits == 1: lo, hi = -9, 9
    else: raise ValueError(f'matmul2: unsupported digits={digits} (use 1)')
    a, b, c, d = [rng.randint(lo, hi) for _ in range(4)]
    e, f, g, h = [rng.randint(lo, hi) for _ in range(4)]
    r1, r2 = a * e + b * g, a * f + b * h
    r3, r4 = c * e + d * g, c * f + d * h
    parts = [per_digit(x) for x in [a, b, c, d, e, f, g, h]]
    out = ' ; '.join(per_digit(x) for x in [r1, r2, r3, r4])
    inp = f'{parts[0]} ; {parts[1]} ; {parts[2]} ; {parts[3]} * {parts[4]} ; {parts[5]} ; {parts[6]} ; {parts[7]}'
    return f'm m [ {inp} ] = {out}\n'


def gen_dot2(rng: random.Random, digits: int = 1) -> str:
    """Vector dot product (a,b) . (c,d) = ac + bd.
    Format: 'd o t [ a ; b ; c ; d ] = result'."""
    if digits == 1: lo, hi = -9, 9
    elif digits == 2: lo, hi = -99, 99
    else: raise ValueError(f'dot2: unsupported digits={digits}')
    a, b, c, d = [rng.randint(lo, hi) for _ in range(4)]
    r = a * c + b * d
    return f'd o t [ {per_digit(a)} ; {per_digit(b)} ; {per_digit(c)} ; {per_digit(d)} ] = {per_digit(r)}\n'


def gen_distance(rng: random.Random, digits: int = 1) -> str:
    """Squared distance between (x1,y1) and (x2,y2): output (x2-x1)^2 + (y2-y1)^2.
    Squared to keep answer integer. Format: 'd s q [ x1 ; y1 ; x2 ; y2 ] = result'."""
    if digits == 1: lo, hi = -9, 9
    elif digits == 2: lo, hi = -99, 99
    else: raise ValueError(f'distance: unsupported digits={digits}')
    x1, y1, x2, y2 = [rng.randint(lo, hi) for _ in range(4)]
    r = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return f'd s q [ {per_digit(x1)} ; {per_digit(y1)} ; {per_digit(x2)} ; {per_digit(y2)} ] = {per_digit(r)}\n'


def gen_triangle_area2(rng: random.Random, digits: int = 1) -> str:
    """Twice the area of triangle with vertices (x1,y1), (x2,y2), (x3,y3) —
    |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|. Doubled to stay integer.
    Format: 't r i [ x1;y1 ; x2;y2 ; x3;y3 ] = result'."""
    if digits == 1: lo, hi = -9, 9
    else: raise ValueError(f'triangle_area2: unsupported digits={digits} (use 1)')
    x1, y1, x2, y2, x3, y3 = [rng.randint(lo, hi) for _ in range(6)]
    r = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    parts = ' ; '.join(f'{per_digit(x)}' for x in [x1, y1, x2, y2, x3, y3])
    return f't r i [ {parts} ] = {per_digit(r)}\n'


def gen_complex_mul(rng: random.Random, digits: int = 1) -> str:
    """Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    Output as 'real ; imag'. Format: 'c m u l [ a ; b ; c ; d ] = real ; imag'."""
    if digits == 1: lo, hi = -9, 9
    elif digits == 2: lo, hi = -99, 99
    else: raise ValueError(f'complex_mul: unsupported digits={digits}')
    a, b, c, d = [rng.randint(lo, hi) for _ in range(4)]
    re_part = a * c - b * d
    im_part = a * d + b * c
    return f'c m u l [ {per_digit(a)} ; {per_digit(b)} ; {per_digit(c)} ; {per_digit(d)} ] = {per_digit(re_part)} ; {per_digit(im_part)}\n'


def gen_complex_modsq(rng: random.Random, digits: int = 1) -> str:
    """Squared modulus of complex number a+bi: |a+bi|^2 = a^2+b^2.
    Squared to keep answer integer. Format: 'c m o d [ a ; b ] = result'."""
    if digits == 1: lo, hi = -9, 9
    elif digits == 2: lo, hi = -99, 99
    else: raise ValueError(f'complex_modsq: unsupported digits={digits}')
    a = rng.randint(lo, hi); b = rng.randint(lo, hi)
    r = a * a + b * b
    return f'c m o d [ {per_digit(a)} ; {per_digit(b)} ] = {per_digit(r)}\n'


def gen_partial_sum(rng: random.Random, digits: int = 1) -> str:
    """Partial sum of 1+2+...+n (Gauss formula = n(n+1)/2). Tests sequence/series.
    Format: 's u m [ n ] = result'."""
    if digits == 1: n_lo, n_hi = 1, 9
    elif digits == 2: n_lo, n_hi = 1, 99
    elif digits == 3: n_lo, n_hi = 1, 999
    else: raise ValueError(f'partial_sum: unsupported digits={digits}')
    n = rng.randint(n_lo, n_hi)
    s = n * (n + 1) // 2
    return f's u m [ {per_digit(n)} ] = {per_digit(s)}\n'


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
    p.add_argument('--polymul', action='store_true',
                   help='polynomial multiplication: ( x + a ) ( x + b ) = x^2 + (a+b)x + ab')
    p.add_argument('--factor', action='store_true',
                   help='quadratic factoring: x^2 + p x + q = ( x + a ) ( x + b )')
    p.add_argument('--diff', action='store_true',
                   help='quadratic differentiation: d/dx[a x^2 + b x + c] = 2a x + b')
    p.add_argument('--collatz-step', action='store_true',
                   help='one step of Collatz/3n+1: col[n] = next')
    p.add_argument('--collatz-time', action='store_true',
                   help='Collatz total stopping time: ctl[n] = steps until 1')
    p.add_argument('--goldbach', action='store_true',
                   help='Goldbach decomposition: gdc[n] = p + q (even n -> two primes)')
    p.add_argument('--fermat-little', action='store_true',
                   help="Fermat's little theorem: flt[a; p] = a^(p-1) mod p")
    # Phase C: cross-discipline tasks
    p.add_argument('--gcd', action='store_true',
                   help='greatest common divisor: gcd[a; b] = result')
    p.add_argument('--lcm', action='store_true',
                   help='least common multiple: lcm[a; b] = result')
    p.add_argument('--totient', action='store_true',
                   help="Euler totient phi: phi[n] = #{k in [1,n] : gcd(k,n)=1}")
    p.add_argument('--modinv', action='store_true',
                   help='modular multiplicative inverse: inv[a; m] = a^-1 mod m')
    p.add_argument('--choose', action='store_true',
                   help='binomial coefficient n choose k: chs[n; k]')
    p.add_argument('--fib', action='store_true',
                   help='nth Fibonacci number: fib[n]')
    p.add_argument('--det2', action='store_true',
                   help='2x2 determinant: det[a; b; c; d] = ad - bc')
    p.add_argument('--dot2', action='store_true',
                   help='2D vector dot product: dot[a; b; c; d] = ac + bd')
    p.add_argument('--distance', action='store_true',
                   help='squared distance: dsq[x1; y1; x2; y2] = (x2-x1)^2 + (y2-y1)^2')
    p.add_argument('--triangle-area2', action='store_true',
                   help='twice triangle area from coords: tri[x1;y1;x2;y2;x3;y3]')
    p.add_argument('--complex-modsq', action='store_true',
                   help='squared modulus of complex number: cmod[a; b] = a^2 + b^2')
    p.add_argument('--partial-sum', action='store_true',
                   help='partial sum 1+2+...+n: sum[n] = n(n+1)/2')
    p.add_argument('--with-calc', action='store_true',
                   help='Phase D: emit the calc-tag-wrapped variant. The model '
                        "learns to output '\\x01{compact_expr}\\x02' instead of the "
                        'final per-digit answer. Inference wrapper evaluates the '
                        'tag content and formats result per-digit for scoring. '
                        'Currently supported only with --ops on basic arithmetic.')
    p.add_argument('--factor-sym', action='store_true',
                   help='Phase D ext-2: emit sympy-augmented quadratic factoring data. '
                        "Per-digit prompt 'x ^ 2 + p x + q', sym-tag answer "
                        "'\\x03factor(x**2 + p*x + q)\\x04'. Inference wrapper applies "
                        'sympy.factor and reformats to per-digit factored form.')
    p.add_argument('--seed', type=int, default=20260504)
    p.add_argument('--shuffle', action='store_true', default=True,
                   help='shuffle problems before writing (default on)')
    args = p.parse_args()

    SYMBOLIC_MODE = (args.algebra or args.distribute or args.polymul or args.factor
                     or args.diff or args.collatz_step or args.collatz_time
                     or args.goldbach or args.fermat_little
                     or args.gcd or args.lcm or args.totient or args.modinv
                     or args.choose or args.fib or args.det2 or args.dot2
                     or args.distance or args.triangle_area2 or args.complex_modsq
                     or args.partial_sum or args.factor_sym)
    if not SYMBOLIC_MODE and (args.digits is None) == (args.max_digits is None):
        p.error('exactly one of --digits or --max-digits must be set (or use a symbolic flag)')
    if SYMBOLIC_MODE and args.digits is None:
        # Sensible per-task default if user didn't specify --digits.
        if args.collatz_time:
            args.digits = 3   # 1-digit collatz_time has tiny output range; 3-digit is the interesting case
        elif args.goldbach:
            args.digits = 3
        elif args.fermat_little:
            args.digits = 2
        else:
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
        if args.polymul:
            problems.append(gen_polymul(rng, args.digits))
            continue
        if args.factor:
            problems.append(gen_factor(rng, args.digits))
            continue
        if args.diff:
            problems.append(gen_differentiate(rng, args.digits))
            continue
        if args.collatz_step:
            problems.append(gen_collatz_step(rng, args.digits))
            continue
        if args.collatz_time:
            problems.append(gen_collatz_time(rng, args.digits))
            continue
        if args.goldbach:
            problems.append(gen_goldbach(rng, args.digits))
            continue
        if args.fermat_little:
            problems.append(gen_fermat_little(rng, args.digits))
            continue
        if args.gcd:
            problems.append(gen_gcd(rng, args.digits)); continue
        if args.lcm:
            problems.append(gen_lcm(rng, args.digits)); continue
        if args.totient:
            problems.append(gen_totient(rng, args.digits)); continue
        if args.modinv:
            problems.append(gen_modinv(rng, args.digits)); continue
        if args.choose:
            problems.append(gen_choose(rng, args.digits)); continue
        if args.fib:
            problems.append(gen_fib(rng, args.digits)); continue
        if args.det2:
            problems.append(gen_det2(rng, args.digits)); continue
        if args.dot2:
            problems.append(gen_dot2(rng, args.digits)); continue
        if args.distance:
            problems.append(gen_distance(rng, args.digits)); continue
        if args.triangle_area2:
            problems.append(gen_triangle_area2(rng, args.digits)); continue
        if args.complex_modsq:
            problems.append(gen_complex_modsq(rng, args.digits)); continue
        if args.partial_sum:
            problems.append(gen_partial_sum(rng, args.digits)); continue
        if args.factor_sym:
            problems.append(gen_factor_sym(rng, args.digits)); continue
        d = args.digits if args.digits else rng.randint(1, args.max_digits)
        if args.mixed:
            problems.append(gen_mixed(d, args.ops, args.n_ops, rng))
        elif args.with_calc:
            op = rng.choice(args.ops)
            problems.append(gen_problem_calc(d, op, rng))
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
