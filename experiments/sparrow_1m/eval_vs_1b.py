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
import re
import sys
import time

import torch

# Local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes_tok import encode, decode, EOS_ID  # noqa: E402
from gen_arith import gen_problem, per_digit  # noqa: E402


# ----- Scoring --------------------------------------------------------------
#
# Two modes:
#   string  - exact whitespace-stripped equality on the per-digit format.
#             What iter1-15 historical scoreboards used. Llama-3.2-1B got 0%
#             on iters 2-9 partly because it wrote "1376" instead of "1 3 7 6"
#             — that's a format mismatch, not a math failure.
#   numeric - parse expected and generated as integers (arithmetic/algebra
#             tasks) or sympy polynomials (distribute/polymul/factor/diff)
#             and compare for equivalence. Honest comparison of math
#             capability across models that don't share the per-digit format.

# Phase D: calc-tag sentinels. Sparrow's calc-mode training data wraps each
# problem's RHS as '\x01<compact_expr>\x02' (e.g. '\x01123 * 456\x02'). At
# inference, the calc-aware decoder extracts the inner expression, evaluates
# it in a sandboxed AST, and re-formats the integer result per-digit for
# scoring. This decouples Sparrow's translation capability from the underlying
# arithmetic — the python tool does the math.
CALC_OPEN_BYTE = '\x01'
CALC_CLOSE_BYTE = '\x02'

import ast as _ast


def _safe_eval_calc_expr(expr: str):
    """Evaluate a tiny arithmetic expression safely. Allowed: int literals,
    +, -, *, // (integer division), unary +/-. Anything else returns None.
    Defends against the obvious 'eval(expr)' RCE risk if Sparrow ever emits
    weird bytes."""
    try:
        tree = _ast.parse(expr.strip(), mode='eval')
    except (SyntaxError, ValueError):
        return None

    def _walk(node):
        if isinstance(node, _ast.Expression):
            return _walk(node.body)
        if isinstance(node, _ast.Constant):
            return node.value if isinstance(node.value, int) else None
        if isinstance(node, _ast.UnaryOp):
            v = _walk(node.operand)
            if v is None:
                return None
            if isinstance(node.op, _ast.UAdd):
                return +v
            if isinstance(node.op, _ast.USub):
                return -v
            return None
        if isinstance(node, _ast.BinOp):
            l = _walk(node.left); r = _walk(node.right)
            if l is None or r is None:
                return None
            op = node.op
            if isinstance(op, _ast.Add):
                return l + r
            if isinstance(op, _ast.Sub):
                return l - r
            if isinstance(op, _ast.Mult):
                return l * r
            if isinstance(op, _ast.FloorDiv):
                return l // r if r != 0 else None
            if isinstance(op, _ast.Mod):
                return l % r if r != 0 else None
            return None
        return None

    return _walk(tree)


def _per_digit(n: int) -> str:
    if n < 0:
        return '- ' + ' '.join(str(-n))
    return ' '.join(str(n))


def apply_calc_wrapper(generated: str):
    """Decode a Sparrow calc-mode OR sym-mode generation. Auto-dispatch by
    which sentinel pair appears in the output:
      \\x01..\\x02 -> python AST eval (integer arithmetic)
      \\x03..\\x04 -> sympy eval (symbolic: factor / expand / diff / integrate)
    Returns the per-digit result string, or '<*-fail-*>' on failure.
    """
    # Sym tag dispatch first (more specific failure modes if both somehow present)
    if SYM_OPEN_BYTE in generated:
        return apply_sym_wrapper(generated)
    open_idx = generated.find(CALC_OPEN_BYTE)
    if open_idx < 0:
        return '<calc-fail-no-open>'
    close_idx = generated.find(CALC_CLOSE_BYTE, open_idx + 1)
    if close_idx < 0:
        return '<calc-fail-no-close>'
    inner = generated[open_idx + 1 : close_idx]
    val = _safe_eval_calc_expr(inner)
    if val is None:
        return '<calc-fail-eval>'
    return _per_digit(val)


# ===== Phase D extension 2: sympy-augmented Sparrow =====
# Separate sentinel pair so the inference dispatcher can pick wrapper by tag:
#   \x01..\x02 -> python AST eval (integer arithmetic only)
#   \x03..\x04 -> sympy eval (symbolic; supports factor/expand/integrate/diff)
SYM_OPEN_BYTE  = '\x03'
SYM_CLOSE_BYTE = '\x04'


def _format_sympy_factor(expr) -> str:
    """Format a sympy factored expression back to Sparrow's per-digit form.
    Example: (x - 12)*(x + 7) -> '( x - 1 2 ) ( x + 7 )'.
    Handles signs, multi-digit coefficients, and the canonical (factor)(factor) layout.
    """
    import sympy as sp
    s = str(sp.factor(expr))
    # sympy emits things like: (x - 12)*(x + 7), (x - 2)*(x + 3)*(x - 5)
    # We need: ( x - 1 2 ) ( x + 7 )  — drop '*', space digits, space symbols.
    # Strategy: tokenize via regex, rebuild with spaces.
    import re
    # Match: literals (-?\d+), 'x', '+', '-', '(', ')', '*'
    parts = re.findall(r'-?\d+|x|\+|-|\(|\)|\*\*|\*', s.replace(' ', ''))
    out = []
    for tok in parts:
        if tok == '*':
            continue   # drop multiplication between factors
        if tok == '**':
            out.append('^')
        elif re.fullmatch(r'-?\d+', tok):
            n = int(tok)
            if n < 0:
                out.append('-')
                out.append(' '.join(str(-n)))
            else:
                out.append(' '.join(str(n)))
        else:
            out.append(tok)
    # Join with spaces; collapse double minus that comes from negative-int + leading sign
    rebuilt = ' '.join(out)
    return rebuilt


def _format_sympy_polynomial(expr) -> str:
    """Format a sympy polynomial in expanded form to Sparrow's per-digit form.
    Example: x**2 - 5*x + 6 -> 'x ^ 2 - 5 x + 6'.
    """
    import sympy as sp
    s = str(sp.expand(expr))
    import re
    parts = re.findall(r'-?\d+|x|\+|-|\*\*|\*', s.replace(' ', ''))
    out = []
    for tok in parts:
        if tok == '*':
            continue
        if tok == '**':
            out.append('^')
        elif re.fullmatch(r'-?\d+', tok):
            n = int(tok)
            if n < 0:
                out.append('-')
                out.append(' '.join(str(-n)))
            else:
                out.append(' '.join(str(n)))
        else:
            out.append(tok)
    return ' '.join(out)


def apply_sym_wrapper(generated: str, output_format: str = 'auto'):
    """Decode a Sparrow sym-mode generation. Inner expression is one of:
        factor(EXPR), expand(EXPR), integrate(EXPR, x), diff(EXPR, x)
    Whitelist enforced via regex; only the allowed sympy functions can be
    called. EXPR is parsed via sympify with a single symbol 'x' available.

    Returns the per-digit result string, or '<sym-fail-*>' on parse failure.
    """
    open_idx = generated.find(SYM_OPEN_BYTE)
    if open_idx < 0:
        return '<sym-fail-no-open>'
    close_idx = generated.find(SYM_CLOSE_BYTE, open_idx + 1)
    if close_idx < 0:
        return '<sym-fail-no-close>'
    inner = generated[open_idx + 1 : close_idx].strip()

    # Whitelist dispatch: only these top-level forms are allowed.
    # Match: <fn_name>(<arg>)  where fn_name is one of {factor, expand, diff, integrate, simplify}
    # arg can contain x, integers, +, -, *, /, **, parens, commas, spaces.
    m = re.fullmatch(r'(factor|expand|diff|integrate|simplify)\(\s*(.+)\s*\)', inner)
    if not m:
        return '<sym-fail-no-fn>'
    fn_name, arg_str = m.group(1), m.group(2)
    # Reject anything non-arithmetic in the arg — keep the surface tiny.
    # Allow: digits, x, +, -, *, /, **, (, ), ',', whitespace
    if not re.fullmatch(r'[\d x\+\-\*\(\),\s\.]+|[\d x\+\-\*\(\),\s\.]+\*\*[\d x\+\-\*\(\),\s\.]+', arg_str):
        # Looser regex: just disallow names other than 'x' and arithmetic chars
        if re.search(r'[A-Za-wy-z_]', arg_str):  # any letter besides 'x' — block (covers __import__ etc.)
            return '<sym-fail-bad-arg>'
    try:
        import sympy as sp
        x = sp.Symbol('x')
        # parse_expr with single-symbol local dict
        from sympy.parsing.sympy_parser import parse_expr
        # If it's a 2-arg call (integrate, diff), split on the LAST top-level comma
        if fn_name in ('integrate', 'diff'):
            # arg_str ends with ', x' typically. Just always integrate/diff w.r.t. x.
            # Split on comma not inside parens
            depth = 0
            split_idx = -1
            for i, ch in enumerate(arg_str):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    split_idx = i
                    break
            if split_idx >= 0:
                expr_str = arg_str[:split_idx].strip()
            else:
                expr_str = arg_str.strip()  # implicit integration variable = x
            expr = parse_expr(expr_str, local_dict={'x': x})
            fn = sp.integrate if fn_name == 'integrate' else sp.diff
            result = fn(expr, x)
        else:
            expr = parse_expr(arg_str, local_dict={'x': x})
            fn = {'factor': sp.factor, 'expand': sp.expand, 'simplify': sp.simplify}[fn_name]
            result = fn(expr)
    except Exception as e:
        return f'<sym-fail-eval:{type(e).__name__}>'
    try:
        # Auto-pick output format by the function called: factor -> factored
        # form `( x - a ) ( x - b )`; everything else -> expanded polynomial.
        if output_format == 'auto':
            output_format = 'factor' if fn_name == 'factor' else 'poly'
        if output_format == 'factor':
            return _format_sympy_factor(result)
        else:
            return _format_sympy_polynomial(result)
    except Exception:
        return '<sym-fail-format>'


ARITHMETIC_TASKS = {'add', 'sub', 'mul', 'div', 'mixed', 'algebra',
                    'collatz_step', 'collatz_time', 'fermat_little',
                    # Phase C single-int answers:
                    'gcd', 'lcm', 'totient', 'modinv',
                    'choose', 'fib',
                    'det2', 'dot2',
                    'distance', 'triangle_area2',
                    'complex_modsq',
                    'partial_sum'}
POLYNOMIAL_TASKS = {'distribute', 'polymul', 'factor', 'diff'}
GOLDBACH_TASKS = {'goldbach'}


def _collapse_digit_spaces(s: str) -> str:
    """'1 2 3 + 4 5 6' -> '123 + 456'. Idempotent."""
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = re.sub(r'(\d) +(\d)', r'\1\2', cur)
    return cur


def _to_int_or_none(token: str):
    try:
        return int(token)
    except ValueError:
        return None


def _numeric_int_match(expected: str, generated: str) -> bool:
    """True if both parse to the same integer."""
    # Strip ALL whitespace from expected so "- 3 2 4" -> "-324" parses cleanly.
    exp_parsed = _to_int_or_none(
        _collapse_digit_spaces(expected).replace(' ', '').replace(',', '').strip()
    )
    if exp_parsed is None:
        return False
    # CRITICAL ORDER: collapse digit-spaces BEFORE stripping equation operators.
    # Otherwise "6 3 + 1 1 = 7 4" -> "6 3 + 1 1   7 4" (= becomes 2 spaces) ->
    # _collapse_digit_spaces eats across the boundary "1   7" -> "17", giving
    # the wrong last-integer extraction (1174 instead of 74).
    g = _collapse_digit_spaces(generated.strip())
    g = g.replace(',', '').replace('$', '').replace('=', ' ').strip().rstrip('.')
    direct = _to_int_or_none(g.replace(' ', ''))
    if direct is not None:
        return direct == exp_parsed
    # Fall back: extract the LAST signed integer (LMs often say
    # "...the answer is 42." or echo the full equation — last int is the answer).
    matches = re.findall(r'-?\d+', g)
    if matches:
        last = _to_int_or_none(matches[-1])
        if last is not None:
            return last == exp_parsed
    return False


_POLY_CACHE: dict = {}


def _to_sympy_expr(s: str):
    """Per-digit format polynomial -> sympy expression. None on parse failure."""
    if s in _POLY_CACHE:
        return _POLY_CACHE[s]
    try:
        from sympy import symbols, sympify
    except ImportError:
        _POLY_CACHE[s] = None
        return None
    x = symbols('x')

    cleaned = _collapse_digit_spaces(s).strip()
    cleaned = cleaned.replace('^', '**')
    # Insert implicit-multiplication asterisks
    cleaned = re.sub(r'(\d)\s*x', r'\1*x', cleaned)        # 4x or 4 x -> 4*x
    cleaned = re.sub(r'\)\s*\(', ')*(', cleaned)            # )( -> )*(
    cleaned = re.sub(r'(\d)\s*\(', r'\1*(', cleaned)        # 4(x+1) -> 4*(x+1)
    cleaned = re.sub(r'\)\s*(\d)', r')*\1', cleaned)        # (x+1)4 -> (x+1)*4

    try:
        expr = sympify(cleaned, locals={'x': x})
    except Exception:
        _POLY_CACHE[s] = None
        return None
    _POLY_CACHE[s] = expr
    return expr


def _polynomial_match(expected: str, generated: str) -> bool:
    """True if expected and generated parse to equivalent sympy polynomials."""
    e_expr = _to_sympy_expr(expected)
    g_expr = _to_sympy_expr(generated)
    if e_expr is None or g_expr is None:
        return False
    try:
        from sympy import simplify, expand
        return simplify(expand(e_expr) - expand(g_expr)) == 0
    except Exception:
        return False


def _goldbach_match(expected: str, generated: str) -> bool:
    """For Goldbach: credit ANY 'p + q' where both p,q are prime and sum to
    the same integer as expected (which is the canonical decomposition
    smallest-p+rest).

    Defends against the trivial `n + 0` cheat that polynomial-equivalence would
    incorrectly credit: requires both p and q to pass a primality test.
    """
    e = _collapse_digit_spaces(expected).replace(' ', '')
    if '+' not in e:
        return False
    try:
        e_p_str, e_q_str = e.split('+', 1)
        e_sum = int(e_p_str) + int(e_q_str)
    except ValueError:
        return False
    g = _collapse_digit_spaces(generated.strip())
    m = re.search(r'(-?\d+)\s*\+\s*(-?\d+)', g)
    if not m:
        return False
    try:
        g_p, g_q = int(m.group(1)), int(m.group(2))
    except ValueError:
        return False
    if g_p + g_q != e_sum:
        return False
    # Inline primality test (don't import gen_arith — keep eval self-contained)
    def _isprime(n: int) -> bool:
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
    return _isprime(g_p) and _isprime(g_q)


def score_match(expected: str, generated: str, task: str, mode: str = 'string') -> bool:
    """Return True if `generated` matches `expected` under the chosen mode."""
    if mode == 'string':
        return generated.strip() == expected.strip()
    if mode != 'numeric':
        raise ValueError(f"unknown scoring mode: {mode!r}")
    if task in ARITHMETIC_TASKS:
        return _numeric_int_match(expected, generated)
    if task in POLYNOMIAL_TASKS:
        return _polynomial_match(expected, generated)
    if task in GOLDBACH_TASKS:
        return _goldbach_match(expected, generated)
    return generated.strip() == expected.strip()


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
            line = gen_algebra(rng, digits).rstrip('\n')
            # Format: "a x ± b = c x = solution"
            # Split at the LAST " x = " to separate equation from answer
            eq_part, answer = line.rsplit(' x = ', 1)
            prompt = eq_part + ' x = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'distribute':
        from gen_arith import gen_distribute
        for _ in range(n):
            line = gen_distribute(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'polymul':
        from gen_arith import gen_polymul
        for _ in range(n):
            line = gen_polymul(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'factor':
        from gen_arith import gen_factor
        for _ in range(n):
            line = gen_factor(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'diff':
        from gen_arith import gen_differentiate
        for _ in range(n):
            line = gen_differentiate(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'collatz_step':
        from gen_arith import gen_collatz_step
        for _ in range(n):
            line = gen_collatz_step(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'collatz_time':
        from gen_arith import gen_collatz_time
        for _ in range(n):
            line = gen_collatz_time(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'goldbach':
        from gen_arith import gen_goldbach
        for _ in range(n):
            line = gen_goldbach(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    if task == 'fermat_little':
        from gen_arith import gen_fermat_little
        for _ in range(n):
            line = gen_fermat_little(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
            problems.append((prompt, answer, line))
        return problems
    # Phase C: same prompt-cut pattern (' = ' boundary). Generic fallback below.
    PHASE_C_GENS = {
        'gcd':            'gen_gcd',
        'lcm':            'gen_lcm',
        'totient':        'gen_totient',
        'modinv':         'gen_modinv',
        'choose':         'gen_choose',
        'fib':            'gen_fib',
        'det2':           'gen_det2',
        'dot2':           'gen_dot2',
        'distance':       'gen_distance',
        'triangle_area2': 'gen_triangle_area2',
        'complex_modsq':  'gen_complex_modsq',
        'partial_sum':    'gen_partial_sum',
    }
    if task in PHASE_C_GENS:
        import gen_arith as _ga
        gen_fn = getattr(_ga, PHASE_C_GENS[task])
        for _ in range(n):
            line = gen_fn(rng, digits).rstrip('\n')
            eq_part, answer = line.split(' = ')
            prompt = eq_part + ' = '
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

def eval_sparrow(model_dir: str, problems: list, device: str, max_new: int = 32,
                 task: str = 'add', scoring: str = 'string',
                 calc_wrapper: bool = False):
    from transformers import Qwen3ForCausalLM
    print(f'  loading Sparrow-1M from {model_dir}'
          + ('  [calc-wrapper ON]' if calc_wrapper else ''))
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
            # Phase D: pass calc-mode generations through the python tool
            # before scoring. The raw output looks like '\x01123 * 456\x02';
            # the wrapper produces '5 6 0 8 8' for per-digit comparison.
            scored_output = generated
            if calc_wrapper:
                scored_output = apply_calc_wrapper(generated)
            n_out_tokens += out.shape[1] - ids.shape[1]
            if score_match(expected, scored_output, task, scoring):
                n_correct += 1
            if len(sample_outputs) < 5:
                # Save BOTH the raw generation and the wrapped output for
                # diagnostic clarity in the report JSON.
                if calc_wrapper:
                    sample_outputs.append((prompt, expected, f'{generated!r} -> {scored_output!r}'))
                else:
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
            examples.append(gen_algebra(rng, digits).rstrip('\n'))
    elif op == 'distribute':
        from gen_arith import gen_distribute
        for _ in range(k):
            examples.append(gen_distribute(rng, digits).rstrip('\n'))
    elif op == 'polymul':
        from gen_arith import gen_polymul
        for _ in range(k):
            examples.append(gen_polymul(rng, digits).rstrip('\n'))
    elif op == 'factor':
        from gen_arith import gen_factor
        for _ in range(k):
            examples.append(gen_factor(rng, digits).rstrip('\n'))
    elif op == 'diff':
        from gen_arith import gen_differentiate
        for _ in range(k):
            examples.append(gen_differentiate(rng, digits).rstrip('\n'))
    elif op == 'collatz_step':
        from gen_arith import gen_collatz_step
        for _ in range(k):
            examples.append(gen_collatz_step(rng, digits).rstrip('\n'))
    elif op == 'collatz_time':
        from gen_arith import gen_collatz_time
        for _ in range(k):
            examples.append(gen_collatz_time(rng, digits).rstrip('\n'))
    elif op == 'goldbach':
        from gen_arith import gen_goldbach
        for _ in range(k):
            examples.append(gen_goldbach(rng, digits).rstrip('\n'))
    elif op == 'fermat_little':
        from gen_arith import gen_fermat_little
        for _ in range(k):
            examples.append(gen_fermat_little(rng, digits).rstrip('\n'))
    elif op in {'gcd', 'lcm', 'totient', 'modinv', 'choose', 'fib',
                'det2', 'dot2', 'distance', 'triangle_area2',
                'complex_modsq', 'partial_sum'}:
        import gen_arith as _ga
        gen_fn = getattr(_ga, f'gen_{op}')
        for _ in range(k):
            examples.append(gen_fn(rng, digits).rstrip('\n'))
    else:
        for _ in range(k):
            examples.append(gen_problem(digits, op, rng).rstrip('\n'))
    return FEW_SHOT_TEMPLATE.format(
        examples='\n'.join(examples),
        prompt=prompt,
    )


def eval_baseline_local(model_id: str, problems: list, device: str,
                        k_shots: int, digits: int, op: str, max_new: int = 32,
                        task: str = 'add', scoring: str = 'string'):
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
            if score_match(expected, first_line, task, scoring):
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
                             api_key: str = None, task: str = 'add',
                             scoring: str = 'string'):
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
        gave_up_429 = False
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
                        print(f'    [{i+1}/{len(problems)}] rate-limited 5x in a row; giving up on this row')
                        failed += 1
                        gave_up_429 = True
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

                # Owl Alpha sometimes returns HTTP 200 with content=None (empty
                # generation, refusal, or upstream hiccup). Don't crash; mark
                # failed and continue.
                if content is None:
                    print(f'    [{i+1}/{len(problems)}] HTTP 200 with content=None; marking failed')
                    failed += 1
                    break

                first_line = content.strip().split('\n')[0].strip()
                if score_match(expected, first_line, task, scoring):
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

        # If we just exhausted retries on 429s, give the upstream rate window
        # an extra cool-off so the next problem isn't pre-doomed.
        if gave_up_429:
            print(f'    [{i+1}/{len(problems)}] 30s extra cool-off after 5x 429s')
            time.sleep(30)

        # Progress log every 50
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(problems) - i - 1) / max(rate, 0.01)
            cur_acc = n_correct / max(i + 1 - failed, 1)
            print(f'    {i+1}/{len(problems)}  acc_so_far={100*cur_acc:.1f}%  '
                  f'rate={rate:.2f}/s  ETA={eta/60:.1f}m  failed={failed}')

        # Bail out early if more than 30% of rows have failed — that's a hard
        # rate limit, not transient. Better to retry the whole eval later than
        # produce a noisy accuracy number.
        if failed > 0 and failed / max(i + 1, 1) > 0.30 and (i + 1) >= 20:
            print(f'    aborting: {failed}/{i+1} ({100*failed/(i+1):.0f}%) rows failed — '
                  f'rate limit pressure too high, retry later with lower rps')
            break

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
                  provider: str = 'auto', rps: float = 5.0, max_new: int = 32,
                  task: str = 'add', scoring: str = 'string'):
    """Dispatch to local or OpenRouter baseline."""
    if provider == 'auto':
        # Use OpenRouter if a key is available AND the model id looks like
        # an OpenRouter slug (contains a '/').
        has_key = bool(os.environ.get('OPENROUTER_API_KEY'))
        looks_like_slug = '/' in model_id
        provider = 'openrouter' if (has_key and looks_like_slug) else 'local'
    if provider == 'openrouter':
        return eval_baseline_openrouter(model_id, problems, k_shots, digits, op,
                                        max_new=max_new, rps=rps,
                                        task=task, scoring=scoring)
    return eval_baseline_local(model_id, problems, device, k_shots, digits, op,
                               max_new=max_new, task=task, scoring=scoring)


# ----- Driver ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sparrow', default=None,
                   help='Sparrow-1M trained final/ dir. Required unless --skip-sparrow is set.')
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
    p.add_argument('--task', default='add', choices=[
        'add', 'sub', 'mul', 'div', 'mixed', 'algebra',
        'distribute', 'polymul', 'factor', 'diff',
        'collatz_step', 'collatz_time', 'goldbach', 'fermat_little',
        # Phase C cross-discipline:
        'gcd', 'lcm', 'totient', 'modinv',
        'choose', 'fib',
        'det2', 'dot2',
        'distance', 'triangle_area2',
        'complex_modsq',
        'partial_sum',
    ])
    p.add_argument('--digits', type=int, default=2)
    p.add_argument('--n', type=int, default=1000, help='test problems')
    p.add_argument('--k-shots', type=int, default=5, help='few-shot examples for baseline')
    p.add_argument('--device', default=None)
    p.add_argument('--report', default=None,
                   help='write JSON report (default: <sparrow>/eval_<task>_<digits>d[_numeric].json)')
    p.add_argument('--scoring', default='string', choices=['string', 'numeric'],
                   help='string = exact whitespace-stripped equality (default; back-compat with iter1-15). '
                        'numeric = parse as int (arithmetic/algebra/conjecture) or sympy polynomial '
                        '(distribute/polymul/factor/diff) or prime-pair (goldbach) and compare for equivalence.')
    p.add_argument('--skip-baseline', action='store_true',
                   help='only eval Sparrow (faster smoke test)')
    p.add_argument('--skip-sparrow', action='store_true',
                   help='only eval the baseline model (use for Owl-only baseline runs '
                        'when no Sparrow is trained yet for a new task)')
    p.add_argument('--calc-wrapper', action='store_true',
                   help='Phase D tool-augmented inference: route Sparrow output through '
                        'a python calc-tag wrapper before scoring. Use with sparrow_calc '
                        'models that emit \\x01<expr>\\x02. Has no effect on baseline.')
    args = p.parse_args()

    if not args.skip_sparrow and not args.sparrow:
        p.error('--sparrow is required unless --skip-sparrow is set')

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'  device: {device}')
    print(f'  task: {args.task} ({args.digits}-digit)  n={args.n}  scoring={args.scoring}')

    # For all non-arithmetic tasks the op key is just the task name (used in
    # build_few_shot_prompt to dispatch to the right generator).
    _BASIC_OPS = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
    op = _BASIC_OPS.get(args.task, args.task)
    problems = build_test_set(args.task, args.digits, args.n)

    sparrow_results = None
    if not args.skip_sparrow:
        print('\n=== Sparrow-1M eval ===')
        sparrow_results = eval_sparrow(args.sparrow, problems, device,
                                       task=args.task, scoring=args.scoring,
                                       calc_wrapper=args.calc_wrapper)

    baseline_results = None
    if not args.skip_baseline:
        print(f'\n=== Baseline eval ({args.baseline}) ===')
        baseline_results = eval_baseline(args.baseline, problems, device,
                                         args.k_shots, args.digits, op,
                                         provider=args.baseline_provider,
                                         rps=args.rps,
                                         task=args.task, scoring=args.scoring)

    # ----- Report
    print()
    print('=' * 78)
    print(f'HEAD-TO-HEAD: {args.task} ({args.digits}-digit)  n={args.n}')
    print('=' * 78)
    print(f'  {"Model":35s} {"Acc":>7s} {"tok/s":>10s} {"p50_latency":>12s}')
    print(f'  {"-"*35} {"-"*7} {"-"*10} {"-"*12}')
    s = sparrow_results
    if s is not None:
        print(f'  {s["name"]:35s} {100*s["accuracy"]:>6.1f}%  {s["tokens_per_sec"]:>9.0f} {s["p50_latency_ms"]:>10.1f}ms')
    else:
        print(f'  {"(Sparrow eval skipped)":35s}')
    if baseline_results:
        b = baseline_results
        print(f'  {b["name"]:35s} {100*b["accuracy"]:>6.1f}%  {b["tokens_per_sec"]:>9.0f} {b["p50_latency_ms"]:>10.1f}ms')

        if s is not None:
            delta = 100 * (s['accuracy'] - b['accuracy'])
            if b.get('n_params'):
                size_str = f'{b["n_params"]/1.078e6:.0f}x smaller'
            else:
                size_str = '~1000x smaller (1M vs ~1B)'
            print()
            if delta > 0.5:
                print(f'  Sparrow-1M WINS by {delta:.1f}pp  ({size_str})')
            elif delta < -0.5:
                print(f'  Sparrow-1M loses by {-delta:.1f}pp  ({size_str})')
            else:
                print(f'  Tie ({delta:+.1f}pp)  ({size_str})')

        # OpenRouter usage stats if applicable
        if b.get('provider') == 'openrouter' and b.get('usage'):
            print(f'  OpenRouter usage: {b["usage"]["input_tokens"]:,} in + '
                  f'{b["usage"]["output_tokens"]:,} out tokens '
                  f'({b.get("n_failed", 0)} failed)')

    print()
    if s is not None:
        print('  Sparrow-1M sample outputs:')
        for prompt, expected, generated in s['sample_outputs']:
            match = 'OK' if score_match(expected, generated, args.task, args.scoring) else 'WRONG'
            print(f'    [{match:5s}] prompt={prompt!r}  exp={expected!r}  got={generated!r}')

    if baseline_results:
        print(f'\n  {args.baseline} sample outputs:')
        for prompt, expected, generated in baseline_results['sample_outputs']:
            match = 'OK' if score_match(expected, generated, args.task, args.scoring) else 'WRONG'
            print(f'    [{match:5s}] prompt={prompt!r}  exp={expected!r}  got={generated!r}')

    # JSON report
    suffix = '' if args.scoring == 'string' else '_numeric'
    default_report = f'eval_{args.task}_{args.digits}d{suffix}.json'
    if args.report:
        report_path = args.report
    elif args.sparrow:
        report_path = os.path.join(args.sparrow, default_report)
    else:
        # --skip-sparrow with no --report and no --sparrow: drop in CWD.
        report_path = default_report
    report = {
        'task': args.task, 'digits': args.digits, 'n': args.n,
        'scoring_mode': args.scoring,
        'baseline_model': args.baseline,
        'sparrow': sparrow_results,
        'baseline': baseline_results,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=str)
    print(f'\n  report saved: {report_path}')


if __name__ == '__main__':
    main()
