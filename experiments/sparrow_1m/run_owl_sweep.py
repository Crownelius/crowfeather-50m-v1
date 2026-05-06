"""Run head-to-head Sparrow vs openrouter/owl-alpha across all canonical iters.

Reads the runtime API key from .openrouter_runtime_key, sets OPENROUTER_API_KEY,
and invokes eval_vs_1b.py for each (iter, task, digits) tuple in the ladder.

Each run writes its own JSON sidecar so partial progress is preserved.
A summary table is printed and saved at the end.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Canonical scoreboard iters (use the WINNER from each sweep, not the first attempt).
LADDER = [
    ("iter1",   "add",        2),
    ("iter2",   "add",        3),
    ("iter3",   "add",        4),
    ("iter4",   "sub",        3),
    ("iter5",   "mul",        2),
    ("iter6a",  "mul",        3),
    ("iter7",   "mixed",      1),
    ("iter8",   "div",        3),
    ("iter9",   "sub",        4),
    ("iter10",  "algebra",    1),
    ("iter11b", "algebra",    2),
    ("iter12d", "algebra",    3),
    ("iter13",  "distribute", 1),
    ("iter14",  "polymul",    1),
    ("iter15",  "factor",     1),
]

KEY_PATH = r"D:\FANT_TRAINING_D_Drive\fant2\.openrouter_runtime_key"
ARTIFACT_ROOT = "E:/sparrow"
N_PER_TASK = 50
# Empirical Owl Alpha rate behavior: small token bucket (~3-4 calls) that
# drains then refills at ~1/30s. Slow probe at rps=0.1 still saw 40% fails;
# bucket size doesn't depend on rps. The eval's 5x exponential backoff
# (1+2+4+8s) gives the bucket time to refill mid-burst, so effective fail
# rate at rps=0.3-0.4 is ~10-15% — acceptable for verdict-grade eval.
# n=50 gives Wilson CI ~±14% at 50% accuracy, sufficient for win/tie/loss.
RPS = 0.3
SCORING = "numeric"
BASELINE = "openrouter/owl-alpha"

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_vs_1b.py")


def main():
    with open(KEY_PATH) as f:
        key = json.load(f)["key"]

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    # Make tqdm/transformers quieter on stdout if present
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    results = []
    t_start = time.time()

    for idx, (iter_dir, task, digits) in enumerate(LADDER):
        sparrow_path = f"{ARTIFACT_ROOT}/{iter_dir}/trained/final"
        report_path = f"{sparrow_path}/eval_{task}_{digits}d_owl.json"

        if not Path(sparrow_path.replace("/", os.sep)).exists():
            print(f"\n[{idx+1}/{len(LADDER)}] SKIP {iter_dir} — sparrow path missing")
            continue

        print(f"\n[{idx+1}/{len(LADDER)}] === {iter_dir}  task={task}  digits={digits} ===")
        print(f"    sparrow: {sparrow_path}")
        print(f"    report:  {report_path}")

        cmd = [
            "py", "-3", "-u", EVAL_SCRIPT,
            "--sparrow", sparrow_path,
            "--baseline", BASELINE,
            "--baseline-provider", "openrouter",
            "--task", task,
            "--digits", str(digits),
            "--n", str(N_PER_TASK),
            "--scoring", SCORING,
            "--rps", str(RPS),
            "--report", report_path,
        ]
        t0 = time.time()
        # Stream child stdout/stderr live so progress is visible. Each child
        # iteration takes ~3-5 minutes; we want to see the inner progress
        # ("[50/100] acc_so_far=...") not just one line at the end.
        stderr_tail: list = []
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            print(f"    | {line}")
            sys.stdout.flush()
            stderr_tail.append(line)
            if len(stderr_tail) > 30:
                stderr_tail.pop(0)
        proc.wait(timeout=900)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"    FAILED rc={proc.returncode} in {elapsed:.1f}s")
            results.append({
                "iter": iter_dir, "task": task, "digits": digits,
                "status": "failed", "elapsed_sec": elapsed,
                "output_tail": stderr_tail[-10:],
            })
            continue

        # Parse the report JSON
        try:
            with open(report_path) as f:
                report = json.load(f)
            sp_acc = report["sparrow"]["accuracy"]
            bl = report["baseline"]
            bl_acc = bl["accuracy"]
            n_failed = bl.get("n_failed", 0)
            delta = sp_acc - bl_acc
            print(f"    DONE in {elapsed:.1f}s   "
                  f"Sparrow={sp_acc*100:.1f}%  Owl={bl_acc*100:.1f}%  "
                  f"Δ={delta*100:+.1f}pp  failed={n_failed}")
            results.append({
                "iter": iter_dir, "task": task, "digits": digits,
                "status": "ok", "elapsed_sec": elapsed,
                "sparrow_acc": sp_acc, "owl_acc": bl_acc, "delta_pp": delta * 100,
                "owl_failed": n_failed,
                "sparrow_p50_ms": report["sparrow"]["p50_latency_ms"],
                "owl_p50_ms": bl["p50_latency_ms"],
                "owl_usage": bl.get("usage", {}),
            })
        except Exception as e:
            print(f"    PARSE FAIL after {elapsed:.1f}s: {e}")
            results.append({
                "iter": iter_dir, "task": task, "digits": digits,
                "status": "parse_failed", "elapsed_sec": elapsed,
                "error": str(e),
            })

    total_elapsed = time.time() - t_start

    # Summary
    print()
    print("=" * 88)
    print(f"SWEEP COMPLETE  total wall = {total_elapsed/60:.1f} min")
    print("=" * 88)
    print(f"{'Iter':10s} {'Task':12s} {'Digits':>6s}  {'Sparrow':>8s} {'Owl':>8s}  {'Δ':>8s}  {'Verdict':10s}")
    print("-" * 88)
    n_win = n_tie = n_loss = n_other = 0
    for r in results:
        if r["status"] != "ok":
            print(f"{r['iter']:10s} {r['task']:12s} {r['digits']:>6d}  {'-':>8s} {'-':>8s}  {'-':>8s}  {r['status']:10s}")
            n_other += 1
            continue
        d = r["delta_pp"]
        if d > 0.5:
            verdict = "WIN"; n_win += 1
        elif d < -0.5:
            verdict = "LOSS"; n_loss += 1
        else:
            verdict = "TIE"; n_tie += 1
        print(f"{r['iter']:10s} {r['task']:12s} {r['digits']:>6d}  "
              f"{r['sparrow_acc']*100:>7.1f}% {r['owl_acc']*100:>7.1f}%  "
              f"{d:>+7.1f}pp {verdict:10s}")

    print("-" * 88)
    print(f"  Wins: {n_win}   Ties: {n_tie}   Losses: {n_loss}   Other: {n_other}")

    # Save the sweep summary
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "owl_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_per_task": N_PER_TASK,
            "rps": RPS,
            "scoring": SCORING,
            "baseline": BASELINE,
            "total_elapsed_sec": total_elapsed,
            "results": results,
            "tally": {"wins": n_win, "ties": n_tie, "losses": n_loss, "other": n_other},
        }, f, indent=2, default=str)
    print(f"\n  summary saved: {summary_path}")


if __name__ == "__main__":
    main()
