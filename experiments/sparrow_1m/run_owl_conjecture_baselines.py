"""Run Owl-only n=50 baselines for the 4 conjecture tasks (iter17-20).

These run serially since they share the OpenRouter rate-limit budget.
Sparrow doesn't exist for these tasks yet (no training has happened),
so we use --skip-sparrow. Each writes its own JSON report.
"""
import json
import os
import subprocess
import sys
import time

KEY_PATH = r"D:\FANT_TRAINING_D_Drive\fant2\.openrouter_runtime_key"
ARTIFACT_ROOT = "E:/sparrow/owl_baselines"
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_vs_1b.py")

TASKS = [
    ("collatz_step",  3),
    ("collatz_time",  3),
    ("goldbach",      3),
    ("fermat_little", 2),
]

N = 50
RPS = 0.3


def main():
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    with open(KEY_PATH) as f:
        key = json.load(f)["key"]
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    results = []
    t_start = time.time()
    for idx, (task, digits) in enumerate(TASKS):
        report_path = f"{ARTIFACT_ROOT}/eval_{task}_{digits}d_owl_only.json"
        print(f"\n[{idx+1}/{len(TASKS)}] === {task}  digits={digits} ===", flush=True)

        cmd = [
            "py", "-3", "-u", EVAL_SCRIPT,
            "--skip-sparrow",
            "--baseline", "openrouter/owl-alpha",
            "--baseline-provider", "openrouter",
            "--task", task, "--digits", str(digits),
            "--n", str(N),
            "--scoring", "numeric",
            "--rps", str(RPS),
            "--report", report_path,
        ]
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(f"    | {line.rstrip()}", flush=True)
        proc.wait(timeout=900)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"    FAILED rc={proc.returncode} in {elapsed:.1f}s")
            results.append({"task": task, "digits": digits, "status": "failed",
                            "elapsed_sec": elapsed})
            continue
        try:
            with open(report_path) as f:
                rep = json.load(f)
            owl_acc = rep["baseline"]["accuracy"]
            n_failed = rep["baseline"].get("n_failed", 0)
            print(f"    DONE in {elapsed:.1f}s  Owl={owl_acc*100:.1f}%  failed={n_failed}", flush=True)
            results.append({
                "task": task, "digits": digits, "status": "ok",
                "elapsed_sec": elapsed,
                "owl_acc": owl_acc, "owl_failed": n_failed,
                "owl_p50_ms": rep["baseline"]["p50_latency_ms"],
            })
        except Exception as e:
            print(f"    PARSE FAIL: {e}")
            results.append({"task": task, "digits": digits, "status": "parse_failed",
                            "elapsed_sec": elapsed, "error": str(e)})

    total = time.time() - t_start
    print()
    print("=" * 80)
    print(f"Owl-only baselines complete in {total/60:.1f} min")
    print("=" * 80)
    print(f'{"Task":15s} {"Digits":>6s}  {"Owl":>9s}  {"Failed":>6s}  {"p50 ms":>9s}')
    print('-' * 80)
    for r in results:
        if r["status"] == "ok":
            print(f'{r["task"]:15s} {r["digits"]:>6d}  '
                  f'{r["owl_acc"]*100:>8.1f}% {r["owl_failed"]:>5d}     {r["owl_p50_ms"]:>7.0f}')
        else:
            print(f'{r["task"]:15s} {r["digits"]:>6d}  STATUS={r["status"]}')
    print()

    summary_path = os.path.join(ARTIFACT_ROOT, "owl_only_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"n": N, "rps": RPS, "results": results,
                   "total_elapsed_sec": total}, f, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
