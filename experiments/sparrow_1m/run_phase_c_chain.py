"""Phase C orchestrator — train 12 Sparrows for missing disciplines, eval each
against both Owl Alpha and Gemma 3 27B.

Steps per iter:
  1. Generate 1M-row dataset via gen_arith.py
  2. Build 1M init via build_init.py
  3. Train 25K steps on RTX 3060
  4. Eval vs openrouter/owl-alpha (n=50, scoring=numeric, rps=0.3)
  5. Eval vs google/gemma-3-27b-it (n=50, scoring=numeric, rps=1.0)

iter21-32 are the new entries:
"""
import json
import os
import subprocess
import time
from pathlib import Path

KEY_PATH = r"D:\FANT_TRAINING_D_Drive\fant2\.openrouter_runtime_key"
SCRIPTS = r"C:/FANT/crowfeather-50m-v1_repo/experiments/sparrow_1m"

# (iter_dir, task, digits, gen_flag)
PIPELINE = [
    ("iter21", "gcd",            2, "--gcd"),
    ("iter22", "lcm",            2, "--lcm"),
    ("iter23", "totient",        2, "--totient"),
    ("iter24", "modinv",         2, "--modinv"),
    ("iter25", "choose",         2, "--choose"),
    ("iter26", "fib",            2, "--fib"),
    ("iter27", "det2",           1, "--det2"),
    ("iter28", "dot2",           1, "--dot2"),
    ("iter29", "distance",       1, "--distance"),
    ("iter30", "triangle_area2", 1, "--triangle-area2"),
    ("iter31", "complex_modsq",  1, "--complex-modsq"),
    ("iter32", "partial_sum",    2, "--partial-sum"),
]

TRAIN_STEPS = 25000
N_EVAL = 50
N_DATA = 1_000_000
RPS_OWL = 0.3
RPS_GEMMA = 1.0


def generate_data(iter_dir: str, gen_flag: str, digits: int, seed: int):
    out = f"E:/sparrow/{iter_dir}/data.txt"
    os.makedirs(f"E:/sparrow/{iter_dir}", exist_ok=True)
    cmd = ["py", "-3", f"{SCRIPTS}/gen_arith.py",
           gen_flag, "--digits", str(digits),
           "--n", str(N_DATA), "--seed", str(seed),
           "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"{iter_dir} data gen failed: {proc.stderr[:500]}")
    print(f"  [{iter_dir}] data: {out}")


def build_init(iter_dir: str, seed: int):
    init = f"E:/sparrow/{iter_dir}/init"
    os.makedirs(init, exist_ok=True)
    cmd = ["py", "-3", f"{SCRIPTS}/build_init.py",
           "--output-dir", init, "--seed", str(seed)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def train(iter_dir: str):
    cmd = [
        "py", "-3", "-u", f"{SCRIPTS}/train_local.py",
        "--resume", f"E:/sparrow/{iter_dir}/init",
        "--output", f"E:/sparrow/{iter_dir}/trained",
        "--data",   f"E:/sparrow/{iter_dir}/data.txt",
        "--steps", str(TRAIN_STEPS),
        "--peak-lr", "3e-3", "--min-lr", "3e-4",
        "--warmup-steps", "200",
        "--ckpt-every", "5000", "--log-every", "1000",
    ]
    log_path = f"/tmp/{iter_dir}_train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=2400)
    if proc.returncode != 0:
        raise RuntimeError(f"{iter_dir} train failed; see {log_path}")


def eval_one(iter_dir: str, task: str, digits: int, env: dict, baseline: str, suffix: str, rps: float):
    cmd = [
        "py", "-3", "-u", f"{SCRIPTS}/eval_vs_1b.py",
        "--sparrow", f"E:/sparrow/{iter_dir}/trained/final",
        "--baseline", baseline,
        "--baseline-provider", "openrouter",
        "--task", task, "--digits", str(digits),
        "--n", str(N_EVAL), "--scoring", "numeric",
        "--rps", str(rps),
        "--report", f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d{suffix}.json",
    ]
    log_path = f"/tmp/{iter_dir}_eval{suffix}.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                              text=True, timeout=1800)
    return proc.returncode, log_path


def read_acc(iter_dir, task, digits, suffix):
    p = f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d{suffix}.json"
    try:
        d = json.load(open(p))
        return d["sparrow"]["accuracy"], d["baseline"]["accuracy"], d["baseline"].get("n_failed", 0)
    except Exception:
        return None, None, None


def main():
    with open(KEY_PATH) as f:
        key = json.load(f)["key"]
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # Step 1: generate all 12 datasets in parallel via background subprocesses
    print("=== Phase C step 1: generating 12 datasets in parallel ===")
    procs = []
    for i, (iter_dir, task, digits, gen_flag) in enumerate(PIPELINE):
        seed = 21500 + i
        out = f"E:/sparrow/{iter_dir}/data.txt"
        os.makedirs(f"E:/sparrow/{iter_dir}", exist_ok=True)
        cmd = ["py", "-3", f"{SCRIPTS}/gen_arith.py",
               gen_flag, "--digits", str(digits),
               "--n", str(N_DATA), "--seed", str(seed),
               "--out", out]
        log = open(f"/tmp/{iter_dir}_gen.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((iter_dir, p, log))
    for iter_dir, p, log in procs:
        rc = p.wait(timeout=900)
        log.close()
        if rc != 0:
            print(f"  [{iter_dir}] data gen FAILED rc={rc}")
        else:
            size = os.path.getsize(f"E:/sparrow/{iter_dir}/data.txt") / 1e6
            print(f"  [{iter_dir}] data done ({size:.1f} MB)")

    # Step 2: build 12 inits (sequential, fast)
    print("\n=== Phase C step 2: building 12 inits ===")
    for i, (iter_dir, *_) in enumerate(PIPELINE):
        seed = 22500 + i
        build_init(iter_dir, seed)
        print(f"  [{iter_dir}] init built")

    # Steps 3+4+5: train + eval Owl + eval Gemma per iter, in series
    print("\n=== Phase C steps 3-5: train + dual-baseline eval per iter ===")
    results = []
    t_start = time.time()
    for iter_dir, task, digits, _ in PIPELINE:
        print(f"\n--- {iter_dir} ({task} {digits}d) ---")
        t0 = time.time()
        try:
            train(iter_dir)
            print(f"  [{iter_dir}] trained ({(time.time()-t0)/60:.1f} min)")
        except Exception as e:
            print(f"  [{iter_dir}] TRAIN FAIL: {e}")
            results.append({"iter": iter_dir, "task": task, "digits": digits, "status": "train_fail", "error": str(e)})
            continue

        rc_owl, _ = eval_one(iter_dir, task, digits, env,
                             "openrouter/owl-alpha", "_owl", RPS_OWL)
        rc_gemma, _ = eval_one(iter_dir, task, digits, env,
                               "google/gemma-3-27b-it", "_gemma", RPS_GEMMA)
        sp_o, owl, owl_failed = read_acc(iter_dir, task, digits, "_owl")
        sp_g, gemma, gemma_failed = read_acc(iter_dir, task, digits, "_gemma")
        sp = sp_o if sp_o is not None else sp_g
        results.append({
            "iter": iter_dir, "task": task, "digits": digits,
            "status": "ok",
            "sparrow": sp, "owl": owl, "gemma": gemma,
            "owl_failed": owl_failed, "gemma_failed": gemma_failed,
            "win_owl_pp":   ((sp - owl)   * 100) if (sp is not None and owl   is not None) else None,
            "win_gemma_pp": ((sp - gemma) * 100) if (sp is not None and gemma is not None) else None,
        })
        if sp is not None:
            wo = (sp - owl)*100 if owl is not None else None
            wg = (sp - gemma)*100 if gemma is not None else None
            print(f"  [{iter_dir}] Sparrow={sp*100:.1f}%  Owl={owl*100 if owl is not None else 0:.1f}% (delta {wo:+.1f}pp)  Gemma={gemma*100 if gemma is not None else 0:.1f}% (delta {wg:+.1f}pp)")

    total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Phase C complete in {total/60:.1f} min")
    print('=' * 80)
    print(f'{"Iter":7s} {"Task":18s} {"Digits":>6s}  {"Sparrow":>8s} {"Owl":>8s} {"Gemma":>8s}  {"vs Owl":>10s} {"vs Gemma":>11s}  {"≥1?":>4s}')
    print('-' * 100)
    n_decisive_either = 0
    for r in results:
        if r["status"] != "ok":
            print(f'{r["iter"]:7s} {r["task"]:18s} {r["digits"]:>6d}  STATUS={r["status"]}')
            continue
        wo = r.get("win_owl_pp", 0) or 0
        wg = r.get("win_gemma_pp", 0) or 0
        decisive = (wo > 5) or (wg > 5)
        if decisive: n_decisive_either += 1
        flag = "YES" if decisive else "  "
        print(f'{r["iter"]:7s} {r["task"]:18s} {r["digits"]:>6d}  '
              f'{r["sparrow"]*100:>7.1f}% {r["owl"]*100:>7.1f}% {r["gemma"]*100:>7.1f}%  '
              f'{wo:>+7.1f}pp {wg:>+8.1f}pp  {flag:>4s}')
    print('-' * 100)
    print(f"  Sparrow beats AT LEAST ONE of (Owl, Gemma) decisively on {n_decisive_either}/{len(results)} iters")
    summary_path = f"{SCRIPTS}/phase_c_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"total_elapsed_sec": total, "results": results}, f, indent=2)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
