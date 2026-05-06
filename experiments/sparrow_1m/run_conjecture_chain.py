"""Auto-chain Sparrow conjecture trainings + evals for iter17-20.

Polls until iter17's already-launched training completes, then runs iter17 eval,
then for iter18-20 builds init + trains + evals each in series. Single GPU, so
trainings are sequential.

Each train: 25K steps, peak_lr=3e-3, vanilla AdamW + cosine. Each eval: n=50,
scoring=numeric, rps=0.3, against openrouter/owl-alpha.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KEY_PATH = r"D:\FANT_TRAINING_D_Drive\fant2\.openrouter_runtime_key"
SCRIPTS = r"C:/FANT/crowfeather-50m-v1_repo/experiments/sparrow_1m"

# (iter_dir, task, digits, build_init?, already_launched?)
PIPELINE = [
    ("iter17", "collatz_step",  3, False, True),   # training already running
    ("iter18", "collatz_time",  3, True,  False),
    ("iter19", "goldbach",      3, True,  False),
    ("iter20", "fermat_little", 2, True,  False),
]

TRAIN_STEPS = 25000
N_EVAL = 50
RPS = 0.3


def wait_for_final(iter_dir: str, poll_sec: int = 30, timeout_sec: int = 1800) -> bool:
    """Block until E:/sparrow/<iter_dir>/trained/final/model.safetensors exists."""
    final_path = Path(f"E:/sparrow/{iter_dir}/trained/final/model.safetensors")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if final_path.exists():
            # Wait a bit longer for the JSON files to flush
            time.sleep(5)
            return True
        time.sleep(poll_sec)
    return False


def build_init(iter_dir: str, seed: int):
    init_dir = f"E:/sparrow/{iter_dir}/init"
    os.makedirs(init_dir, exist_ok=True)
    cmd = ["py", "-3", f"{SCRIPTS}/build_init.py",
           "--output-dir", init_dir, "--seed", str(seed)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  [{iter_dir}] init built at {init_dir}")


def train(iter_dir: str):
    cmd = [
        "py", "-3", "-u", f"{SCRIPTS}/train_local.py",
        "--resume", f"E:/sparrow/{iter_dir}/init",
        "--output", f"E:/sparrow/{iter_dir}/trained",
        "--data",   f"E:/sparrow/{iter_dir}/data.txt",
        "--steps", str(TRAIN_STEPS),
        "--peak-lr", "3e-3", "--min-lr", "3e-4",
        "--warmup-steps", "200",
        "--ckpt-every", "5000", "--log-every", "500",
    ]
    log_path = f"/tmp/{iter_dir}_train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=2400)
    if proc.returncode != 0:
        raise RuntimeError(f"{iter_dir} training failed; see {log_path}")
    print(f"  [{iter_dir}] training done; log={log_path}")


def eval_owl(iter_dir: str, task: str, digits: int, env: dict):
    cmd = [
        "py", "-3", "-u", f"{SCRIPTS}/eval_vs_1b.py",
        "--sparrow", f"E:/sparrow/{iter_dir}/trained/final",
        "--baseline", "openrouter/owl-alpha",
        "--baseline-provider", "openrouter",
        "--task", task, "--digits", str(digits),
        "--n", str(N_EVAL), "--scoring", "numeric",
        "--rps", str(RPS),
        "--report", f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d_owl.json",
    ]
    log_path = f"/tmp/{iter_dir}_eval.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                              text=True, timeout=1800)
    if proc.returncode != 0:
        # Don't abort the whole chain on a single eval failure
        print(f"  [{iter_dir}] eval rc={proc.returncode}; see {log_path}")
    print(f"  [{iter_dir}] eval done; log={log_path}")


def read_result(iter_dir: str, task: str, digits: int):
    p = f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d_owl.json"
    try:
        with open(p) as f:
            d = json.load(f)
        return {
            "iter": iter_dir, "task": task, "digits": digits,
            "sparrow": d["sparrow"]["accuracy"],
            "owl":     d["baseline"]["accuracy"],
            "owl_failed": d["baseline"].get("n_failed", 0),
            "delta_pp": (d["sparrow"]["accuracy"] - d["baseline"]["accuracy"]) * 100,
        }
    except Exception as e:
        return {"iter": iter_dir, "error": str(e)}


def main():
    with open(KEY_PATH) as f:
        key = json.load(f)["key"]
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    results = []
    t_start = time.time()

    for iter_dir, task, digits, do_init, already in PIPELINE:
        print(f"\n=== {iter_dir}  task={task}  digits={digits} ===", flush=True)

        if already:
            # iter17 was already launched separately; just wait for it.
            print(f"  [{iter_dir}] waiting for already-launched training...")
            ok = wait_for_final(iter_dir, poll_sec=20, timeout_sec=1800)
            if not ok:
                print(f"  [{iter_dir}] TIMEOUT waiting for training. Skipping.")
                continue
        else:
            seed = 17500 + int(iter_dir.replace("iter", ""))  # iter18 -> 17518, etc.
            if do_init:
                build_init(iter_dir, seed)
            print(f"  [{iter_dir}] training (~12-15 min)...", flush=True)
            try:
                train(iter_dir)
            except Exception as e:
                print(f"  [{iter_dir}] TRAIN FAILED: {e}")
                continue

        print(f"  [{iter_dir}] evaluating vs Owl Alpha (~7 min)...", flush=True)
        eval_owl(iter_dir, task, digits, env)
        r = read_result(iter_dir, task, digits)
        results.append(r)
        if "error" not in r:
            print(f"  [{iter_dir}] DONE  Sparrow={r['sparrow']*100:.1f}%  "
                  f"Owl={r['owl']*100:.1f}%  delta={r['delta_pp']:+.1f}pp", flush=True)

    total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Conjecture chain complete in {total/60:.1f} min")
    print('=' * 80)
    print(f'{"Iter":7s} {"Task":15s}  {"Sparrow":>9s} {"Owl":>9s}  {"Delta":>9s}')
    print('-' * 80)
    for r in results:
        if "error" in r:
            print(f'{r["iter"]:7s}  ERROR: {r["error"][:60]}')
        else:
            print(f'{r["iter"]:7s} {r["task"]:15s}  '
                  f'{r["sparrow"]*100:>8.1f}% {r["owl"]*100:>8.1f}%  '
                  f'{r["delta_pp"]:>+7.1f}pp')

    # Save summary
    summary_path = f"{SCRIPTS}/conjecture_chain_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_elapsed_sec": total}, f, indent=2)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
