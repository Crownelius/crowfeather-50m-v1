"""Phase D extension chain: train + eval calc-tagged Sparrow on harder
arithmetic tasks where original Sparrow would have failed AND Owl Alpha
likely degrades.

Target tasks:
  iter34 = 5-digit calc-mul   (output up to 10 digits; Sparrow couldn't
                              fit raw at 1M, Owl probably degrades)
  iter35 = 4-digit calc-div   (4-digit / 1-digit; harder than iter8)

Each iter: generate 1M calc-tagged rows, build init, train 25K, eval
vs Owl + Gemma with --calc-wrapper.
"""
import json
import os
import subprocess
import time

KEY_PATH = r"D:\FANT_TRAINING_D_Drive\fant2\.openrouter_runtime_key"
SCRIPTS = r"C:/FANT/crowfeather-50m-v1_repo/experiments/sparrow_1m"

PIPELINE = [
    ("iter34", "mul", 5, "*"),
    ("iter35", "div", 4, "/"),
]

TRAIN_STEPS = 25000
N_EVAL = 50


def run(cmd, log_path, env=None, timeout=2400):
    with open(log_path, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout).returncode


def main():
    with open(KEY_PATH) as f:
        key = json.load(f)["key"]
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    results = []
    t_start = time.time()
    for iter_dir, task, digits, op in PIPELINE:
        print(f"\n=== {iter_dir}  task={task}  digits={digits}  op={op!r} ===", flush=True)

        # 1. Generate calc-tagged data
        out = f"E:/sparrow/{iter_dir}/data.txt"
        os.makedirs(f"E:/sparrow/{iter_dir}", exist_ok=True)
        seed = 34000 + (1 if iter_dir == "iter35" else 0)
        gen_cmd = ["py", "-3", f"{SCRIPTS}/gen_arith.py",
                   "--with-calc", "--ops", op, "--digits", str(digits),
                   "--n", "1000000", "--seed", str(seed),
                   "--out", out]
        rc = run(gen_cmd, f"/tmp/{iter_dir}_gen.log", timeout=600)
        if rc != 0:
            print(f"  [{iter_dir}] gen FAILED rc={rc}")
            continue
        size = os.path.getsize(out) / 1e6
        print(f"  [{iter_dir}] data done ({size:.1f} MB)")

        # 2. Build init
        init = f"E:/sparrow/{iter_dir}/init"
        os.makedirs(init, exist_ok=True)
        init_cmd = ["py", "-3", f"{SCRIPTS}/build_init.py",
                    "--output-dir", init, "--seed", str(seed)]
        run(init_cmd, f"/tmp/{iter_dir}_init.log", timeout=120)
        print(f"  [{iter_dir}] init built")

        # 3. Train
        t_train = time.time()
        train_cmd = [
            "py", "-3", "-u", f"{SCRIPTS}/train_local.py",
            "--resume", init,
            "--output", f"E:/sparrow/{iter_dir}/trained",
            "--data", out,
            "--steps", str(TRAIN_STEPS),
            "--peak-lr", "3e-3", "--min-lr", "3e-4",
            "--warmup-steps", "200",
            "--ckpt-every", "5000", "--log-every", "1000",
        ]
        rc = run(train_cmd, f"/tmp/{iter_dir}_train.log", timeout=2400)
        if rc != 0:
            print(f"  [{iter_dir}] train FAILED rc={rc}")
            results.append({"iter": iter_dir, "status": "train_fail"})
            continue
        print(f"  [{iter_dir}] trained ({(time.time()-t_train)/60:.1f} min)")

        # 4. Eval vs Owl + Gemma with calc-wrapper
        for baseline, suffix, rps in [
            ("openrouter/owl-alpha", "_owl_calc",   "0.3"),
            ("google/gemma-3-27b-it", "_gemma_calc", "1.0"),
        ]:
            eval_cmd = [
                "py", "-3", "-u", f"{SCRIPTS}/eval_vs_1b.py",
                "--sparrow", f"E:/sparrow/{iter_dir}/trained/final",
                "--baseline", baseline, "--baseline-provider", "openrouter",
                "--task", task, "--digits", str(digits),
                "--n", str(N_EVAL), "--scoring", "numeric",
                "--rps", rps, "--calc-wrapper",
                "--report", f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d{suffix}.json",
            ]
            rc = run(eval_cmd, f"/tmp/{iter_dir}_eval{suffix}.log", env=env, timeout=1800)
            if rc != 0:
                print(f"  [{iter_dir}] eval{suffix} rc={rc}")

        # 5. Read both results
        try:
            o = json.load(open(f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d_owl_calc.json"))
            g = json.load(open(f"E:/sparrow/{iter_dir}/trained/final/eval_{task}_{digits}d_gemma_calc.json"))
            sp = o["sparrow"]["accuracy"]
            ow = o["baseline"]["accuracy"]
            ge = g["baseline"]["accuracy"]
            print(f"  [{iter_dir}] DONE  Sparrow_calc={sp*100:.1f}%  Owl={ow*100:.1f}%  Gemma={ge*100:.1f}%  "
                  f"vs-Owl={(sp-ow)*100:+.1f}pp  vs-Gemma={(sp-ge)*100:+.1f}pp", flush=True)
            results.append({"iter": iter_dir, "task": task, "digits": digits, "status": "ok",
                            "sparrow_calc": sp, "owl": ow, "gemma": ge,
                            "vs_owl_pp": (sp-ow)*100, "vs_gemma_pp": (sp-ge)*100})
        except Exception as e:
            print(f"  [{iter_dir}] read result failed: {e}")
            results.append({"iter": iter_dir, "status": "parse_fail"})

    total = time.time() - t_start
    print(f"\n{'=' * 80}\nPhase D extension chain complete in {total/60:.1f} min\n{'=' * 80}")
    summary = f"{SCRIPTS}/phase_d_extension_summary.json"
    with open(summary, "w") as f:
        json.dump({"results": results, "total_elapsed_sec": total}, f, indent=2)
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
