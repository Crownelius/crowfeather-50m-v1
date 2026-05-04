# Sparrow-1M-Arith

A 1.08M-parameter dense Qwen3 model trained as an arithmetic specialist. The thesis: **with enough specialization on a narrow task, a 1M model can beat a 1B generalist on that specific task**. Sparrow-1M is the proof.

This is an experiment, not a product. It will not be released to HuggingFace as a chatbot. Its only job is to win one contest: per-digit multi-digit arithmetic, head-to-head against `meta-llama/Llama-3.2-1B-Instruct`.

---

## Honest framing

**What Sparrow-1M can credibly beat Llama-3.2-1B at:**

- 4-digit addition exact-match accuracy
- 3-digit subtraction exact-match accuracy
- 3-digit multiplication exact-match accuracy
- 3-digit-by-1-digit division exact-match accuracy
- Mixed expressions in our specific format
- Inference latency (~50× faster, ~1000× smaller VRAM)
- Tokens/sec on CPU
- Per-token training cost

**What Sparrow-1M will NEVER beat Llama-3.2-1B at:**

- MMLU
- HumanEval
- General prose / chat / Q&A
- Anything OOD from arithmetic
- Coherent multi-paragraph generation

If you're tempted to claim more, don't. The point of this experiment is to demonstrate the principle ("massive specialization compensates for capacity at narrow tasks"), not to build a viable general model at 1M scale.

---

## Architecture

| Component | Value |
|---|---|
| Architecture | Qwen3 dense (HF transformers, same family as Crowfeather-50M) |
| Vocab | **256** (raw bytes; no BPE training needed; covers all UTF-8) |
| hidden_size | 128 |
| num_hidden_layers | 5 |
| num_attention_heads | 4 (Q) |
| num_key_value_heads | 2 (KV, GQA 2:1) |
| head_dim | 32 |
| intermediate_size | 512 (4× hidden, SwiGLU 3-matrix) |
| max_position_embeddings | 512 |
| tied embeddings | yes |
| **Total params** | **~1.078M** |

Embedding is only 32K params (3% of total) because vocab=256 is tiny — that leaves 1.05M for transformer layers, which is meaningful capacity at this scale.

Why bytes (vocab=256): arithmetic content is all ASCII. A char-level / byte-level vocab makes per-digit format trivially uniform, removes any BPE merging that could collapse "12" into one token, and produces a tokenizer with zero training cost.

---

## Task & data format

Single training format, single line per problem:

```
1 2 3 + 4 5 6 = 5 7 9\n
```

- Operands are space-separated digits (per-digit).
- Operator: one of `+`, `-`, `*`, `/`.
- `\n` ends the example; the trainer treats it as both EOS and the boundary between concatenated examples.
- Result is also space-separated. For division, results are integers (truncated); a future iter can extend to remainder format.

At inference, you prompt with `"1 2 3 + 4 5 6 = "` and the model generates digits until `\n`.

The synthetic generator (`gen_arith.py`) emits an effectively infinite stream of these. A typical training run uses 5-50M tokens — at ~30 bytes per problem, that's ~150K-1.6M problems.

---

## Eval — head-to-head vs Llama-3.2-1B-Instruct

For each iteration's task (e.g. "4-digit addition"):

1. Generate 1000 held-out problems sampled uniformly from the digit-count range.
2. **Sparrow-1M**: prompt `"{a} {op} {b} = "`, greedy decode until `\n`, exact-match the answer string.
3. **Llama-3.2-1B-Instruct**: 5-shot prompt with our format, greedy decode, exact-match.
4. Report a comparison table:

```
Task: 4-digit addition (n=1000)
  Sparrow-1M    accuracy=94.7%   tokens/sec=12,400   p50_latency=2.1ms
  Llama-3.2-1B  accuracy=43.2%   tokens/sec=180      p50_latency=140ms
  -> Sparrow-1M wins by 51.5pp
```

The win condition is rigorous: **same problems, same prompt format, same hardware**. We're not comparing zero-shot Llama to specialist Sparrow — we give Llama 5 in-context examples of our format. If Sparrow still wins, that's a real result.

---

## Iteration roadmap

| Iter | Task | Token budget | Sparrow target | Llama-1B (5-shot) expected | Win delta |
|---|---|---|---|---|---|
| 1 | 2-digit add | 5M | ~99% | ~95% | small |
| 2 | 3-digit add | 10M | ~98% | ~80-90% | clear |
| 3 | 4-digit add | 15M | ~95% | ~40-60% | strong |
| 4 | 3-digit subtract | 10M | ~95% | ~70% | clear |
| 5 | 2-digit multiply | 15M | ~95% | ~30% | dominant |
| 6 | 3-digit multiply | 25M | ~90% | ~10% | dominant |
| 7 | mixed (a+b−c×d, 2-digit) | 25M | ~85% | ~30% | strong |
| 8 | 3-digit ÷ 1-digit | 15M | ~85% | ~50% | clear |

If iter N falls short of target, the lever for iter N+1 is curriculum: more examples at the failing operation, longer training, possibly intermediate-step "thinking" traces.

After all 8 iterations land, write up the result.

---

## Compute envelope

| Setup | Wall time per iter (avg ~15M tokens) |
|---|---|
| CPU (8-core) | 1-3 hours |
| RTX 3060 12GB | 5-15 minutes |
| RTX 4090 | 2-5 minutes |

Each iteration is a fresh training run from random init (or warm-start from prior iter). No multi-day GPU rentals required.

Eval against Llama-3.2-1B-Instruct: first run downloads ~2.5 GB of weights from HF; thereafter ~30 sec to load and ~3-5 min to score 1000 problems.

---

## Files

```
experiments/sparrow_1m/
├── README.md           # this file
├── gen_arith.py        # synthetic data generator (per-digit format, configurable digit count + ops)
├── build_init.py       # instantiate the 1M Qwen3 from config
├── train_local.py      # CPU-or-GPU training loop, AdamW + cosine, ~50M tokens default
└── eval_vs_1b.py       # head-to-head vs Llama-3.2-1B-Instruct, prints comparison table
```

---

## Quick start

```bash
# 1. Generate iteration-1 data: 2-digit addition
python experiments/sparrow_1m/gen_arith.py \
    --out E:/sparrow/iter1_2digit_add.txt \
    --n 200000 --digits 2 --ops +

# 2. Build the 1M model from scratch
python experiments/sparrow_1m/build_init.py \
    --output-dir E:/sparrow/iter1/init

# 3. Train (auto-detects CUDA / falls back to CPU)
python experiments/sparrow_1m/train_local.py \
    --resume   E:/sparrow/iter1/init \
    --output   E:/sparrow/iter1/trained \
    --data     E:/sparrow/iter1_2digit_add.txt \
    --steps    5000 \
    --batch-size 64 \
    --seq-len  128 \
    --peak-lr  3e-3 \
    --min-lr   3e-4

# 4. Eval vs Llama-3.2-1B-Instruct (downloads Llama on first run)
python experiments/sparrow_1m/eval_vs_1b.py \
    --sparrow  E:/sparrow/iter1/trained \
    --baseline meta-llama/Llama-3.2-1B-Instruct \
    --task add --digits 2 --n 1000
```

---

## Why this matters scientifically

Each iteration that lands a Sparrow-1M-beats-Llama-1B result is evidence for one of these claims:

1. **Capacity isn't compute.** Llama-1B at 5-shot has 1000× more parameters. If it loses on a narrow well-defined task, the loss is about training data, not capacity.
2. **Format adherence is a separate axis from intelligence.** A specialist that has saturated the per-digit format will exact-match more reliably than a generalist that knows the answer but emits it in a slightly different format.
3. **Ablation playground.** Sparrow trains in <15 min on a modest GPU. Hyperparameter sweeps, architecture variants (depth vs width), and curriculum experiments are all tractable in a single afternoon.

If we land all 8 iterations, the writeup is "an arithmetic specialist 1000× smaller than a 1B generalist beats it on 8 narrow tasks". That's a credible small contribution to the literature on small-LM specialization.

---

## License

Apache 2.0 (inherits from the parent Crowfeather-50M-v1 repo).
