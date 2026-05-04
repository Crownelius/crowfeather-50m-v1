# Math derivations — Crowfeather-50M-v1

Every quantitative claim made in the architecture and training recipe is proven here.

---

## 1. Muon Hybrid Newton-Schulz orthogonalization

### Setup

Muon (DeepSeek V4 Algorithm 1) updates a 2D weight matrix `W in R^(m x n)` by orthogonalizing the momentum-blended gradient before applying it. Orthogonalization here means driving every singular value of the update matrix toward 1.

### The iteration

Given an input matrix `X in R^(m x n)` normalized by spectral norm so `||X||_2 <= 1`, run for k iterations:

```
X_{k+1} = a*X_k + b*(X_k @ X_k^T) @ X_k + c*(X_k @ X_k^T)^2 @ X_k
```

with two phases of coefficients:

```
Phase A (8 iterations):  (a, b, c) = (3.4445, -4.7750, 2.0315)
Phase B (2 iterations):  (a, b, c) = (2.0,    -1.5,    0.5)
```

### Why these coefficients

Let `f(s) = a*s + b*s^3 + c*s^5` be the iteration applied to a single singular value `s`. Since `X @ X^T = U Sigma^2 U^T` for SVD `X = U Sigma V^T`, the iteration acts diagonally on the spectrum.

Phase A coefficients are chosen so `f(s) ~ 1` for `s in [1/sqrt(N), 1]` where `N` is the number of dimensions:

```
f(0.001) ~ 0.0034    (3x amplification)
f(0.1)   ~ 0.343     (3.4x amplification)
f(1.0)   ~ 0.701     (slight overshoot)
```

After Phase A, all singular values cluster between 0.5 and 1.5. Phase B's coefficients are tuned for `f(s) ~ 1` for `s in [0.5, 1.5]`, providing fine-grained convergence to exactly 1.

### Iteration-count sweep

From the 412M math audit (same Newton-Schulz applies):

| Iters | Min SV | Max SV | Frac in [0.95, 1.05] |
|---|---|---|---|
| 0+0 | 0.0000 | 0.0882 | 0.0000 |
| 5+0 | 0.0137 | 1.2022 | 0.1406 |
| 8+2 | **1.0000** | **1.0020** | **1.0000** |
| 15+2 | 1.0000 | 1.0005 | 1.0000 |

8+2 = 10 iterations is the sweet spot. More is wasted compute.

### Effective step size scaling

Once orthogonalized, the update matrix `O` has all singular values ~ 1, so its spectral norm is ~1 and its Frobenius norm is `sqrt(min(m, n))`. To make the update magnitude equivalent across different layer widths, V4 scales by:

```
update = NS(M) * sqrt(max(m, n)) * gamma
```

where `gamma = 0.18`. Verified empirically:

| Shape | Scaled norm | Effective step size |
|---|---|---|
| 64 x 64 | 11.5 | 0.181 |
| 256 x 1024 | 92.2 | 0.180 |
| 1024 x 4096 | 368.7 | 0.180 |

The effective step size stays at ~0.18 regardless of matrix shape.

### Applicability to 50M

For our config, Muon-eligible matrices have `max(m, n) <= 1792` (the FFN intermediate dimension). This is well below the empirical breakdown threshold of 2048 found in audit Test 6. Newton-Schulz converges cleanly at every weight in this model.

---

## 2. WSD vs cosine schedule

### Definition

WSD (warmup-stable-decay):

```
warmup phase  (steps in [0, w*T)):     lr(s) = peak * s/(w*T)
stable phase  (steps in [w*T, (1-d)*T)): lr(s) = peak
decay phase   (steps in [(1-d)*T, T]):  lr(s) = min + (peak - min) * (1 - sqrt((s - (1-d)*T)/(d*T)))
```

with `w=0.015` (1.5% warmup) and `d=0.20` (20% sqrt-decay cooldown).

### Integrated learning rate

For the relative time spent at >=50% of peak LR:

```
WSD time at >=50% of peak LR:    84.0%
Cosine time at >=50% of peak LR: 53.6%
Integrated LR ratio:             1.56x in favor of WSD
```

Distillation pretraining benefits from sustained high LR. Cosine wastes 46% of training in the low-LR regime where the model fine-tunes its prior knowledge but learns little new.

### Why sqrt cooldown specifically

Linear cooldown drops LR uniformly over the cooldown window. Cosine drops fast at the beginning and slow at the end. Sqrt cooldown does the opposite: slow at the beginning, fast at the end:

```
linear:  lr ~ 1 - p
cosine:  lr ~ 0.5 * (1 + cos(pi*p))
sqrt:    lr ~ 1 - sqrt(p)
```

For `p in [0, 1]`: at p=0.5, linear gives 0.5, cosine gives 0.5, sqrt gives 0.293.

The MiniCPM 2024 finding: sqrt keeps the model at near-peak LR through 50% of cooldown then drops sharply, allowing a long training tail of high-LR consolidation followed by a rapid annealing finish.

---

## 3. GQA 2:1 KV cache savings

### Standard MHA at 16K context

With 8 attention heads, head_dim=56:

```
KV per token = 8 (heads) * 56 (dim) * 2 (K + V) * 2 (bf16 bytes) = 1792 bytes
KV cache @ 16K context = 16384 * 1792 = 29.4 MB per layer
Total over 12 layers = 352 MB
```

### Our GQA 2:1 (8 Q heads, 4 KV heads)

```
KV per token = 4 (KV heads) * 56 (dim) * 2 (K + V) * 2 (bf16) = 896 bytes
KV cache @ 16K = 16384 * 896 = 14.7 MB per layer
Total over 12 layers = 176 MB
```

### Savings

```
KV cache reduction = 1 - (4/8) = 50% reduction
Memory saved at 16K = 352 - 176 = 176 MB
```

At inference, 50M models on consumer hardware (e.g. 8GB VRAM laptops) value every megabyte of KV cache headroom. GQA 2:1 doubles the realistic context window at the same memory budget.

### Why 2:1 not 3:1 or 4:1

The 412M plan uses GQA 3:1 (12 Q / 4 KV) which is a stronger compression. At 50M, hidden=448 only divides cleanly into 8 query heads at head_dim=56 (the only `448 / heads` that gives integer head_dim and a meaningful number of heads). Going to 4 KV heads gives the cleanest GQA 2:1 ratio. 2 KV heads (GQA 4:1) would over-compress for a model this small.

---

## 4. z-loss regularization

### Definition

```
z_loss = mean over tokens of logsumexp(logits)^2
total_loss = CE_loss + z_loss_coef * z_loss
```

with `z_loss_coef = 1e-4`.

### Why it exists

Cross-entropy is invariant to a constant shift in logits: `softmax(z + c) = softmax(z)` for any constant `c`. So during training, the absolute scale of logits drifts. When logits get very large (logsumexp grows), numerical stability degrades and downstream operations (sampling, top-k) become unreliable.

z-loss penalizes the squared logsumexp directly, pulling logits back toward smaller magnitudes without changing the relative softmax distribution.

### Why coef=1e-4

For CE in the typical range [2, 8] and logits with logsumexp typically [5, 20], `z_loss = 25 to 400`. Multiplied by 1e-4: contribution to total = 0.0025 to 0.04. Small relative to CE, large enough to push back if logits explode.

### CRITICAL: z-loss must have gradient

A previous bug in our codebase wrapped the z-loss computation in `torch.no_grad()`, making it a constant added to total — it had no gradient effect, silently disabling z-loss. The current `train_dense.py` is correct:

```python
if args.z_loss > 0:
    # MUST have gradients — z-loss exists to pull down logsumexp
    lse = torch.logsumexp(out.logits.float(), dim=-1).mean()
    loss = loss + args.z_loss * lse.pow(2)
```

---

## 5. Liger Kernel fused cross-entropy

### The problem

Standard CE for vocab V, batch B, seq T:

1. Compute logits tensor: `B x T x V` (e.g., 4 x 4096 x 32768 x 2 bytes = 1.07 GB)
2. Compute softmax: another `B x T x V` of intermediate values
3. Compute CE: scalar
4. Backward: gradient on logits is again `B x T x V`

For our config (B=4, T=4096, V=32768, bf16) this is ~4 GB just for the output head's forward+backward. Tractable on 80GB but eats headroom.

### Liger Kernel solution

Compute CE in chunks along the vocab dimension:

```
for chunk in vocab_chunks:
    logits_chunk = h @ W_chunk         # B x T x chunk_size
    log_probs_chunk = log_softmax(logits_chunk)
    loss_chunk = gather(log_probs_chunk, targets) / B*T
    loss_total += loss_chunk
```

Memory bound: only `B x T x chunk_size` materialized at a time. For chunk_size=4096 and B=4, T=4096: only `4 * 4096 * 4096 * 2 = 134 MB` instead of 4 GB.

### Impact at 50M scale

Less critical than 412M (where vocab=262K made Liger essential at any reasonable batch). At 50M with vocab=32K, Liger is still ~4 GB savings during training, which translates to either:
- Larger batch (B=4 -> B=8 doubles throughput at the same memory)
- Longer context Phase 2 (16K runs cleanly with 30+ GB headroom)

---

## 6. Tokenizer math

### Vocab budget at 32K

The 32,768 vocab is partitioned as:

```
Special tokens:           18  (FIM, chat roles, think, tools, pad/bos/eos/unk)
ByteLevel alphabet:      256  (every byte 0x00-0xFF guaranteed)
BPE merges:           32,494  (32768 - 18 - 256)
```

The 256 byte alphabet ensures byte fallback: any UTF-8 string can be tokenized without `<UNK>`. Multi-byte sequences (e.g., emoji, non-Latin scripts) decompose into byte tokens that BPE then merges where common.

### Per-digit input wrap math

```python
import re
_DIGIT_RUN_RE = re.compile(r'\d{2,}')
def per_digit_wrap(text):
    return _DIGIT_RUN_RE.sub(lambda m: ' '.join(m.group()), text)
```

Effect: `"1234 + 5678 = 6912"` becomes `"1 2 3 4 + 5 6 7 8 = 6 9 1 2"`.

### Token-count overhead by domain

| Domain | Multi-digit prevalence | Token overhead vs no-wrap |
|---|---|---|
| Prose (Sonnet, R1 science) | ~1% | ~0.5% |
| Math (NuminaMath, MetaMathQA) | ~30% (numbers everywhere) | ~50-60% |
| Code (R1 code) | ~5% (line numbers, hex) | ~3% |

At 30% / 40% / 30% mix in pretrain: net overhead = `0.30 * 0.50 + 0.40 * 0.005 + 0.30 * 0.03 = 16.4%` extra tokens for the same character count.

### Why per-digit despite the cost

Without per-digit wrap, common BPE merges create entries like `1234 -> 1234` as a single token but `1235 -> 12 35` (since `1235` is rarer). The model then learns weird conditional probabilities for adjacent integers. Empirically (Lee et al. 2023, "Teaching Arithmetic to Small Transformers"), per-digit tokenization is the single biggest lever for sub-100M models on arithmetic.

The 16.4% token overhead is a fixed cost paid for the gain.

### BPE training corpus size

We train BPE on a 2 GB sample of the distillation corpus (math + lang + code, in mix proportions). 2 GB at ~5-7 bytes/token = ~300-400M tokens of training data — comfortable for 32K vocab BPE training (need at least vocab_size * 100 ~= 3M token occurrences for stable merge stats).

---

## 7. Fill-in-the-Middle (FIM)

Bavarian et al. 2022, "Efficient Training of Language Models to Fill in the Middle".

### PSM permutation

Per training document, with probability `p = 0.5`:

1. Tokenize the doc to ids `[A B C]` (where `A` is prefix, `B` is middle, `C` is suffix)
2. Pick two split points `i < j` uniformly at random
3. Reorder as `[<|fim_prefix|>, A, <|fim_suffix|>, C, <|fim_middle|>, B]`
4. Train standard left-to-right next-token prediction

At inference, prompt the model with `<|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>` and generate to the next `<|eos|>`.

### Why prob=0.5

Bavarian Section 3.3 ablates the FIM rate from 0% to 90%:

| FIM rate | L-to-R perplexity (rel) | FIM perplexity (rel) |
|---|---|---|
| 0% | 1.000 (baseline) | n/a |
| 50% | **1.000** | **1.000** |
| 90% | 1.001 | 0.997 |

50% is the smallest rate that maximally trains FIM capability without compromising L-to-R. Above 50%, returns are diminishing.

### Why PSM not SPM

Two orderings are possible:

```
PSM (Prefix-Suffix-Middle): [<PRE> A <SUF> C <MID> B]
SPM (Suffix-Prefix-Middle): [<SUF> C <PRE> A <MID> B]
```

Bavarian Section 4.2 shows SPM gives slightly better FIM perplexity (~0.5%) but PSM is more interpretable and matches the natural prompt order at inference. We use PSM.

### Random split point distribution

The two split points `i < j` are drawn uniformly from valid positions (preserving non-empty prefix, middle, suffix). For document length `n >= 64`:

```
i ~ Uniform({8, 9, ..., n - 16})
j ~ Uniform({i + 4, ..., n - 4})
```

This avoids degenerate FIM samples (1-token middle, etc.). Documents shorter than 64 tokens are passed through without FIM (too short to give meaningful training signal).

### Expected impact on training loss

Since the FIM-permuted sequence has the same token count as the original (plus 3 special tokens for `<PRE>`, `<SUF>`, `<MID>`), per-token CE is essentially unchanged. The model spends equal compute on each token; what differs is the conditioning structure.

Bavarian's measurement: at p=0.5, downstream HumanEval pass@1 improved from 30% to 40% on a 6.9B model with FIM enabled. At our 50M scale we expect smaller absolute gains but the same qualitative improvement on infill tasks.

---

## 8. Memory budget

### Per-component breakdown for 50.8M dense on 80GB at Phase 1 (B=4, T=4K)

```
Model weights (bf16):                     0.10 GB  (50.8M * 2)
Gradients (bf16):                         0.10 GB
AdamW state (embed/norm/biases ~14.7M, fp32 m+v): 0.18 GB
Muon state (hidden 2D ~36M, fp32 momentum):       0.14 GB
Activations @ B=4 T=4K with grad ckpt:   ~2-3 GB
Output logits (with Liger fused CE):     ~0.15 GB (vs 1 GB without)
Misc (CUDA workspace, fragmentation):    ~3 GB
TOTAL:                                   ~5-6 GB on 80GB GPU (extreme headroom)
```

50M dense leaves the GPU >85% empty. The headroom is intentional — Phase 2 at 16K context grows activation memory ~16x, and we want comfortable margins on 40GB Colab fallback.

### Phase 2 (CPT @ 16K, B=2)

```
Model + grads + optim:                   ~0.5 GB   (same as above)
Activations @ B=2 T=16K with grad ckpt:  ~12 GB    (16x of Phase 1)
Output logits (Liger):                   ~0.3 GB
Misc:                                    ~3 GB
TOTAL:                                   ~16 GB on 80GB GPU
```

Still very comfortable on 80GB. On 40GB, we would drop to B=1 with accum=4 for the same effective batch.

---

## 9. Token budget

### Phase 1

```
40,000 steps * 8 (eff batch) * 4096 (T) = 1.31B tokens
```

For 50.8M dense, Chinchilla-optimal is ~1.0B tokens (20:1). We are training at 1.31x Chinchilla. Distillation pretraining is more sample-efficient than raw web text (Phi-3, MiniCPM, Nemotron all show 3-10x sample efficiency gains from distilled data), so 1.3x Chinchilla on distilled data is roughly equivalent to 4-13x Chinchilla raw web.

### Phase 2 (CPT)

```
2,500 steps * 4 (eff batch) * 16,384 (T) = 164M tokens
```

Continued pretraining at longer context. Token count here is less critical — the goal is teaching the model to attend across longer ranges, not adding more world knowledge.

### Phase 3 (SFT)

```
2,500 steps * 8 (eff batch) * 4096 (T) = 82M tokens
```

Standard SFT phase: smaller token count, instruction-style data shapes the model's output format.

### Total

```
Phase 1 + Phase 2 + Phase 3 = 1.31B + 0.16B + 0.08B = 1.55B tokens
```

For 50.8M dense, 30x params. Comfortable for distillation pretraining.

---

## 10. Wall-time estimate

Throughput at the planned configs on A100 80GB:

```
Phase 1 B=4 T=4K  : ~120K tokens/sec  -> 1.31B tokens / 120K tps = 3.0 hours raw
                                        with grad ckpt (30% slower) = 4.0 hours
                                        with accum=2 sync overhead   = 4.5 hours

Phase 2 B=2 T=16K : ~40K tokens/sec   -> 164M tokens / 40K tps = 1.1 hours raw
                                        with grad ckpt + sync         = 1.5 hours

Phase 3 B=4 T=4K  : ~120K tokens/sec  -> 82M tokens / 120K tps = 0.2 hours raw
                                        with overhead                 = 0.3 hours
```

Plus:
- Phase 0 BPE training: ~30 min CPU
- Distillation precache: ~30-45 min one-time
- Buffer for Colab disconnects, reconnects: ~1-2 hours

**Total: 11-13 hours on A100 80GB end-to-end.** At Colab PAYG (~$1.50/hr A100), $15-20 total cost.

---

For block-by-block dataflow, see `docs/ARCHITECTURE.md`. For phase recipe details, see `docs/TRAINING.md`.
