# Architecture deep dive — Crowfeather-50M-v1

Block-by-block walkthrough of the forward pass, with shapes and parameter counts at every stage.

---

## Top-level dataflow

```
input_ids  [B, T]
    |
    v
embedding (32768 x 448, tied with output)        -> 14.68M params
    |
    v
hidden state  [B, T, 448]
    |
    v
12 x Transformer block  -> 36.13M params
    |
    v
final RMSNorm                                    -> 448 params
    |
    v
lm_head (tied, 448 x 32768)                      -> (shared with embedding)
    |
    v
logits [B, T, 32768]
```

Each transformer block:

```
hidden  [B, T, 448]
    |
    v RMSNorm pre-attn
    |
    v Self-attention (GQA 2:1)
    |
    | residual add -- back to hidden
    v
    |
    v RMSNorm post-attn
    |
    v FFN SwiGLU (single, no MoE)
    |
    | residual add -- back to hidden
    v
hidden  [B, T, 448]
```

---

## Tokenization pipeline (training time)

```
raw_text
    |
    v per_digit_wrap  ("1234" -> "1 2 3 4")
    |
    v fim_permute     (probability 0.5 in Phase 1; identity otherwise)
    |
    | when permuted: doc -> [<|fim_prefix|>] A [<|fim_suffix|>] C [<|fim_middle|>] B
    |
    v BPE encode      (32K vocab, byte-level)
    |
    v packed sequences of length T
    |
    v shifted: input = [t0..tN-1], target = [t1..tN]
```

Per-digit wrap is applied first so the BPE never sees multi-digit numbers. FIM permutation happens at the token-id level after BPE encoding.

---

## Attention block (GQA 2:1)

### Projections

```
Q: linear 448 x 448  (8 query heads x 56 head_dim)             0.20M params
K: linear 448 x 224  (4 KV heads   x 56 head_dim)              0.10M params
V: linear 448 x 224  (4 KV heads   x 56 head_dim)              0.10M params
O: linear 448 x 448  (output projection)                       0.20M params
                                                          -----
                                                          0.60M params per layer
```

Plus Qwen3-native per-head RMS norms on Q and K:

```
q_norm: 56 params  (broadcast across 8 heads)
k_norm: 56 params  (broadcast across 4 KV heads)
```

### Forward shapes

```
hidden [B, T, 448]
  |
  v Q proj
[B, T, 8, 56]                  (8 query heads)
  |
  v K proj
[B, T, 4, 56]                  (4 KV heads)
  |
  v V proj
[B, T, 4, 56]
  |
  v Q-norm and K-norm (RMS per head)
[B, T, 8, 56], [B, T, 4, 56]
  |
  v RoPE on Q and K (rope_theta = 1,000,000)
[B, 8, T, 56]                  (transpose to head-major)
[B, 4, T, 56]
  |
  v repeat_interleave KV across query group (2 queries per KV)
[B, 8, T, 56]
  |
  v scaled dot-prod attention with causal mask
[B, 8, T, 56]
  |
  v concatenate heads
[B, T, 448]
  |
  v O proj
[B, T, 448]
```

### Why GQA 2:1

8 query heads / 4 KV heads = 2 queries per KV. This:
1. Cuts KV cache memory by 50% at inference (see MATH.md section 3)
2. Preserves query expressiveness (all 8 heads have independent Q)
3. The smallest GQA ratio that gives meaningful KV reduction at 8-head models

### Why Q-norm and K-norm

Qwen3 introduced per-head RMS normalization on Q and K *before* RoPE. The math: it stabilizes the attention scale across heads independently, preventing one head from dominating the softmax due to outlier weight magnitudes early in training.

Cost: 56 + 56 = 112 params per layer (negligible). Benefit: better training stability without learning-rate scheduling acrobatics.

### RoPE

Applied to Q and K only (not V). Base frequency 1,000,000 chosen for long-context stability up to 16K. RoPE theta=1M extends gracefully past 100K positions if we ever need to extend (and the math audit verified this on the 412M plan).

---

## FFN block (single SwiGLU, no MoE)

### Structure

```
h [B, T, 448]
  |
  v gate_proj 448 x 1792                      0.80M params
[B, T, 1792]
  |
  v silu activation
[B, T, 1792]
  |
  v multiply with up_proj(h) [B, T, 1792]     0.80M params
[B, T, 1792]
  |
  v down_proj 1792 x 448                      0.80M params
[B, T, 448]
```

Total per FFN: 2.41M params.

### Why no MoE at 50M

Three experts at ~13M each cannot meaningfully specialize. The Crowfeather-412M-3E sibling architecture *does* use 3 experts, but at 408M active per token, each expert is ~136M — large enough to develop a real bias (math expert, language expert, code expert). At 50M dense, the single FFN learns all three competencies through distillation alone.

### Per-block parameter total

```
Attention:    0.60M
Q/K norms:    0.0001M
FFN:          2.41M
Pre-attn RMS: 448 params
Post-attn RMS: 448 params
              -----
TOTAL:        ~3.01M params per layer
```

### Whole model

```
Embedding (tied):           14.68M
12 layers * 3.01M:          36.13M
Final norm:                 448
                            -----
TOTAL:                      50.82M params (dense)
```

Dense means stored = active. Every token at every step uses every parameter.

---

## Per-digit input wrap

Applied at the data loader level, not the tokenizer level:

```python
import re
_DIGIT_RUN_RE = re.compile(r'\d{2,}')
def per_digit_wrap(text):
    return _DIGIT_RUN_RE.sub(lambda m: ' '.join(m.group()), text)
```

Effect: `"1234 + 5678 = 6912"` becomes `"1 2 3 4 + 5 6 7 8 = 6 9 1 2"`.

The BPE tokenizer (also trained on per-digit-wrapped corpus) tokenizes each digit independently because they're separated by whitespace. Combined with byte-level pre-tokenization, this forces uniform per-digit treatment across the entire pipeline.

Cost: ~16% more tokens at the planned data mix (see MATH.md section 6).

Benefit: arithmetic learning becomes tractable. Without per-digit wrap, "9000" might be one token but "8999" is three tokens — making the model learn weird conditional probabilities for adjacent integers.

---

## FIM permutation (training-time data augmentation)

Applied at the token-id level, after BPE encoding, before packing into sequences.

```python
def fim_permute(ids, fim_pre, fim_suf, fim_mid, prob, rng):
    if rng.random() >= prob:
        return ids
    n = len(ids)
    if n < 64:
        return ids
    a = rng.randint(8, n - 16)
    b = rng.randint(a + 4, n - 4)
    return [fim_pre] + ids[:a] + [fim_suf] + ids[b:] + [fim_mid] + ids[a:b]
```

For each document independently:
- With probability 0.5 (Phase 1 only), pick two random split points and reorder PSM
- Otherwise pass through unchanged

Phase 2 and Phase 3 use prob=0.0 (no FIM): CPT and SFT data benefit more from preserved L-to-R structure.

### Special tokens injected

Three new tokens appear in the FIM-permuted sequences:

```
<|fim_prefix|>    ID 4   marks: "what follows is the prefix"
<|fim_suffix|>    ID 5   marks: "what follows is the suffix"
<|fim_middle|>    ID 6   marks: "what follows is the middle (target during training)"
```

The model learns to predict the token after `<|fim_middle|>` conditioned on having seen `<|fim_prefix|> A <|fim_suffix|> C <|fim_middle|>`. At inference, this becomes the natural prompt for infill: paste prefix and suffix surrounded by the right markers, generate the middle.

### Inference example

```
prompt:   <|fim_prefix|>def add(a, b):<|fim_suffix|>    return result<|fim_middle|>
generate: \n    result = a + b\n
```

Generation continues until `<|eos|>`. The full document is assembled as `prefix + middle + suffix`.

---

## Final norm and lm_head

```
hidden [B, T, 448]
  |
  v RMSNorm (final)
[B, T, 448]
  |
  v lm_head (tied with input embedding, 448 x 32768)
logits [B, T, 32768]
```

The `lm_head` is the same weight tensor as the input embedding — they share storage via `tie_word_embeddings=True`. This saves 14.68M parameters (the size of an untied output head).

### Output dtype

Logits are computed in bf16 by default. For loss computation, Liger Kernel chunks them along the vocab dimension to avoid materializing the full `B x T x 32768` tensor.

---

## Loss computation

```
total_loss = CE_loss + z_loss

where:
  CE_loss = standard cross-entropy on next-token targets, computed via Liger fused kernel
  z_loss  = z_loss_coef * mean(logsumexp(logits)^2)
```

No aux loss (no MoE router to balance). No expert-load monitoring code path.

Coefficients:
- `z_loss_coef = 1e-4` (see MATH.md section 4)

---

## What this architecture deliberately does NOT have

| Feature | Alternative | Why we don't have it |
|---|---|---|
| MoE / mixture of experts | DeepSeek V3, Qwen3-MoE | At 50M, experts cannot specialize meaningfully — see the Crowfeather-412M-3E sibling for the MoE plan |
| MTP (Multi-Token Prediction) head | DeepSeek V3/V4 | Adds complexity, no inference benefit for our llama.cpp target |
| SwiGLU clamping | DeepSeek V4 | Standard SiLU + multiplicative gate; no clamps needed at 50M scale |
| Per-Layer Embeddings (PLE) | Gemma 4 | Defer until Gemma 4 llama.cpp support stabilizes |
| Mixture of Depths (MoD) | Google | Adds routing-style complexity that's only beneficial at >100M; our depth is 12 |
| Memory module (SpinorApollonian, AHN) | FANT line | No GGUF support; explicitly removed for clean Qwen3-compatible build |
| SleepGate | FANT line | Removed; not relevant for dense architecture |
| Dynamic depth / early-exit | Layer Skip, MoR | Defer; our 12 layers are already shallow |

This is **a small Qwen3-style dense transformer with custom 32K BPE, distilled from frontier teachers, optimized with Muon + AdamW, with FIM data augmentation**. Nothing original to FANT line. Nothing experimental at the architectural level. The novelty is in the *combination*: dense at 50M with all the modern levers stacked.

---

For training recipe details, see `TRAINING.md`. For math derivations, see `MATH.md`.
