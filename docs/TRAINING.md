# Training recipes — Crowfeather-50M-v1

Four phases (one of which is tokenizer training). All resume cleanly. All log to wandb. All save HF-format checkpoints to Drive.

---

## Phase 0 — BPE tokenizer training

| Parameter | Value | Notes |
|---|---|---|
| Vocab size | 32,768 | 18 special + 256 byte alphabet + 32,494 BPE merges |
| Min frequency | 2 | merge created only if pair appears >= 2 times |
| Pre-tokenizer | ByteLevel (GPT-2 regex) | guarantees byte fallback |
| Normalizer | NFC | Unicode normalization |
| Decoder | ByteLevel | inverse of pre-tokenizer |
| Corpus | 2 GB sample of distillation mix | math 30% / lang 40% / code 30% |
| Per-digit wrap | applied before BPE training | so merges never produce multi-digit numbers |

### Wall time

~30 minutes CPU.

### Output

```
runs/tokenizer/tokenizer.json
runs/tokenizer/tokenizer_config.json
runs/tokenizer/special_tokens_map.json
runs/tokenizer/added_tokens.json
```

### Resume logic

If `runs/tokenizer/tokenizer.json` exists, Phase 0 is skipped and the existing tokenizer is loaded.

---

## Phase 1 — Pretrain @ 4K context

| Parameter | Value | Notes |
|---|---|---|
| Steps | 40,000 | |
| Sequence length | 4,096 | |
| Batch size | 4 | same on 40GB and 80GB after limit-test (peak <2 GB) |
| Gradient accumulation | 2 | Effective batch = 8 |
| Peak LR | 3e-4 | Slightly higher than 412M's 2e-4 because smaller model |
| Min LR | 3e-5 | |
| Warmup fraction | 0.015 (1.5%) | |
| Decay fraction | 0.20 (20%) | |
| Schedule | WSD with sqrt cooldown | |
| z-loss coefficient | 1e-4 | With gradients (see MATH.md section 4) |
| FIM rate | **0.5** | PSM permutation per document |
| Grad clip | 1.0 | |
| Mixed precision | bf16 | |
| Gradient checkpointing | ON | comfortable headroom |
| Liger Kernel | ON if available | optional at 50M |
| Checkpoint every | 2,500 steps | |
| Logging interval | 20 steps | |

### Wall time

~6-7 hours on A100 80GB (limit-test 2026-05-04 measured ~28K tokens/sec at this config; 40K * 2 / 28K * 4096 ~= 23K seconds = ~6.5h).

### Resume logic

The launch cell detects existing step checkpoints in `runs/phase1/`. If present, resumes from the latest. Otherwise loads the random-initialized model from `runs/init/` (saved by `build_init.py`).

### Initialization

Random init only via Qwen3Config defaults (HuggingFace). No grafting from any other model. Embeddings learn from scratch on the custom 32K BPE.

---

## Phase 2 — Continued Pretrain @ 16K context

| Parameter | Value | Notes |
|---|---|---|
| Steps | 2,500 | |
| Sequence length | 16,384 | matches `max_position_embeddings` |
| Batch size | 2 | bumped from B=1 after limit-test stress test passed (peak 1.5 GB) |
| Gradient accumulation | 2 | Effective batch = 4 |
| Peak LR | 6e-5 | 20% of pretrain LR (standard for context extension) |
| Min LR | 6e-6 | |
| Warmup fraction | 0.05 (5%) | |
| Decay fraction | 0.30 (30%) | |
| FIM rate | **0.0** | no FIM in CPT |
| All other params | same as Phase 1 | |

### Why no FIM in Phase 2

CPT teaches the model to attend across longer ranges. Long-document data (textbook chapters, multi-turn dialogues, code files) benefits from preserved L-to-R structure. FIM permutation breaks the natural document order in a way that conflicts with the goal of the phase.

### Why 16K not 32K

Our `max_position_embeddings = 16384`. The 412M sibling can extend to 128K because RoPE theta=1M handles it; but at 50M, attention quality degrades fast at very long contexts (limited compute per token). 16K is the sweet spot for this scale.

If you need longer context later, RoPE theta extension (NTK or YaRN) at Phase 4 is the path.

### Resume logic

Resumes from `runs/phase1/final/`. Skipped if `runs/phase2/final/` exists.

---

## Phase 3 — SFT @ 4K context

| Parameter | Value | Notes |
|---|---|---|
| Steps | 2,500 | |
| Sequence length | 4,096 | |
| Batch size | 4 | same on 40GB and 80GB |
| Gradient accumulation | 2 | Effective batch = 8 |
| Peak LR | 4e-5 | 13% of pretrain LR |
| Min LR | 4e-6 | |
| Warmup fraction | 0.05 (5%) | |
| Decay fraction | 0.30 (30%) | |
| FIM rate | **0.0** | no FIM in SFT |
| All other params | same as Phase 1 | |

### Why this LR is lower

SFT shapes the model's output style on instruction-format data. Higher LR risks "unlearning" the broad capabilities from pretrain. 4e-5 is conservative.

### Why no FIM in Phase 3

Instruction-format data has natural conversation order. Permuting it would break the user/assistant turn structure.

### Resume logic

Tries `runs/phase2/final/` first. Falls back to `runs/phase1/final/` if Phase 2 was skipped.

---

## Cross-phase shared infrastructure

### Optimizer split (`muon.py`)

```python
def split_params_for_muon(model):
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_router = '.mlp.gate' in name and 'gate_proj' not in name
        is_2d_hidden = (
            p.ndim >= 2
            and 'embed' not in name.lower()
            and 'norm' not in name.lower()
            and 'bias' not in name.lower()
            and not is_router
        )
        (muon_params if is_2d_hidden else adamw_params).append(p)
    return muon_params, adamw_params
```

For dense Qwen3, `is_router` never matches (no `.mlp.gate` outside of `gate_proj`). So:
- Muon: all 2D weights in attention (q/k/v/o_proj) and FFN (gate/up/down_proj). ~36M params.
- AdamW: embedding, all RMSNorms, q_norm/k_norm. ~14.7M params.

Muon handles ~71% of params. AdamW handles 29% (mostly embedding).

### LR schedule (WSD)

```python
def wsd_lr(step, total, peak, mn, warmup_frac=0.015, decay_frac=0.20):
    warmup = int(total * warmup_frac)
    decay_start = int(total * (1.0 - decay_frac))
    if step < warmup:
        return mn + (peak - mn) * (step / max(warmup, 1))
    if step < decay_start:
        return peak
    progress = (step - decay_start) / max(total - decay_start, 1)
    return mn + (peak - mn) * (1.0 - math.sqrt(progress))
```

Sqrt decay (not linear, not cosine) is the MiniCPM 2024 finding — keeps LR high through 50% of decay, then drops fast at the end.

### Beta2 ramp

```python
def cooldown_beta2(step, total, b2_start=0.95, b2_end=0.97, warmup_frac=0.015):
    warmup = int(total * warmup_frac)
    if step < warmup:
        return b2_start
    progress = (step - warmup) / max(total - warmup, 1)
    return b2_start + (b2_end - b2_start) * min(max(progress, 0.0), 1.0)
```

Beta2 is the AdamW second-moment EMA. Ramping from 0.95 to 0.97 reduces gradient variance impact during late training. Stability improvement, not a performance lever.

---

## FIM data augmentation details

Applied per-document, after tokenization, with probability `fim_rate`:

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

### Phase rates

| Phase | FIM rate | Why |
|---|---|---|
| Phase 1 (pretrain) | 0.5 | maximally trains FIM capability without compromising L-to-R (Bavarian 2022) |
| Phase 2 (CPT) | 0.0 | long-document attention extension; preserve L-to-R structure |
| Phase 3 (SFT) | 0.0 | conversation-format data; preserve turn structure |

### Documents shorter than 64 tokens

Pass through unchanged. Too short to give meaningful infill training signal.

### What gets FIM'd

In Phase 1, every document — math, language, code — gets the same 50% FIM probability. We do NOT FIM only code (which is what some implementations do): the goal is general infilling capability, useful for prose paragraph completion as well as code completion.

---

## Distillation data mix

Same mix across all phases (Phase 2 just operates on longer contexts):

| Domain | Sources | Per-domain weight | Mix weight |
|---|---|---|---|
| Math | NuminaMath-CoT (40%), MetaMathQA (30%), R1 math (30%) | combined to math.jsonl | 30% |
| Language | Sonnet 4.6 (55%), R1 science (30%), Opus 4.6 (15%) | combined to lang.jsonl | 40% |
| Code | R1 code subset (100%) | code.jsonl | 30% |

The trainer (`train_dense.py`) randomly selects one of these three domain JSONLs per document according to the mix weights. Per-digit wrap is applied to all text before tokenization. FIM permutation is applied after tokenization, per document.

### Why these weights

- 40% language (the largest slice) because that's the foundational capability
- 30% math + 30% code (intentionally large) because these are the capability gaps the predecessor (Shard-40m-v1) hit hardest
- Same proportions as the 412M sibling for direct cross-scale comparison

---

## Optional Phase 4 (not in this notebook)

If Phase 1-3 produce a competent base model, future iterations may add:

- **DPO/SimPO alignment** on preference pairs
- **GSPO sequence-level RL** for math reasoning
- **Self-consistency decoding** at inference (no training change needed)
- **RoPE extension to 64K-128K** via NTK-aware scaling

These are deferred until the base model exists and we know what it can and can't do.

---

For architecture deep dive, see `ARCHITECTURE.md`. For math derivations, see `MATH.md`.
