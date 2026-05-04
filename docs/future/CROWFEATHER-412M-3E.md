# Crowfeather-412M-3E (deferred)

A Qwen3-MoE 3-expert plan, parked while we ship the 50M dense baseline first.

**Sibling repo**: [`Crownelius/crowfeather-412m-3e`](https://github.com/Crownelius/crowfeather-412m-3e)

---

## Why deferred

Two reasons made shipping 50M dense first the better starting point:

1. **Vocab cost at 50M scale**: the 412M plan uses Gemma 3's 262K SentencePiece vocab. At hidden=192 (the largest the 50M budget supports with 262K vocab), embeddings alone consume the full 50M. Zero left for transformer layers.

2. **MoE economics at 50M**: 3 experts at ~13M each cannot meaningfully specialize. MoE earns its complexity tax at 100M+ active per expert.

So the 50M baseline gets all the orthogonal modern levers (distillation, GQA, Muon, WSD, FIM, custom 32K BPE) without the MoE complexity. Once the dense recipe is validated end-to-end, the 412M MoE plan resumes with the same data mix and trainer skeleton.

---

## Architecture summary (412M)

| Component | 412M plan |
|---|---|
| Architecture | Qwen3-MoE (3 experts, top-1) |
| Active params | 408M per token |
| Stored params | 748M |
| Hidden | 768 |
| Layers | 24 |
| Q / KV heads | 12 / 4 (GQA 3:1) |
| Head dim | 64 |
| FFN per expert | 3072 SwiGLU |
| Vocab | 262,144 (Gemma 3 SP) |
| Max context | 131,072 (128K) |

The 412M repo contains the full math derivations, architecture deep dive, training recipe, and a tested 80GB limit-test notebook. None of that work is wasted — the optimizer, schedule, distillation pipeline, and Liger Kernel infrastructure all transfer to this 50M build directly.

---

## Resumption criteria

The 412M plan resumes when:

1. Crowfeather-50M-v1 ships and benchmarks beat Shard-40m-v1
2. The 50M FIM capability is verified at inference
3. Compute budget allows ~55h A100 80GB ($70-90 PAYG)

At that point the 50M dense weights become the warm-start for the 412M MoE — the embedding can be swapped (32K -> 262K) by re-tokenizing and projecting, and the dense FFN per layer becomes the seed for one of the 3 experts. (Or fully fresh init; both are reasonable.)
