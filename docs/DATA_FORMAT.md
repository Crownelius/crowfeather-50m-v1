# Distillation data format — Crowfeather-50M-v1

Every line of every JSONL file in `distill_data/` follows the same unified schema. The trainer reads only `text`; the other fields are for filtering, analysis, and reproducibility.

---

## Record schema

```json
{
  "text": "<chat-formatted with our 18 reserved special tokens>",
  "source_dataset": "AI-MO/NuminaMath-CoT",
  "domain": "math",
  "format": "chat_with_thinking",
  "has_thinking": true,
  "tokens_est": 1234,
  "metadata": {
    "idx": 42,
    "...": "..."
  }
}
```

| Field | Type | Description |
|---|---|---|
| `text` | str | The full training string with reserved special tokens. **This is what the model sees.** |
| `source_dataset` | str | HuggingFace ID (or `Anthropic/opus-4.6-traces` for Drive sources). For multi-config datasets includes the subset, e.g. `open-r1/Mixture-of-Thoughts:math`. |
| `domain` | str | One of `math`, `lang`, `code`. Determines which combined `.jsonl` the record lands in. |
| `format` | str | One of `chat_with_thinking` (has `<\|think\|>...</\|think\|>` block), `chat` (user+assistant), `qa` (problem+solution), `raw` (assistant-only). |
| `has_thinking` | bool | Whether the `text` contains a `<\|think\|>...</\|think\|>` block. |
| `tokens_est` | int | Approximate token count (`len(text) // 4`). Useful for filtering very long or very short docs. |
| `metadata` | dict | Dataset-specific extras (row index, original field names, optional quality flags). Free-form. |

---

## Text format — the chat structure

```
<|user|>
{user content}
<|assistant|>
<|think|>
{reasoning trace if present}
</|think|>
{final answer}
<|eos|>
```

Variants:
- **No thinking trace** — drop the `<|think|>...</|think|>` block:
  ```
  <|user|>
  {user content}
  <|assistant|>
  {response}
  <|eos|>
  ```
- **System prompt present** — prepend to user:
  ```
  <|user|>
  [System]: {system content}

  {user content}
  <|assistant|>
  ...
  ```
- **No user prompt available** (raw assistant data, e.g. some Sonnet/Opus dumps):
  ```
  <|assistant|>
  {raw text}
  <|eos|>
  ```

The trainer (`train_dense.py`) appends `<|eos|>` between concatenated docs at packing time. Records already end with `<|eos|>` for explicit doc boundary clarity; the duplicate is one wasted token per doc and is harmless.

---

## Per-source mapping

| Source dataset | HF ID (config) | Domain | Format | Notes |
|---|---|---|---|---|
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | math | `qa` / `chat_with_thinking` | Splits `<think>` blocks if present in `solution` |
| MetaMathQA | `meta-math/MetaMathQA` | math | `qa` | `query` + `response` |
| R1-math | `open-r1/Mixture-of-Thoughts` (`math`) | math | `chat_with_thinking` | DeepSeek R1 traces; `<think>` always present |
| R1-science | `open-r1/Mixture-of-Thoughts` (`science`) | lang | `chat_with_thinking` | Same |
| R1-code | `open-r1/Mixture-of-Thoughts` (`code`) | code | `chat_with_thinking` | Same |
| Sonnet 4.6 | `Roman1111111/claude-sonnet-4.6-120000x` | lang | `chat` / `chat_with_thinking` | Tries `messages` then `prompt`/`output` then raw `text` |
| Opus 4.6 | local Drive JSONL | lang | `chat` / `raw` | User-curated; lives at `{DRIVE_ROOT}/distill_data/opus_4_6.jsonl` |

The R1 adapter (`stream_r1_subset`) is robust to schema drift across the three Mixture-of-Thoughts configs: tries `messages` first, then `prompt`/`completion`, then raw `text`. If a future dataset rev breaks the adapter, `scripts/diagnose_datasets.py` prints the actual schema for quick debugging.

---

## Per-domain combined files

After per-source download, the precache builds:

| File | Sources | Mix proportion (8 GB total budget) |
|---|---|---|
| `math.jsonl` | numinamath (40%), metamathqa (30%), r1_math (30%) | 30% (~2400 MB) |
| `lang.jsonl` | sonnet (55%), r1_science (30%), opus (15%) | 40% (~3200 MB) |
| `code.jsonl` | r1_code (100%) | 30% (~2400 MB) |

The trainer's `make_mixed` reads from these three combined files at the configured weights (math 0.30 / lang 0.40 / code 0.30).

---

## Detecting and migrating from the old format

The old precache wrote `{"text": "..."}` with no metadata. To detect old-format records, the notebook checks for the `source_dataset` field on the first line. If absent, the precache cell wipes the cache and re-downloads in unified format. There is no in-place converter — re-download is the canonical path because some old records were truncated by the schema bugs the new format exists to fix.

---

## Diagnostic tool

```bash
python scripts/diagnose_datasets.py
python scripts/diagnose_datasets.py --dataset sonnet  # one at a time
```

Prints the schema (field names, types, content previews) of the first 1-2 records of each registered dataset. ~30-60 sec total. Run this before bulk download if you suspect schema drift.
