# Optimal training format — Crowfeather-50M-v1

This document is the **canonical spec** for what training records should look like. It overrides any conflicting guidance elsewhere.

The format is designed for a 50M-parameter dense model with:
- 32K Byte-Level BPE vocabulary
- 18 reserved special tokens at IDs 0-17
- 4096-token training context (Phase 1), 16K (Phase 2)
- bf16 mixed precision, Liger Kernel fused CE
- FIM data augmentation at 50% rate during pretrain
- Per-digit number wrap applied at training time

Every byte of training format overhead reduces effective sequence length, so the format is **minimal** by design. ChatML and Llama-style formats are explicitly rejected because they waste 3-5 tokens per role boundary.

---

## Record schema (JSONL line)

```json
{
  "text":           "<training text — see Text Formats below>",
  "source_dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
  "domain":         "web|math|lang|code",
  "format":         "raw|chat|chat_with_thinking|qa",
  "has_thinking":   true | false,
  "tokens_est":     <int — len(text) // 4>,
  "quality_score":  <float 0-1, optional, from post-hoc QA>,
  "metadata":       {<source-specific>}
}
```

The trainer reads only `text`. Metadata enables post-hoc filtering, deduplication, and source-balanced sampling without re-downloading.

---

## Text formats (exactly four)

### 1. `raw` — web domain only

```
{document text}
```

**No special tokens.** No `<|user|>`, no `<|eos|>`, no markup. Just the cleaned document. The trainer adds `<|eos|>` at packing time between docs.

Reserved for FineWeb-Edu and any other foundation-pretrain web text.

---

### 2. `chat_with_thinking` — reasoning sources with explicit reasoning

```
<|user|>
{user prompt}
<|assistant|>
<|think|>
{concise reasoning trace, 100-2000 tokens preferred}
</|think|>
{final answer}
<|eos|>
```

This is the structure that trains the model's think-then-answer behavior. The 4 reserved tokens (`<|user|>`, `<|assistant|>`, `<|think|>`, `</|think|>`, `<|eos|>`) act as unambiguous boundaries the model latches onto early in training.

**Required content:**
- `{user prompt}` is non-empty and is the question/task.
- `{reasoning trace}` is non-empty — if no reasoning is available, drop the `<|think|>...</|think|>` block and use format `chat` instead.
- `{final answer}` is non-empty and is the answer/solution to the prompt.

---

### 3. `chat` — chat without an explicit reasoning block

```
<|user|>
{user prompt}
<|assistant|>
{response}
<|eos|>
```

For genuinely-no-thinking sources (basic Q&A, instruction-following without reasoning).

---

### 4. `qa` — alias for chat

Functionally identical to `chat`. Distinct `format` value lets us track source provenance (e.g. NuminaMath records were originally `problem`/`solution` pairs without a chat structure; the data is converted to chat at precache time but tagged `qa` so post-hoc filters can find them).

---

## Hard rules

1. **Only the 18 reserved tokens.** No ChatML, no Llama, no OpenAI assistant format. The BPE was trained with these tokens reserved at IDs 0-17; any other role markers waste tokens.
2. **Web records: zero special tokens.** Raw text only. Mixing chat tokens into web text would train the model to expect chat structure on continuation tasks.
3. **Trailing `<|eos|>` is included** in every chat-format record. The trainer ALSO adds an EOS between concatenated docs at packing time; the resulting double-EOS is one wasted token per doc and is harmless. Including it makes the record self-contained for inspection.
4. **Newlines are `\n` (LF)**, not `\r\n` (CRLF). Code that reads JSONL on Windows must not introduce CRLF.
5. **Encoding is UTF-8 NFC** (Unicode normalization form C). The BPE was trained with NFC normalization; mismatched normalization leaks bytes.
6. **Per-digit wrap is NOT applied here.** That happens at training time via a regex applied to `text` before tokenization. Training records keep numbers as-is.
7. **Length cap per record: 8K tokens.** Records longer than 8K should be truncated to a coherent ≤4K-token window (keep the highest-information part — e.g. for chat, keep the user prompt + start of assistant; for web, keep one self-contained paragraph chunk). This caps any single record's contribution to a single training sequence.
8. **`<|think|>...</|think|>` is never empty.** If reasoning is missing, downgrade format from `chat_with_thinking` to `chat`.

---

## Quality criteria (Sonnet enforces these post-precache)

**Drop the record entirely if:**
- The text is gibberish, machine-generated noise, or non-English with no `language` flag
- The record is a near-duplicate of an earlier record (>90% n-gram overlap on first 200 chars)
- HTML/markdown clutter dominates the content (more nav/header bytes than prose)
- Encoding artifacts make the text unreadable (unfixed mojibake, double-encoded UTF-8)
- The "reasoning" is just the question repeated, or the answer with no derivation
- ASCII art, banner images, or repetitive symbol blocks exceed 30 lines
- The record is unambiguously an advertisement, SEO spam, or auto-generated boilerplate

**Keep + clean if:**
- Coherent English prose
- Well-formed reasoning that would be useful for the model to imitate
- Clear Q&A structure (even if needs reformatting into our chat format)
- Real code with comments, not just compiler output or stack traces

**Cleaning operations:**
- Strip HTML tags / markdown navigation / footer boilerplate
- Fix mojibake when correctable
- Collapse repetitive whitespace (3+ blank lines → 1)
- Remove URL trackers from inline links
- Trim obviously-injected ad text
- Truncate to 4K tokens (preserve highest-value section)

---

## Why this format is optimal for 50M

1. **Single-token role boundaries.** Each of `<|user|>`, `<|assistant|>`, `<|think|>`, `</|think|>`, `<|eos|>` tokenizes to exactly 1 BPE token. ChatML's `<|im_start|>user\n` is 4 tokens. Over a 4K context, our format saves ~50-100 tokens per typical chat doc — equivalent to giving the model a 1-2% larger effective context.
2. **Mixed-mode training.** Raw web text (no special tokens) and chat (special tokens) coexist cleanly. The chat tokens act as a switch: when `<|user|>` appears in the prompt, the model knows to follow chat structure; otherwise it continues fluently. No mode-flag needed at inference.
3. **FIM compatibility.** The Bavarian 2022 PSM permutation wraps with `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` — also single-token markers. FIM can be applied to any of the four format types without conflict.
4. **Inspectable.** Each record's `text` is a self-contained string a human can read and verify. No nested JSON inside `text`, no multi-key reconstruction.
5. **Forward-compatible with chat templates.** When we build a HuggingFace `tokenizer_config.json` chat template later (for `apply_chat_template`), it directly mirrors this format — no conversion at inference.

---

## Sonnet conversion prompt (reference)

The conversion script (`scripts/sonnet_convert.py`) uses the following system prompt to enforce the format above. Copy if you build your own conversion pipeline.

```
You are converting raw training records into the optimal format for Crowfeather-50M-v1, a 50M-parameter language model. The model has 18 reserved special tokens; only these may appear as role markers:

  <|user|>  <|assistant|>  <|system|>  <|tool|>
  <|think|>  </|think|>
  <|tool_call|>  </|tool_call|>  <|tool_response|>  </|tool_response|>
  <|fim_prefix|>  <|fim_suffix|>  <|fim_middle|>  <|fim_pad|>
  <|pad|>  <|bos|>  <|eos|>  <|unk|>

Output exactly the cleaned text, no JSON wrapper, no commentary.

If the record's domain is "web", output ONLY the cleaned document text — no special tokens at all.

If the record is a chat with reasoning, output:
<|user|>
{user prompt}
<|assistant|>
<|think|>
{concise reasoning, 100-2000 tokens, preserves key derivation steps}
</|think|>
{final answer}
<|eos|>

If the record is chat without reasoning, output:
<|user|>
{user prompt}
<|assistant|>
{response}
<|eos|>

Cleanup mandate: strip HTML/markdown clutter, fix mojibake, remove ads/SEO/banners, collapse repetitive whitespace, truncate to ~4K tokens preserving the highest-value section. Drop the record entirely (output the literal token SKIP and nothing else) if the input is gibberish, near-duplicate boilerplate, untrainable noise, or has no coherent reasoning when reasoning is required.

Never invent content. If you cannot extract a clean version that preserves the original meaning, output SKIP.
```

---

## Recommended workflow: deterministic conversion + Sonnet verification

The cheapest path to high-quality data is:

1. **Deterministic conversion** (free, fast). `scripts/precache_distill.py` already enforces our format at adapter level — every record comes out in unified schema with the right tokens. ~5-10 GB/hour throughput.

2. **Sonnet verification** (~$25 per 5K-row check). `scripts/sonnet_verify.py` samples N rows stratified across `web.jsonl` / `math.jsonl` / `lang.jsonl` / `code.jsonl`, sends each to Sonnet 4.6 with a strict QA prompt, and aggregates pass/fail/borderline rates plus a top-10 issue-tag breakdown.

3. **Iterate on the deterministic step** (free). Sonnet's most-common issue tags point you at specific bugs in the precache adapters: `mojibake` → add `ftfy.fix_text` to the adapter; `empty_thinking` → tighten the `chat_with_thinking` filter; `boilerplate` → strip per-source headers/footers; etc.

4. **Re-verify** until pass rate >=90%. Then train on the deterministic output. Total verification cost across 3-5 iterations: ~$75-150 vs ~$70K for full Sonnet conversion.

**When to escalate to full Sonnet conversion** (`scripts/sonnet_convert.py`): only if the deterministic step's pass rate plateaus below 80% even after iteration. In practice this happens for low-quality user-curated dumps (badly-scraped web, unverified Sonnet/Opus traces). For HF-hosted datasets (FineWeb-Edu, NuminaMath, R1, Kimi K2.5, MetaMathQA) the deterministic adapter usually clears 90%+ on first pass.

### Issue-tag → adapter-fix mapping

| Sonnet issue tag | What it means | Fix in precache adapter |
|---|---|---|
| `mojibake` | UTF-8 double-encoded or corrupt | Add `ftfy.fix_text(s)` before yielding |
| `broken_html` | Tags/entities in supposedly-clean text | Add `bs4.BeautifulSoup(s, 'lxml').get_text()` or stricter regex |
| `empty_thinking` | `<\|think\|></\|think\|>` with no content | Drop record or downgrade format to `chat` |
| `wrong_format` | Web record has chat tokens, or vice-versa | Bug in adapter — assert format matches domain |
| `untrainable_noise` | Random characters, encoding garbage | Add a Shannon-entropy filter to the adapter |
| `ad_text` | "Subscribe now", "Click here", SEO spam | Source-specific blocklist (mostly for web/Opus) |
| `boilerplate` | Headers, footers, nav menus | Per-source aggressive trim of first/last N lines |
| `near_duplicate` | Same content, different wording | Add MinHash dedup pass after precache |
| `truncated_midsentence` | Cut off in the middle of a token/word | Adapter dropping 8K-char clips on word boundary |
| `ascii_art` | Banner art, table-of-contents bars | Drop records with >30 lines of repeating chars |
| `non_english` | Foreign-language doc when we want English | Add `language_score` filter (FineWeb-Edu has this in metadata) |
| `code_only_no_explanation` | Pure code dump in `domain=code` reasoning | Drop records where assistant content is `^\s*```` |
| `thinking_just_repeats_question` | Reasoning lacks substance | Drop if `<\|think\|>` content is <100 chars or >70% overlap with prompt |

---

## Format conformance check

The converter writes a per-row metadata field `metadata.format_validated: true` if the output passes these checks:

```python
def validate(text: str, format: str) -> bool:
    if format == 'raw':
        # Web records must NOT contain any reserved special tokens
        for tok in ['<|user|>', '<|assistant|>', '<|think|>', '</|think|>',
                    '<|eos|>', '<|bos|>', '<|fim_prefix|>']:
            if tok in text:
                return False
        return len(text) >= 100

    if format == 'chat_with_thinking':
        return (
            text.startswith('<|user|>')
            and '<|assistant|>' in text
            and '<|think|>' in text
            and '</|think|>' in text
            and text.rstrip().endswith('<|eos|>')
            and text.find('<|user|>') < text.find('<|assistant|>')
            < text.find('<|think|>') < text.find('</|think|>')
        )

    if format in ('chat', 'qa'):
        return (
            text.startswith('<|user|>')
            and '<|assistant|>' in text
            and text.rstrip().endswith('<|eos|>')
            and '<|think|>' not in text  # downgraded from chat_with_thinking
        )

    return False
```

Records that fail the conformance check are written to a `*.invalid.jsonl` sidecar file for manual review.
