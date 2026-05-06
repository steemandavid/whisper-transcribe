# Code Fixes #3 — Session 2026-05-06 (REVIEW3.md corrections)

Fixes applied from the third code review (REVIEW3.md). REVIEW3 was a "PASS WITH
NOTES" verdict identifying 4 major and 12 minor items. All must-fix and should-fix
items are addressed below.

---

## Must-fix items (all fixed)

### Fix #1. Revert glossary bonus from additive to multiplicative

**Source**: REVIEW3 Deviation #4 / NEW-P1-2 (self-inflicted by REVIEW2)
**File**: `pipeline/rover.py:112-127`
**FSD Reference**: §2 stage 2 — "Multiply by glossary weight"

REVIEW2 recommended switching from `prob *= (1 + weight)` to `prob += weight` so
the tie-break cascade's step 3 (glossary membership) could actually fire. REVIEW3
correctly identified this as a spec violation and noted a worse side effect: with
`weight = 1.0` for brand entries, the additive bonus (`prob + 1.0`) made brand
glossary hits **always** win regardless of confidence — a 0.10 brand candidate
(score 1.10) beats a 0.99 non-brand candidate (score 0.99).

**Change**: Reverted to `prob *= (1.0 + weight)`. Added a comment explaining that
the cascade's step 3 (`not is_glossary`) is *correctly redundant* — the
multiplicative score already encodes glossary preference, so the tiebreak only
fires in the mathematically degenerate case, which is intentional:

```python
# FSD §2 stage 2: "Multiply by glossary weight."
# The cascade's step 3 (glossary membership) is correctly redundant here:
# the multiplicative score already encodes glossary preference, so
# the tiebreak only fires on the rare case of identical multiplicative
# scores — which is intentional.
prob *= (1.0 + weight)
```

### Fix #2. Post-correct batch failures are no longer silent

**Source**: REVIEW3 §6.2
**File**: `pipeline/postcorrect.py:413`

The `run()` function's batch loop caught all exceptions and silently extended with
verbatim segments. On failure (quota exhausted, malformed JSON, Z.ai validation
rejection), the user saw zero corrections and no explanation.

**Change**: Added stderr logging to the except block:

```python
except Exception as e:
    print(f"Post-correct batch {i // _BATCH_SIZE}: {type(e).__name__}: {e}", file=sys.stderr)
    all_corrected.extend(batch)
```

This also addresses §4.4's concern that a `cache_control` validation rejection
would silently disable cloud correction — the user now sees the error message.

### Fix #3. System prompt restructured; `cache_control` removed

**Source**: REVIEW3 Deviation #2
**File**: `pipeline/postcorrect.py` (both `OllamaCorrector` and `ZaiCorrector`)

Three problems identified:

1. `cache_control` is Anthropic-API syntax, not OpenAI. The OpenAI SDK 1.54.0
   (pinned in Dockerfile) doesn't support it. Unknown fields may be silently
   dropped (no caching) or rejected (400 error → all corrections fail).
2. Even if accepted, `cache_control` at the message level doesn't enable prefix
   caching — the Anthropic semantic requires it on specific content blocks.
3. The long system prompt (template + full glossary) was concatenated into the
   user message, while the actual system role sent only a one-liner. This meant
   the expensive-to-cache content was in the per-batch-different user message,
   providing no caching benefit even if the mechanism worked.

**Change**: For both backends:

- Moved the full system prompt (glossary, rules, language) to the actual `system`
  role message, appended with "Respond only in valid JSON."
- The `user` message now contains only the batch-specific segment text
  (`"Transcript segments to correct:\n{batch_text}"`).
- Removed all `cache_control` keys from message dicts entirely.

This is structurally correct (system prompt in system role, batch data in user
role) even without caching. If Z.ai adds caching support later, the system
message will be a stable prefix across batches, making caching trivially
re-enableable.

### Fix #4. Z.ai token usage accumulated across batches

**Source**: REVIEW3 Deviation #3
**File**: `pipeline/postcorrect.py:196,222-233,415-418`

`ZaiCorrector.last_usage` was overwritten on each `correct()` call. For a
48-segment recording with 2 batches, only the last batch's usage was recorded.
The sidecar `cleaned_usage.json` could not satisfy the verification check
("~3–8 k input tokens per 30 min of audio").

**Change**: `last_usage` now accumulates:

```python
if self.last_usage is None:
    self.last_usage = {
        "model": self.model,
        "batches": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }
self.last_usage["batches"] += 1
self.last_usage["total_prompt_tokens"] += batch_usage["prompt_tokens"]
self.last_usage["total_completion_tokens"] += batch_usage["completion_tokens"]
```

The sidecar JSON now reflects the entire run. Per-batch breakdown is logged to
stderr as before.

---

## Should-fix items (all fixed)

### Fix #5. Unreachable mixed-language branch and dead `rover_eligible` flag removed

**Source**: REVIEW3 Deviation #1
**File**: `pipeline/asr_engines.py`

WhisperX detects language once for the whole audio, so all segments report the
same language. The mixed-language branch of `_detect_nl_windows` (building 30s
windows with majority-language voting) was unreachable. The `rover_eligible` flag
written onto `language_chunks` was never read by `pipeline/rover.py`.

**Change**:

- `_detect_nl_windows` simplified to two branches:
  - `detected_lang == "nl"` → return `[(0.0, total_end)]` (Engine B runs)
  - otherwise → return `[]` (Engine B skipped)
- Removed the unreachable `else` branch (mixed-language 30s windowing).
- Removed the `rover_eligible` flag entirely (both the write in `run()` and any
  reference to it).
- Added docstring explaining the current limitation and noting that per-window
  detection is deferred pending the P1-3 subprocess refactor.

Also fixed §4.2 (`total_end` using bare `s["end"]` → `s.get("end", 0.0)`).

### Fix #6. Unmatched B-words merged into nearest segment instead of fragments

**Source**: REVIEW3 §4.1
**File**: `pipeline/rover.py:185-228`

The previous fix (sort by `start`) created one-word segments for each unmatched
B word. On samples with many unmatched B words, this produced `[SPEAKER_X]` labels
around single words, breaking up natural speaker turns.

**Change**: Instead of creating new segments, each unmatched B word is merged into
the temporally-nearest Engine A segment by center-point distance:

```python
for w in remaining:
    w_center = (w["start"] + w["end"]) / 2
    best_idx = 0
    best_dist = float("inf")
    for i, seg in enumerate(segments):
        dist = abs(w_center - (seg["start"] + seg["end"]) / 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    segments[best_idx]["words"].append(w)
    segments[best_idx]["text"] += " " + w["word"]
    if w["end"] > segments[best_idx]["end"]:
        segments[best_idx]["end"] = w["end"]
```

This preserves the natural segment/speaker structure while still surfacing Engine
B's extra words. No re-sorting or re-numbering needed.

### Fix #7. More transient exception types in `_retry`

**Source**: REVIEW3 §6.1
**File**: `pipeline/postcorrect.py:243-252`

`_retry` caught `URLError`, `socket.timeout`, `ConnectionError`, `TimeoutError`,
`openai.RateLimitError`, `openai.APIConnectionError`. Missing:

- `openai.APITimeoutError` — fires when the API responds too slowly
- `openai.InternalServerError` — 5xx responses, textbook transient

**Change**: Added both to the transient tuple:

```python
from openai import (
    RateLimitError as _RateLimit,
    APIConnectionError as _APIConn,
    APITimeoutError as _APITimeout,
    InternalServerError as _Internal,
)
transient = transient + (_RateLimit, _APIConn, _APITimeout, _Internal)
```

---

## Minor fixes

### 4.3. `huggingface_hub` version parser handles pre-release tags

**File**: `pipeline/diarize.py:56`

`tuple(int(x) for x in _hf.__version__.split('.')[:2])` raised `ValueError` on
pre-release versions like `1.0rc1` or `1.0.0.dev0`.

**Change**: Use regex to extract major.minor:

```python
_ver_match = _re.match(r"(\d+)\.(\d+)", _hf.__version__)
_hf_version = (int(_ver_match.group(1)), int(_ver_match.group(2))) if _ver_match else (0, 0)
```

Defaults to `(0, 0)` (no patch) if version is unparseable.

### 7.1. Removed dead `from typing import Any` import

**File**: `pipeline/asr_engines.py:15`

Unused import removed.

### 7.2. Fixed stale glossary comment

**File**: `pipeline/glossary.py:131`

Comment referenced `term` section that was removed in codefixes2.md (NEW-P3-1):

```
-# Sections weight the entry: brand > person > place > term.
+# Sections weight the entry: brand > person > place.
```

### 7.5. Removed no-op `dynapse -> Dynapse` case-only mapping

**File**: `pipeline/glossary.py:152`

`dynapse -> Dynapse` is a case-only rewrite (no phonetic correction). Kept
`dinapse -> Dynapse` which corrects the common mishearing. Removed the no-op.

---

## Files changed

| File | Changes |
|------|---------|
| `pipeline/rover.py` | Fix #1 (multiplicative revert + comment), Fix #6 (merge into nearest) |
| `pipeline/postcorrect.py` | Fix #2 (stderr logging), Fix #3 (system prompt restructure + cache_control removal), Fix #4 (accumulate usage), Fix #7 (more transient types) |
| `pipeline/asr_engines.py` | Fix #5 (simplify detect + remove rover_eligible), 7.1 (dead import), 4.2 (safe total_end) |
| `pipeline/diarize.py` | 4.3 (regex version parser) |
| `pipeline/glossary.py` | 7.2 (stale comment), 7.5 (drop no-op mapping) |

## Items acknowledged but not changed

These were noted in REVIEW3 as "nice-to-have" or already documented:

- **`DOCKER_STAGE_EXTRA` global state** (§5) — noted as low priority; current code
  is correct.
- **`_score_word` engine fallback** (§7.3) — harmless defensive default, kept.
- **README / Task 14** — still pending, flagged by all three reviews.

## Remaining deferred items (unchanged from prior reviews)

- P1-3: ASR subprocess refactor (large change, deferred)
- P1-16 / P1-17: AudioSR / DeepFilterNet full installs (need Rust toolchain)
- Per-30s-window language detection (depends on P1-3)
