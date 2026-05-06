# Phase 2 Code Review — Whisper Pipeline v2 (post-`codefixes2.md`)

**Document ID:** WHISPER-REVIEW-P2-003
**Reviewer:** Code Review Agent (Claude)
**Date:** 2026-05-06
**Scope:** Third-pass review after `codefixes2.md` addressed `REVIEW2.md` findings.
**FSD Reference:** `HANDOFF.md` (sections 1–8)
**Prior Reviews:** `REVIEW1.md`, `REVIEW2.md`
**Commit Reviewed:** `2b6ad42` (working tree, uncommitted v2 work)

---

## Verdict: **PASS WITH NOTES**

The third pass shows steady, real progress. Of the eleven items
`codefixes2.md` claims to address, eight are correctly fixed (NEW-P0-1,
NEW-P0-3, P1-6, NEW-P1-1, NEW-P2-1, NEW-P2-2, NEW-P2-3, NEW-P3-1,
NEW-P3-2). Three are partially fixed or shape-only (NEW-P0-2 language
gating, P1-10 prompt caching, P1-11 sidecar usage) and one round-2
recommendation (NEW-P1-2, glossary additive bonus) introduced a
**deviation from the FSD** — the spec explicitly says "multiply by
glossary weight" and the fix changed it to additive. This was my own
recommendation in REVIEW2 and was wrong. The remaining defects are
contained and not regressions. Recommend one more focused fix pass before
declaring v2.0 stable, plus the long-pending README update.

---

## Table of Contents

1. [Files Reviewed](#files-reviewed)
2. [Coverage Analysis](#1-coverage-analysis)
3. [Deviation Report](#2-deviation-report)
4. [Plan vs. Implementation](#3-plan-vs-implementation)
5. [Edge Cases & Safety](#4-edge-cases--safety)
6. [Concurrency & Platform Issues](#5-concurrency--platform-issues)
7. [Error Handling](#6-error-handling)
8. [Code Quality](#7-code-quality)
9. [Summary](#8-summary)
10. [Recommendation](#9-recommendation)

---

## Files Reviewed

| File | Purpose | Touched in this round? |
|------|---------|------------------------|
| `transcribe` | Bash orchestrator | Yes — NEW-P0-1, NEW-P2-1, NEW-P3-2 |
| `pipeline/asr_engines.py` | Engine A/B + ROVER gating | Yes — NEW-P0-2 |
| `pipeline/diarize.py` | Speaker diarization | Yes — NEW-P0-3 |
| `pipeline/rover.py` | ROVER reconciliation | Yes — P1-6 sort, NEW-P1-1, NEW-P1-2 |
| `pipeline/postcorrect.py` | LLM post-correction | Yes — P1-10, P1-11, NEW-P2-2, NEW-P2-3 |
| `pipeline/glossary.py` | Glossary loader | Yes — NEW-P3-1 |
| `pipeline/preprocess.py` | Audio preprocessing | No — already fixed in round 1 |
| `pipeline/prompt_builder.py` | Prompt sanitiser | No |
| `pipeline/render.py` | Text rendering | No |
| `pipeline/artifacts.py` | Path helpers | No |
| `Dockerfile` | Image definition | No |

---

## 1. Coverage Analysis

Mapped against the items `codefixes2.md` claims to address.

| ID | Source | Claim | Verdict |
|----|--------|-------|---------|
| NEW-P0-1 | REVIEW2 | Auto-prompt no longer falls back to raw LLM output | **DONE** — `transcribe:395-400` adds an explicit `else` that disables auto-prompt and clears the text. Verified that no path between sed-stripped raw Ollama output and `final_prompt.txt` remains. |
| NEW-P0-2 | REVIEW2 | Per-chunk language gating actually gates | **PARTIAL** — Engine B is now skipped on uniform non-nl audio, which is a real improvement. But mixed-language gating is still not effective (see Deviation #1) and `rover_eligible` remains unread by ROVER. |
| NEW-P0-3 | REVIEW2 | Pin-vs-patch contradiction resolved | **DONE** — `diarize.py:55-77` gates the monkeypatch on `huggingface_hub.__version__ >= (1, 0)`. With the pinned 0.30.2 the patch is a no-op; if the pin drifts, the patch activates. Comment explains the relationship. |
| P1-6 (sort) | REVIEW2 | Unmatched B words appear in chronological order | **DONE** — `rover.py:215-231` emits each unmatched B word as its own segment, then sorts all segments by `start` and renumbers ids. Caveat in §4 below about render-time fragmentation. |
| P1-10 | REVIEW2 | Prompt caching enabled against Z.ai | **PARTIAL / SHAPE-ONLY** — `cache_control` is set on both messages but (a) the OpenAI SDK 1.54.0 doesn't natively support it; (b) the long content is concatenated into the user message so caching the whole user message provides no prefix-cache benefit; (c) Z.ai may reject the unknown field. See Deviation #2. |
| P1-11 | REVIEW2 | Sidecar token-usage JSON | **PARTIAL** — `cleaned_usage.json` is written for Z.ai backend, but only the **last batch's** usage is recorded; multi-batch totals are lost. See Deviation #3. |
| NEW-P1-1 | REVIEW2 | `_score_word` no longer reads phantom `avg_logprob` | **DONE** — `rover.py:127` reduced to `w.get("logprob", 0.0)`. |
| NEW-P1-2 | REVIEW2 | Glossary tiebreak no longer dead | **DONE in code, BAD in concept** — Changed multiplicative to additive (`prob += weight`) per my own REVIEW2 suggestion. But the FSD §2 stage 2 says **"Multiply by glossary weight"**. The fix made the tiebreak engageable at the cost of deviating from the spec. See Deviation #4. |
| NEW-P2-1 | REVIEW2 | `--no-correct` chain is `if/then/fi` | **DONE** — `transcribe:145-148`. |
| NEW-P2-2 | REVIEW2 | `_retry` only catches transient errors | **DONE (with gaps)** — `postcorrect.py:243-249` lists `URLError`, `socket.timeout`, `ConnectionError`, `TimeoutError`, `RateLimitError`, `APIConnectionError`. Missing: `openai.APITimeoutError` and `openai.InternalServerError` (5xx). See §6. |
| NEW-P2-3 | REVIEW2 | Retry log goes to stderr | **DONE** — `postcorrect.py:258`. |
| NEW-P3-1 | REVIEW2 | Empty `[term]` section removed | **DONE** — `glossary.py:163`. (Minor doc nit: `_DEFAULT_GLOSSARY` line 131 still says "brand > person > place > term".) |
| NEW-P3-2 | REVIEW2 | `docker_stage` accepts extra mounts | **DONE** — `transcribe:169,279,421-427` introduces `DOCKER_STAGE_EXTRA` array. |

### Items not in `codefixes2.md` that remain unfixed (per REVIEW2 § "Items still on the deferred list")

| ID | Status |
|----|--------|
| P1-3 | ASR subprocess pattern still present. Acknowledged deferred. |
| P1-16 / P1-17 | AudioSR / DeepFilterNet still silently no-op. Acknowledged deferred. |
| `rover_eligible` consumed by ROVER | Still unread. Tied to deferred per-window Engine B. |
| HANDOFF Task 14 (README update) | Still pending. |

---

## 2. Deviation Report

### Deviation #1 — `_detect_nl_windows` mixed-language branch is unreachable in practice

**Severity:** MAJOR
**File:** `pipeline/asr_engines.py:254-288`
**FSD Reference:** §2 stage 2 ("Per-chunk language gating: WhisperX's built-in language ID over 30 s windows. nl chunks → run both engines and ROVER. en / fr chunks → Engine A only").

The new gating logic computes:

```python
detected_lang = segments[0].get("language", "nl")
all_same = all(s.get("language", detected_lang) == detected_lang for s in segments)
```

WhisperX wraps faster-whisper and detects language **once** for the whole
audio, then forces that language for all subsequent transcription. The
`segments` returned therefore have a uniform `language` value (or no
`language` field at all, in which case the `.get(..., detected_lang)`
fallback makes them uniform anyway). `all_same` is always True in
practice.

Net effect: the function takes one of two branches:

- Uniform `nl` → `[(0.0, total_end)]` → Engine B runs over the whole
  audio (same as pre-fix).
- Uniform non-`nl` → `[]` → Engine B is skipped entirely (**genuine
  improvement** for English/French recordings).

The `else` branch (lines 274-286) that builds 30 s windows by majority
language is unreachable until per-30-s language detection is actually
implemented (deferred per `codefixes2.md`).

The `rover_eligible` flag added at `asr_engines.py:247` is never read by
`pipeline/rover.py` (verified by `grep`). It documents intent but
provides no actual gating to ROVER.

**Recommendation:** either (a) implement explicit 30 s-window detection
(call `whisperx.detect_language` on slices of the audio array), or (b)
delete the unreachable mixed-language branch and the unused
`rover_eligible` flag and document the simpler "uniform skip" behaviour
that the code actually performs. Today the code claims more than it
delivers, which makes the next reviewer waste effort tracing it.

### Deviation #2 — Z.ai prompt caching is shape-only and may break the API call

**Severity:** MAJOR
**File:** `pipeline/postcorrect.py:200-218`
**FSD Reference:** §2 stage 4 ("Use prompt caching where the SDK supports it").

Three problems:

1. **`cache_control` is Anthropic-API syntax, not OpenAI.** The OpenAI SDK
   1.54.0 (pinned in the Dockerfile) does not advertise this field. In
   newer SDK versions, unknown fields on `ChatCompletionMessageParam` are
   silently dropped by the typed-dict serialiser. In Z.ai's
   OpenAI-compatible endpoint, behaviour is implementation-defined: it may
   ignore the field (no caching) or reject the request (400).

2. **Prefix-caching needs structured content blocks.** Even if Z.ai
   accepts `cache_control`, the documented Anthropic semantic is to mark
   a specific text block within `content`, not the whole message. Setting
   `cache_control` at the message level does not enable prefix caching.

3. **System prompt is concatenated into the user message.** Line 200:

   ```python
   user_prompt = f"{system_prompt}\n\nTranscript segments to correct:\n{batch_text}"
   ```

   The actual `system` role only sends "Respond only in valid JSON
   array. No markdown fences." (a one-liner, not worth caching). The
   long template-with-glossary lives in the user message, where every
   batch's `batch_text` differs — so the user messages also won't match
   any cache.

**Recommendation:**

- Stop concatenating `system_prompt` into the user message. Send the
  long template + glossary as the actual system message (with
  `cache_control` if Z.ai supports it).
- For the OpenAI SDK, route caching hints through `extra_body={...}`
  rather than message-level dict keys, since `extra_body` is the
  documented escape hatch for non-standard params.
- Or: accept that this is best-effort and gate it behind a debug flag
  until Z.ai's caching semantics are confirmed by a real call. The
  current implementation costs nothing at best and 100 % of the API call
  at worst — and the failure mode is silent (the user sees fewer
  corrections, not an error).

### Deviation #3 — `last_usage` records only the last batch, not the run total

**Severity:** MINOR
**File:** `pipeline/postcorrect.py:196,222-227,415-418`
**FSD Reference:** §5 task 13 verification check 4 ("log token usage").

`ZaiCorrector.last_usage` is overwritten on every `correct()` call.
For a 48-segment recording (2 batches of 40 + the spillover), only the
spillover batch's usage lands in `cleaned_usage.json`. The verification
check ("expect ~3–8 k input tokens per 30 min of audio") cannot be
satisfied by reading the sidecar.

**Recommendation:** accumulate. Either store a list of per-batch usage
records, or sum into running totals (`total_prompt_tokens`,
`total_completion_tokens`, `batches`). The sidecar file should reflect
the entire run.

### Deviation #4 — Glossary bonus changed from multiplicative to additive (deviation from FSD)

**Severity:** MAJOR (regression vs. spec; my own REVIEW2 recommendation
caused it)
**File:** `pipeline/rover.py:117-126`
**FSD Reference:** §2 stage 2 ("Per cluster, score candidates by `prob`
… **Multiply by glossary weight** when `glossary.is_canonical(word)`").

The spec explicitly says **multiply**. `codefixes1.md` correctly
implemented `prob *= (1 + weight)`. REVIEW2 then observed that with a
multiplicative bonus, the third tie-break (`is_glossary`) was dead code.
My recommendation was to switch to additive — but that was a mistake on
my part, since "make the tiebreak fire" is not worth contradicting the
spec.

The current additive form has another problem: with brand
`weight = entry.weight / 4.0 = 1.0`, a brand glossary hit gets `prob +
1.0`. Since base `prob` is capped at 1.0, **a brand hit always wins**
regardless of how confident a non-glossary candidate is. A 0.99 non-brand
candidate loses to a 0.10 brand candidate (final scores 0.99 vs 1.10).
That is far stronger than "prefer glossary on ties."

**Recommendation:** revert to multiplicative `prob *= (1 + weight)`. The
fact that step 3 of the cascade becomes structurally redundant is fine —
the cascade is `score → logprob → glossary → engine A`, and a
multiplicative score already encodes the glossary preference, so the
glossary tiebreak is *correctly redundant*, not buggy. Document this in
a one-line comment so the next reviewer doesn't repeat my mistake.

If a softer behaviour is genuinely wanted (e.g., a 0.95 strong
non-glossary candidate should not always lose to a 0.20 glossary one),
adjust the spec — don't drift the implementation.

---

## 3. Plan vs. Implementation

`codefixes2.md` is itself the plan for this round. Compared to that plan:

| Plan Item | Planned | Actual | Status |
|-----------|---------|--------|--------|
| NEW-P0-1: harden auto-prompt | Disable auto-prompt + clear text on prompt_builder failure | Done at `transcribe:395-400` | ✅ Match |
| NEW-P0-2: per-chunk language gating | Detect uniform non-nl, skip Engine B; mixed-language gating deferred | Skip-on-non-nl works; mixed-language code path is unreachable | ⚠️ Plan acknowledged the deferral; but the unreachable code remains in the file rather than being commented out or stubbed |
| NEW-P0-3: pin-vs-patch | Gate monkeypatch on installed version | Done at `diarize.py:55-77` | ✅ Match |
| P1-6 sort | Insert unmatched B words individually, sort by start, renumber ids | Done at `rover.py:215-231` | ✅ Match |
| P1-10: prompt caching | Add `cache_control` ephemeral hints | Added but with the structural problems in Deviation #2 | ⚠️ Implementation deviates from working caching semantics |
| P1-11: sidecar usage JSON | Accumulate usage for the run, write sidecar | Records last batch only | ⚠️ Partial |
| NEW-P1-1: drop phantom key | Single-key lookup | Done at `rover.py:127` | ✅ Match |
| NEW-P1-2: glossary additive | Switch to additive | Done at `rover.py:125`, but **deviates from FSD §2 stage 2** | ❌ Plan implemented; spec violated |
| NEW-P2-1: explicit `if`s | Replace `&&` chain | Done at `transcribe:145-148` | ✅ Match |
| NEW-P2-2: transient-only retry | Filter exception types | Done at `postcorrect.py:243-249` (with gaps in §6) | ✅ Mostly match |
| NEW-P2-3: stderr log | `print(..., file=sys.stderr)` | Done at `postcorrect.py:258` | ✅ Match |
| NEW-P3-1: empty section | Remove `[term]` header | Done at `glossary.py:163` | ✅ Match (one stale doc comment remains) |
| NEW-P3-2: extra mounts | `DOCKER_STAGE_EXTRA` array | Done at `transcribe:169,279,421-427` | ✅ Match |

**Undocumented deviations from the plan:** none material — the round-2 plan
is honest about what was deferred.

---

## 4. Edge Cases & Safety

### 4.1 ROVER unmatched-B word render-time fragmentation

**Severity:** MINOR
**File:** `pipeline/rover.py:215-231` + `pipeline/render.py`

After the chronological-sort fix, an unmatched B word from second 12 of
a 30-minute recording becomes a single-word segment with one entry in
`segments`. The renderer at `render.py:_assign_speakers` assigns a
speaker to it via segment-centre overlap and then prints `[SPEAKER_X]`
followed by that one-word fragment, breaking up an otherwise natural
speaker turn. On samples with many unmatched B words, the cleaned output
will have many one-word interruptions.

**Recommendation:** merge an unmatched B word into the temporally-nearest
Engine A segment instead of creating a fresh one. The segment's `text`
gains one extra word, the natural turn structure is preserved, and ROVER
no longer needs to re-sort or renumber.

### 4.2 `_detect_nl_windows` panics on empty `end` field

**Severity:** MINOR
**File:** `pipeline/asr_engines.py:264`

`total_end = max(s["end"] for s in segments)` raises `KeyError` if any
segment lacks `end` (defensive code elsewhere uses `seg.get("end", ...)`).
WhisperX always populates `end`, so this is a low-probability failure,
but the function is otherwise quite defensive (`s.get(...)`) and the
inconsistency suggests the bracket access wasn't intentional.

**Recommendation:** `total_end = max(s.get("end", 0.0) for s in segments)`.

### 4.3 `huggingface_hub` version parser fails on pre-release tags

**Severity:** MINOR
**File:** `pipeline/diarize.py:55`

`tuple(int(x) for x in _hf.__version__.split('.')[:2])` raises
`ValueError` for versions like `1.0rc1` or `1.0.0.dev0`. With the
current pin to 0.30.2, this is moot; if the pin ever drifts to a
pre-release, the diarize stage will crash on its first call before the
patch even runs.

**Recommendation:** wrap in `try/except ValueError` or use a more
permissive parser (e.g. `re.match(r"(\d+)\.(\d+)", _hf.__version__)`).

### 4.4 ZaiCorrector `cache_control` may cause Pydantic validation rejection

**Severity:** MAJOR (already covered as Deviation #2; re-listed here as
the top safety risk because if `cache_control` causes a 400, the run
fails for every batch and there is no fallback)

If Z.ai validates request bodies strictly, every Z.ai post-correction
call returns 400 → `_retry` does NOT catch validation errors (it only
retries `RateLimitError` / `APIConnectionError`) → exception propagates
to `run()` line 405 → `except Exception: all_corrected.extend(batch)`
keeps verbatim segments. The user sees zero corrections from
`--cloud-correct` and **no error message** (the `except` swallows it).

**Recommendation:** at minimum, log the swallowed exception in the
`except` at `postcorrect.py:405-407`. Better: do a smoke call without
`cache_control`, on success re-issue with caching; on Pydantic /
validation error, retry without caching transparently.

### 4.5 Glossary additive bonus over-dominates

Already covered as Deviation #4. Re-listed here because under the
current implementation, every brand-glossary word in either ASR engine
output (even one Engine B fabricated due to Dutch-model bias) wins
unconditionally. That is a transcription-quality risk.

---

## 5. Concurrency & Platform Issues

No new concurrency or threading issues introduced this round. The
pipeline is still single-threaded per stage with Docker stages serialised
by the bash orchestrator. Ollama lifecycle (start/stop for VRAM
management) is handled the same way as round 1.

One observation:

- **`DOCKER_STAGE_EXTRA` global state.** `transcribe:169` declares it
  globally. The manual-prompt block at `transcribe:421-427` sets and
  resets it. If a future contributor adds a third caller that sets but
  forgets to reset, the extra mount leaks to subsequent stages. Mitigation:
  pass extras as an explicit function parameter rather than via global.
  Low priority; the current code is correct.

---

## 6. Error Handling

### 6.1 `_retry` doesn't cover all transient OpenAI exceptions

**Severity:** MINOR
**File:** `pipeline/postcorrect.py:243-249`

Caught: `URLError`, `socket.timeout`, `ConnectionError`, `TimeoutError`,
`openai.RateLimitError`, `openai.APIConnectionError`.

Not caught:

- `openai.APITimeoutError` — fires when the API responds too slowly,
  which is genuinely transient.
- `openai.InternalServerError` (5xx) and other `APIStatusError`
  subclasses for `502`, `503`, `504`. These are textbook transient.

**Recommendation:** add `openai.APITimeoutError` and either
`openai.InternalServerError` specifically or the broader
`openai.APIStatusError` filtered by `e.status_code in {502, 503, 504}`.

### 6.2 Z.ai correction failures are silently swallowed

**Severity:** MAJOR
**File:** `pipeline/postcorrect.py:402-407`

```python
try:
    corrected = corrector.correct(batch, system_prompt, language)
    all_corrected.extend(corrected)
except Exception as e:
    # On failure, keep original segments
    all_corrected.extend(batch)
```

Bare `except Exception: pass` (the `as e` is unused). On any failure —
quota exhausted, malformed JSON from the LLM, validation rejection from
Z.ai, programming bug — the user sees no warning, only that no
corrections were applied. Combined with Deviation #4.4, a `cache_control`
rejection silently disables cloud correction for the whole run.

**Recommendation:** `except Exception as e: print(f"Batch {i}: {type(e).__name__}: {e}", file=sys.stderr); all_corrected.extend(batch)`.
Two lines, infinite debuggability gain.

### 6.3 `_retry` returns `None` if all attempts raise non-transient exceptions

**Severity:** MINOR
**File:** `pipeline/postcorrect.py:251-259`

The for-loop only catches `transient`. A non-transient exception escapes
on the first attempt — that's correct. But the function has no `return`
at the bottom of the for-loop body if all attempts raise transient
exceptions and the last `raise` re-raises. There is no fall-through path
that returns `None`, so this is actually fine. Listed only to confirm
no implicit-None bug.

---

## 7. Code Quality

Substantive observations only.

### 7.1 Unused import in `asr_engines.py`

`from typing import Any` at line 15 is unused. Cosmetic.

### 7.2 `_DEFAULT_GLOSSARY` doc comment lists a removed section

`pipeline/glossary.py:131`:

```python
# Sections weight the entry: brand > person > place > term.
```

The `[term]` section was removed (NEW-P3-1) but the comment still
mentions term-weight ordering. Either re-add an empty `[term]` with a
comment placeholder so users know they can extend the file, or strip
"term" from the comment. Tiny inconsistency that costs nothing to fix.

### 7.3 `_score_word` returns a 4-tuple with optional `engine` fallback

`pipeline/rover.py:128`:

```python
engine = w.get("engine", "A")
```

Words created in `_build_anchor_clusters` are decorated with
`{"engine": "A", ...}` or `{"engine": "B", ...}` (lines 89, 91, 98), so
the fallback `"A"` is never reached. Keeping it is harmless; calling
attention only because it suggests defensive copying of legacy code.

### 7.4 `ROVER` `rover_eligible` flag is dead state

`pipeline/asr_engines.py:243-249` writes `rover_eligible` onto each
`language_chunks` entry. Nothing in the codebase reads it. It is currently
documentation, not a control signal. Either remove until ROVER actually
consumes it, or add a one-line consumer-side check (e.g., ROVER skips
clusters whose start..end is fully outside any nl-eligible chunk).

### 7.5 `_DEFAULT_GLOSSARY` has duplicate canonical mappings

`pipeline/glossary.py:151-152`:

```
dinapse -> Dynapse
dynapse -> Dynapse
```

`dynapse -> Dynapse` is a no-op rewrite (case-only). Either drop it or
move it to a "case normalisation" section. Cosmetic.

---

## 8. Summary

| Category | Critical | Major | Minor | Info |
|----------|----------|-------|-------|------|
| Spec conformance | 0 | 2 (Dev #1, Dev #4) | 0 | 0 |
| Plan conformance | 0 | 0 | 2 (P1-10/P1-11 partial) | 0 |
| Correctness | 0 | 1 (4.5 glossary over-dominance) | 1 (4.1 fragmentation) | 0 |
| Safety | 0 | 1 (4.4 cache_control silent fail) | 2 (4.2, 4.3) | 0 |
| Concurrency | 0 | 0 | 1 (DOCKER_STAGE_EXTRA global) | 0 |
| Error handling | 0 | 1 (6.2 silent swallow) | 2 (6.1, 6.3) | 0 |
| Code quality | 0 | 0 | 4 (7.1 - 7.5) | 0 |

**Critical:** none.
**Major (4):** prompt-cache shape, glossary additive deviation,
glossary over-dominance, silent swallow in postcorrect run-loop.
**Minor (12):** scattered.

---

## 9. Recommendation

**PASS WITH NOTES** — proceed to README/Task 14, then a focused fix pass.

The codebase is now substantively closer to the FSD than it was at the
start of REVIEW1. Three rounds of fixes have eliminated most P0/P1
defects. The remaining issues are real but tractable:

### Must-fix before declaring v2.0 stable

1. **Revert glossary bonus to multiplicative** (`rover.py:125`,
   Deviation #4). One-line change. Aligns with FSD §2 stage 2.
   Document the dead-but-correct tiebreak in a comment.
2. **Make `--cloud-correct` audible on failure** (`postcorrect.py:405-407`,
   §6.2). Add `print(f"...{e}", file=sys.stderr)` to the `except`. Two
   lines.
3. **Decide on `cache_control` strategy** (`postcorrect.py:200-218`,
   Deviation #2). Either move the system prompt out of the user message
   and set caching properly, or remove the `cache_control` keys until
   Z.ai's caching contract is verified. Current state is unsafe.
4. **Accumulate Z.ai usage across batches** (`postcorrect.py:223-227,415-418`,
   Deviation #3). Trivial change to either sum or list.

### Should-fix in the same pass

5. Either implement real per-30-s language detection (`asr_engines.py`,
   Deviation #1) or delete the unreachable mixed-language branch and the
   unread `rover_eligible` flag.
6. Merge unmatched B-words into the temporally-nearest A-segment
   (`rover.py`, §4.1) instead of emitting tiny one-word segments.
7. Add `openai.APITimeoutError` and `InternalServerError` to `_retry`'s
   transient list (§6.1).

### Nice-to-have

- Dead-import cleanup (`from typing import Any`).
- Stale-comment cleanup in `_DEFAULT_GLOSSARY`.
- Drop the `dynapse -> Dynapse` no-op or move it.

### Then

- Land Task 14 (README update) — still pending from `HANDOFF.md`. New
  flags (`--cloud-correct`, `--ensemble`, `--context`, `--glossary`,
  `--diarize-model`, `--refresh-model`, `--keep-intermediates`,
  `--no-*`) and the `~/.config/whisper/` files (`glossary.txt`,
  `zai-key`, `.glm-resolved`) are not documented anywhere user-facing.

### Then re-verify

- Run the three samples at `--quality perfect` and `--quality perfect
  --cloud-correct` per HANDOFF §5 task 13.
- Run a non-Dutch sample at `--quality perfect` to verify the
  skip-Engine-B-on-non-nl path works.
- Run any sample with `--prompt /tmp/prompt.txt` (a path *outside*
  `INPUT_DIR`) to verify NEW-P3-2 works.
- Run with `~/.config/whisper/hf-token` removed to verify the warning
  prints and diarize gracefully degrades.
- Run `--quality perfect --no-correct` to verify the `--no-*` overrides.

End of review.
