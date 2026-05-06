# Phase 2 Code Review #4 — Whisper Pipeline v2 (post-`codefixes3.md`)

**Document ID:** WHISPER-REVIEW-P2-004
**Reviewer:** Code Review Agent (Claude)
**Date:** 2026-05-06
**Scope:** Fourth-pass review after `codefixes3.md` addressed `REVIEW3.md` findings.
**FSD Reference:** `HANDOFF.md` (sections 1–8)
**Prior Reviews:** `REVIEW1.md`, `REVIEW2.md`, `REVIEW3.md`
**Commit Reviewed:** `2b6ad42` (working tree, uncommitted v2 work; pipeline files mtime 2026-05-06 13:52-13:53)

---

## Verdict: **PASS WITH NOTES**

This round is the cleanest yet. Of the eleven items `codefixes3.md` claims
to address, ten land correctly: glossary reverted to multiplicative per
FSD, post-correct failures now log to stderr, system prompt is in the
right role with `cache_control` removed, Z.ai usage accumulates across
batches, B-words merge into the nearest A-segment, retry covers the
right exception types, version parser tolerates pre-release tags, dead
imports gone, stale doc comments fixed. **One bug carries over from
round 2 into round 3's "fix"**: `_detect_nl_windows` now looks up
`"language"` on segments whose actual key is `"lang"`, so the
non-Dutch-skip path is silently unreachable and Engine B still runs on
every ensemble call regardless of detected language. Two minor edge
cases noted below. The remaining work is the long-pending README (Task
14) and the deferred per-30 s window detection.

---

## Table of Contents

1. [Coverage Analysis](#1-coverage-analysis)
2. [Deviation Report](#2-deviation-report)
3. [Plan vs. Implementation](#3-plan-vs-implementation)
4. [Edge Cases & Safety](#4-edge-cases--safety)
5. [Concurrency & Platform Issues](#5-concurrency--platform-issues)
6. [Error Handling](#6-error-handling)
7. [Code Quality](#7-code-quality)
8. [Summary](#8-summary)
9. [Recommendation](#9-recommendation)

---

## Files Reviewed

| File | Touched in this round? | Purpose |
|------|------------------------|---------|
| `pipeline/rover.py` | Yes — Fix #1, Fix #6 | ROVER reconciliation |
| `pipeline/postcorrect.py` | Yes — Fix #2, Fix #3, Fix #4, Fix #7 | LLM post-correction |
| `pipeline/asr_engines.py` | Yes — Fix #5, 4.2, 7.1 | Engine A/B + gating |
| `pipeline/diarize.py` | Yes — 4.3 | Speaker diarization |
| `pipeline/glossary.py` | Yes — 7.2, 7.5 | Glossary loader |
| `transcribe`, `Dockerfile`, `pipeline/preprocess.py`, `prompt_builder.py`, `render.py`, `artifacts.py`, `__init__.py` | No — verified mtime unchanged | (unchanged this round) |

---

## 1. Coverage Analysis

Mapped against the items `codefixes3.md` claims to address.

| ID | Source | Claim | Verdict |
|----|--------|-------|---------|
| Fix #1 | REVIEW3 Dev #4 | Glossary bonus reverted to multiplicative | **DONE** — `rover.py:128` is `prob *= (1.0 + weight)`. Comment at lines 118-122 explains why step 3 of the cascade is correctly redundant. |
| Fix #2 | REVIEW3 §6.2 | `run()` batch-loop logs the swallowed exception | **DONE** — `postcorrect.py:412-414`. |
| Fix #3 | REVIEW3 Dev #2 | System prompt in system role, `cache_control` removed | **DONE** — both `OllamaCorrector` (`postcorrect.py:156-169`) and `ZaiCorrector` (`postcorrect.py:198-216`) now structure the prompt correctly. No `cache_control` keys remain (verified via grep). |
| Fix #4 | REVIEW3 Dev #3 | Z.ai usage accumulated across batches | **DONE** — `postcorrect.py:219-230` initialises `last_usage` once, increments per-batch totals. Sidecar JSON now reflects the run total. |
| Fix #5 | REVIEW3 Dev #1 | Unreachable mixed-language branch + `rover_eligible` removed | **PARTIALLY DONE** — Mixed-language branch deleted, `rover_eligible` flag removed (verified via grep). But the simplification kept the round-2 key-name bug; see Deviation #1. |
| Fix #6 | REVIEW3 §4.1 | Unmatched B-words merged into nearest A-segment | **DONE (with edge case)** — `rover.py:213-228`. Edge case if Engine A produced zero segments; see §4.1. |
| Fix #7 | REVIEW3 §6.1 | `_retry` covers more transient OpenAI exceptions | **DONE** — `postcorrect.py:247-254` adds `APITimeoutError` and `InternalServerError`. Together they cover the documented transient cases on OpenAI SDK 1.54. |
| 4.2 | REVIEW3 §4.2 | `total_end` uses defensive `.get("end", 0.0)` | **DONE** — `asr_engines.py:259`. |
| 4.3 | REVIEW3 §4.3 | Version parser handles pre-release tags | **DONE** — `diarize.py:55-57` uses regex; falls back to `(0, 0)` when unparseable. |
| 7.1 | REVIEW3 §7.1 | Dead `from typing import Any` removed | **DONE** — verified via grep (no remaining match in `asr_engines.py`). |
| 7.2 | REVIEW3 §7.2 | Stale "term" reference in glossary doc comment fixed | **DONE** — `glossary.py:131` now says "brand > person > place." |
| 7.5 | REVIEW3 §7.5 | Removed no-op `dynapse -> Dynapse` | **DONE** — `glossary.py:151` retains `dinapse -> Dynapse`, `dynapse -> Dynapse` is gone. |

### Items not addressed this round (acknowledged in `codefixes3.md`'s closing list)

| Item | Status |
|------|--------|
| README / Task 14 | Still pending. Flagged by all four reviews. |
| P1-3 (ASR subprocess refactor) | Deferred. |
| P1-16 / P1-17 (AudioSR / DeepFilterNet) | Deferred. |
| Per-30 s window language detection | Deferred (depends on P1-3). |
| `DOCKER_STAGE_EXTRA` global state | Acknowledged low priority; current code correct. |

---

## 2. Deviation Report

### Deviation #1 — `_detect_nl_windows` reads `"language"` but `_run_whisperx_engine` writes `"lang"`

**Severity:** MAJOR (regression of round-2 bug, retained through round-3 simplification)
**File:** `pipeline/asr_engines.py:257`
**FSD Reference:** §2 stage 2 ("Per-chunk language gating: WhisperX's
built-in language ID over 30 s windows. nl chunks → run both engines
and ROVER. en / fr chunks → Engine A only.")

The inner WhisperX subprocess (`asr_engines.py:117-125`) writes each
output segment with the key `"lang"` (intentionally renamed from
WhisperX's `"language"` to avoid colliding with anything else):

```python
segments_out.append({{
    ...
    "lang": seg.get("language", top_lang),
    ...
}})
```

The outer Python (`asr_engines.py:255-261`) reads:

```python
detected_lang = segments[0].get("language", "nl")
if detected_lang == "nl":
    return [(0.0, total_end)]
return []
```

`segments[0]` has the key `"lang"`, not `"language"`. The lookup misses
on every call → `detected_lang` falls through to the default `"nl"` →
the function always returns `[(0.0, total_end)]` → Engine B runs on the
entire audio for every ensemble call, regardless of what language
WhisperX detected.

**This silently un-fixes Fix #5.** The codefixes3 plan asserts Engine B
is "skipped on uniform non-nl audio"; the actual behaviour is Engine B
runs always. An English-only meeting recording with `--ensemble` still
makes Engine B (Dutch fine-tune) transcribe English into garbage that
ROVER then has to outvote on every word — exactly the failure mode
REVIEW1's P1-1 / P1-14 flagged.

The bug is mechanically simple. Both copies of round 2 had it, and
round 3's simplification preserved it. It's a one-character fix:

```python
detected_lang = segments[0].get("lang", "nl")
```

**Recommendation:** apply the one-character fix. After applying, verify
on a known non-Dutch sample that `asr_engine_b.json` is **not** written
when `--ensemble` is on — that's the cheap acceptance test for whether
the gating actually engaged.

### Deviation #2 — Per-window language gating is still not implemented

**Severity:** MINOR (acknowledged deferral, listed for completeness)
**File:** `pipeline/asr_engines.py:245-261`

Even after Deviation #1 is fixed, the function only differentiates
"uniformly Dutch" from "not Dutch". The FSD §2 stage 2 calls for 30 s
windowed language ID with mixed-language audio routed per-window. The
codefixes3 plan correctly notes this is deferred pending the P1-3
subprocess refactor. No action required this round; flagged so it isn't
forgotten.

---

## 3. Plan vs. Implementation

`codefixes3.md` is the plan for this round. Compared to that plan:

| Plan Item | Planned | Actual | Status |
|-----------|---------|--------|--------|
| Fix #1: multiplicative glossary | `prob *= (1.0 + weight)` + comment | `rover.py:118-128` matches | ✅ |
| Fix #2: stderr log on batch failure | `print(f"Post-correct batch {i // _BATCH_SIZE}: …", file=sys.stderr)` | `postcorrect.py:413` matches | ✅ |
| Fix #3: prompt restructure | System content into system role, `cache_control` removed | Both backends match | ✅ |
| Fix #4: accumulate usage | `last_usage` dict with running totals | `postcorrect.py:219-230` matches | ✅ |
| Fix #5: simplify gating, remove `rover_eligible` | Two-branch detection; flag deleted | Code structure matches; **but the key-name bug carries through** | ⚠️ Plan implemented as written; the plan didn't notice the underlying key bug |
| Fix #6: merge to nearest | Center-distance loop | `rover.py:213-228` matches | ✅ |
| Fix #7: more transient types | Add `APITimeoutError`, `InternalServerError` | `postcorrect.py:247-254` matches | ✅ |
| 4.2: safe `total_end` | `s.get("end", 0.0)` | `asr_engines.py:259` matches | ✅ |
| 4.3: regex version | `re.match(r"(\d+)\.(\d+)", ...)` with `(0,0)` fallback | `diarize.py:55-57` matches | ✅ |
| 7.1: drop dead import | Remove `from typing import Any` | Verified gone | ✅ |
| 7.2: fix doc comment | Drop "term" from comment | `glossary.py:131` matches | ✅ |
| 7.5: drop no-op | Remove `dynapse -> Dynapse`, keep `dinapse -> Dynapse` | `glossary.py:150-151` matches | ✅ |

**Undocumented deviations from the plan:** none. The plan is honest;
the failure mode in Fix #5 is a missed root cause, not a deviation.

---

## 4. Edge Cases & Safety

### 4.1 `_rebuild_segments` raises IndexError if Engine A produces zero segments

**Severity:** MINOR
**File:** `pipeline/rover.py:213-228`

If Engine A's `segments_out` is empty (zero-segment Engine A output) and
Engine B produced unmatched words, the merge loop does:

```python
remaining = reconciled_words[word_ptr:]
for w in remaining:
    ...
    best_idx = 0
    best_dist = float("inf")
    for i, seg in enumerate(segments):  # empty iteration
        ...
    segments[best_idx]["words"].append(w)  # IndexError on empty list
```

`segments` is empty, the inner loop never runs, `best_idx` stays at 0,
the access raises `IndexError`. Engine A producing zero segments is rare
but possible (very short / silent audio, ASR error). Caller sees a stack
trace from ROVER, not a graceful fall-back.

**Recommendation:** before the merge loop, guard:

```python
if not segments and remaining:
    segments.append({
        "id": 0,
        "start": remaining[0]["start"],
        "end": remaining[-1]["end"],
        "text": " ".join(w["word"] for w in remaining),
        "lang": "nl",
        "words": list(remaining),
    })
    return segments
```

Or skip the merge entirely on empty `segments` (the file just becomes
verbatim Engine A, which is what ROVER produces today on no-ensemble).

### 4.2 Merged B-word loses temporal placement within the host segment

**Severity:** MINOR
**File:** `pipeline/rover.py:225-228`

Each unmatched B-word is appended to the end of the host segment's
`text` and `words`. If the B-word's true timestamp is *before* the host
segment's start (e.g., orphan at 5 s gets merged into a 10-30 s
segment), the rendered transcript shows it at the segment's end (30 s)
instead of where it was actually spoken. The segment's `end` is updated
when needed; the segment's `start` is **not** updated when `w["start"]
< segments[best_idx]["start"]`, so:

- Segment range remains `[10, 30]` even though it now contains a word
  with `start=5`.
- Word list contains a word whose `start` predates the segment's
  `start`.

For the typical case (B-word arrives between two A-words within a
segment's range), the placement is fine. For orphans before/after the
A-timeline, the placement is wrong but the word is at least preserved.
Cosmetic for post-correction quality; misleading if any downstream tool
trusts segment ranges.

**Recommendation:** also widen `start` symmetrically:

```python
if w["start"] < segments[best_idx]["start"]:
    segments[best_idx]["start"] = w["start"]
```

### 4.3 `result_b` is assigned but never used

**Severity:** INFO
**File:** `pipeline/asr_engines.py:233-238`

```python
if ensemble:
    nl_windows = _detect_nl_windows(result_a.get("segments", []))
    result_b = None
    if nl_windows:
        result_b = run_engine_b(...)
```

After Fix #5 removed the `rover_eligible` block, `result_b` has no
remaining reader. The side effect (`run_engine_b` writing
`asr_engine_b.json`) is what the next pipeline stage consumes, so this
is harmless dead code, but it suggests the function returned to be
discarded. Drop the assignment, or comment-justify it ("side effect is
the file write; result discarded").

### 4.4 `_detect_nl_windows` accepts a `window_s` parameter that is no longer used

**Severity:** INFO
**File:** `pipeline/asr_engines.py:245`

After the simplification, the `window_s` parameter is dead. It survives
because the previous mixed-language branch used it. Either remove it
from the signature or note that it is reserved for a later
windowing implementation.

### 4.5 Glossary `[term]` section is parser-supported but un-documented after 7.2

**Severity:** INFO
**File:** `pipeline/glossary.py:25-26,131`

`SECTION_WEIGHTS` includes `"term": 1` and `DEFAULT_SECTION = "term"`,
so users can still write `[term]` sections in their own glossary files
and the loader will weight them. The seeded default's comment now reads
"brand > person > place" — accurate for the seeded sections, but a user
reading that comment will assume `[term]` is unsupported. Either keep
the seeded default minimal (current state) and add a one-liner like
"`[term]` also supported, weight 1," or add a commented placeholder
showing how to use it. Not a bug; documentation drift.

---

## 5. Concurrency & Platform Issues

No new concurrency or platform issues this round. Everything still runs
single-threaded per stage with Docker stages serialised by the bash
orchestrator. Ollama lifecycle and GPU memory management are unchanged.

The pyannote monkeypatch (now version-gated) re-applies on every
`_run_pyannote` invocation. With the pin to `huggingface_hub==0.30.2`,
the gate is False so nothing happens; if the pin drifts, the gate is
True and the same patch is reapplied to already-patched modules.
Reapplying the patch wraps `_remap_token` around itself (since the
module-level `hf_hub_download` was already replaced), but only after
unwrapping `__wrapped__`; in practice each call re-creates the wrapper
fresh and replaces the bound, so no chain accumulates. Safe for the
synchronous one-shot usage we have today.

---

## 6. Error Handling

### 6.1 `_retry` exception coverage

**Severity:** INFO
**File:** `pipeline/postcorrect.py:241-256`

Caught: `URLError`, `socket.timeout`, `ConnectionError`, `TimeoutError`,
`openai.RateLimitError`, `openai.APIConnectionError`,
`openai.APITimeoutError`, `openai.InternalServerError`.

`InternalServerError` covers the 5xx range in OpenAI SDK 1.54 (it's the
catch-all 5xx subclass), so 502/503/504 are also caught transitively.
Coverage is now appropriate.

The non-transient path (the hot try/except on the per-batch call in
`run()`) now logs the swallowed exception (Fix #2). Combined: a Z.ai
validation rejection no longer disappears silently, and a transient
network blip retries with backoff.

### 6.2 Glossary parser silently drops malformed lines

**Severity:** INFO (pre-existing, surfaced because doc was edited this
round)
**File:** `pipeline/glossary.py:62-68`

Lines that don't split into two parts on `->` / `→` are silently
skipped:

```python
parts = _SEP_RE.split(line, maxsplit=1)
if len(parts) != 2:
    continue
wrong, right = parts[0].strip(), parts[1].strip()
if not wrong or not right:
    continue
```

A user editing the file by hand and typo-ing the arrow will lose
entries silently. Not a regression from this round, but worth thinking
about: a `print(f"glossary: skipped malformed line: {raw!r}",
file=sys.stderr)` is two minutes of work and would save a frustrating
debug session.

---

## 7. Code Quality

Substantive observations only.

### 7.1 `_score_word` returns a 4-tuple where step 3 is now correctly redundant

`pipeline/rover.py:103-132`. The tuple `(prob, logprob, is_glossary,
engine)` and the cascade in `_reconcile_cluster` correctly handle the
multiplicative case: `is_glossary` is structurally redundant because the
score already encodes glossary preference, but it's kept for legibility
and future flexibility. Comment at lines 118-122 documents this. Good.

### 7.2 Inner WhisperX script string-templates user-supplied paths via `!r`

`pipeline/asr_engines.py:64-135`. Pre-existing pattern (P1-3, deferred),
not a regression. Listed only to confirm it hasn't gotten worse.
Filenames containing single backslashes or quote characters could
produce malformed Python via `!r`; in the deployment context (Docker
volume, basename mangled by the orchestrator) this is unlikely.

### 7.3 `result_b` and `window_s` dead code

Already covered in §4.3 and §4.4.

### 7.4 Glossary `dnai -> de AI` may produce odd post-correction prompts

`pipeline/glossary.py:150`. Spec lists this as a known mishearing. The
glossary's role in the LLM system prompt is to tell the model what the
correct forms are; "de AI" with an embedded space and capital is an
unusual rewrite target. Not a bug; flagged because if post-correction
ever produces "de AI" where the speaker said "DNAI" (the acronym for
something), that's a paraphrase, not a phonetic correction.

### 7.5 Glossary loader's `DEFAULT_SECTION = "term"` vs seeded `_DEFAULT_GLOSSARY` no longer including `[term]`

`pipeline/glossary.py:26,131`. As covered in §4.5. The default-section
constant is fine; only the seeded file's documentation went stale. Low
priority.

---

## 8. Summary

| Category | Critical | Major | Minor | Info |
|----------|----------|-------|-------|------|
| Spec conformance | 0 | 1 (Dev #1 key mismatch) | 1 (Dev #2 per-window deferred) | 0 |
| Plan conformance | 0 | 0 | 1 (Fix #5 plan didn't catch root cause) | 0 |
| Correctness | 0 | 0 | 2 (4.1 IndexError, 4.2 temporal placement) | 2 (4.3, 4.4) |
| Safety | 0 | 0 | 0 | 0 |
| Concurrency | 0 | 0 | 0 | 0 |
| Error handling | 0 | 0 | 0 | 2 (6.1 OK, 6.2 silent drop) |
| Code quality | 0 | 0 | 0 | 5 (7.1-7.5) |

**Critical:** none.
**Major (1):** key-mismatch in `_detect_nl_windows` makes Engine B run
on every ensemble call regardless of language.
**Minor (4):** scattered.
**Info (9):** documentation drift, dead code, defensive style.

Compared to REVIEW3:

- 4 majors → **1 major** (3 majors closed: glossary additive, prompt
  caching, sidecar usage, silent swallow).
- 12 minors → 4 minors + 9 info. Net trend: down.

---

## 9. Recommendation

**PASS WITH NOTES** — proceed to Task 14 (README), then re-verification.
Apply the one-character fix below in the same change as the README
update.

### Must-fix

1. **`asr_engines.py:257`** — change `segments[0].get("language", "nl")`
   to `segments[0].get("lang", "nl")`. One-character change, restores
   the intended behaviour of Fix #5 (skip Engine B on non-Dutch
   audio). Without this, the only thing the simplified gating buys is
   removing dead code; the language-skip path is unreachable.

### Should-fix in the same pass

2. **`rover.py:213-228`** (§4.1) — guard the empty-`segments` case so
   ROVER doesn't crash if Engine A produces zero segments and Engine B
   has unmatched words.
3. **`rover.py:225-228`** (§4.2) — also widen `start` symmetrically
   when a B-word is merged earlier than the host segment's start. Two
   lines.
4. **`asr_engines.py:233-238`** (§4.3) — drop the unused `result_b`
   assignment, or comment-justify the side-effect-only call.

### Nice-to-have

- Remove the unused `window_s` parameter from `_detect_nl_windows`
  (§4.4).
- Add a "skipped malformed line" warning in the glossary parser (§6.2).
- Restore a commented `[term]` placeholder to the seeded glossary so
  users know the section is supported (§4.5).

### Then

- Land **Task 14** (README update). New flags (`--cloud-correct`,
  `--ensemble`, `--context`, `--glossary`, `--diarize-model`,
  `--refresh-model`, `--keep-intermediates`, `--no-*`) and the
  `~/.config/whisper/` files (`glossary.txt`, `zai-key`,
  `.glm-resolved`) are still undocumented. Pending across all four
  reviews.

### Then re-verify

Real-world acceptance, end-to-end:

- The three samples at `--quality perfect` and `--quality perfect
  --cloud-correct`.
- A non-Dutch sample at `--quality perfect`. After the one-character
  fix, `asr_engine_b.json` should **not** appear in the scratch dir.
  This is the regression test for Fix #5.
- A sample with `--prompt /tmp/prompt.txt` (path outside `INPUT_DIR`)
  to confirm `DOCKER_STAGE_EXTRA` plumbing.
- A run with `~/.config/whisper/hf-token` removed, verifying the
  warning prints and rendering still completes (single SPEAKER_00
  block).
- A `--quality perfect --no-correct` run to confirm `--no-*` overrides.

End of review.
