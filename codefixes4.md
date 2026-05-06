# Code Fixes #4 — Session 2026-05-06 (REVIEW4.md corrections)

Fixes applied from the fourth code review (REVIEW4.md). REVIEW4 was a "PASS WITH
NOTES" verdict — the cleanest round yet, with one major bug and a handful of minor
edge cases.

---

## Must-fix (1)

### Key mismatch in `_detect_nl_windows` — Engine B always runs

**Source**: REVIEW4 Deviation #1
**File**: `pipeline/asr_engines.py:257` (pre-fix line number)
**Severity**: MAJOR — silently un-fixes the language gating from round 2/3

The inner WhisperX subprocess writes each segment with key `"lang"`:

```python
segments_out.append({{
    ...
    "lang": seg.get("language", top_lang),
    ...
}})
```

But `_detect_nl_windows` reads:

```python
detected_lang = segments[0].get("language", "nl")
```

The `.get("language", ...)` always misses → falls through to default `"nl"` →
returns `[(0.0, total_end)]` → Engine B runs on every ensemble call regardless of
detected language. An English-only meeting with `--ensemble` still runs the Dutch
fine-tune over English audio.

**Fix**: One-character change — `"language"` → `"lang"`:

```python
detected_lang = segments[0].get("lang", "nl")
```

This restores the intended behaviour: Engine B is skipped when the detected
language is not Dutch.

---

## Should-fix (3)

### Empty segments guard in ROVER

**Source**: REVIEW4 §4.1
**File**: `pipeline/rover.py:213-228`

If Engine A produces zero segments (very short/silent audio, ASR error) but
Engine B has unmatched words, the merge loop would crash with `IndexError` because
`segments` is empty and `segments[best_idx]` accesses index 0.

**Fix**: Added early return guard before the merge loop:

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

### Temporal widening on merged B-words

**Source**: REVIEW4 §4.2
**File**: `pipeline/rover.py:238`

When a merged B-word starts before the host segment's start, the segment range
was not widened, producing a segment with `[start=10, end=30]` containing a word
at `start=5`.

**Fix**: Added symmetric start widening:

```python
if w["start"] < segments[best_idx]["start"]:
    segments[best_idx]["start"] = w["start"]
```

### Dead `result_b` assignment removed

**Source**: REVIEW4 §4.3
**File**: `pipeline/asr_engines.py:233-239`

After round 3 removed the `rover_eligible` block, `result_b` had no remaining
reader. The side effect (writing `asr_engine_b.json`) is what the downstream
ROVER stage consumes.

**Fix**: Dropped the `result_b =` assignment. Added a comment noting the
side-effect-only nature of the call. Also removed the unused `window_s` parameter
from `_detect_nl_windows` (§4.4).

---

## Nice-to-have (3)

### Glossary malformed-line warning

**Source**: REVIEW4 §6.2
**File**: `pipeline/glossary.py:63-64`

Lines that don't split into two parts on `->` / `→` were silently skipped. A user
editing the file by hand and typo-ing the arrow would lose entries without knowing.

**Fix**: Added stderr warning:

```python
print(f"glossary: skipped malformed line: {raw!r}", file=__import__("sys").stderr)
```

### Commented `[term]` placeholder in seeded glossary

**Source**: REVIEW4 §4.5
**File**: `pipeline/glossary.py:164-165`

After round 2 removed the empty `[term]` section, users had no indication that
`[term]` is a supported section (it's still in `SECTION_WEIGHTS` and is the
`DEFAULT_SECTION`).

**Fix**: Added commented placeholder:

```
# [term]
# Add domain-specific terms here (weight 1, lowest priority).
# example -> Example
```

### Removed unused `window_s` parameter

**Source**: REVIEW4 §4.4
**File**: `pipeline/asr_engines.py:245`

After simplifying `_detect_nl_windows` to two branches, the `window_s` parameter
was dead code from the removed mixed-language branch.

**Fix**: Removed from function signature.

---

## Files changed

| File | Changes |
|------|---------|
| `pipeline/asr_engines.py` | Key fix (`"lang"`), dead `result_b`, dead `window_s`, side-effect comment |
| `pipeline/rover.py` | Empty-segments guard, temporal start widening |
| `pipeline/glossary.py` | Malformed-line warning, `[term]` placeholder |

## Remaining items (unchanged)

- README / Task 14 — pending, to be done next
- P1-3: ASR subprocess refactor — deferred
- P1-16 / P1-17: AudioSR / DeepFilterNet — deferred
- Per-30s-window language detection — deferred (depends on P1-3)
