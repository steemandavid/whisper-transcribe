# Phase 2 Code Review #5 — Whisper Pipeline v2 (post-`codefixes4.md` + Task 14)

**Document ID:** WHISPER-REVIEW-P2-005
**Reviewer:** Code Review Agent (Claude)
**Date:** 2026-05-06
**Scope:** Fifth-pass review after `codefixes4.md` addressed `REVIEW4.md` findings; also covers the now-completed Task 14 (`README.md`) update.
**FSD Reference:** `HANDOFF.md` (sections 1–8)
**Prior Reviews:** `REVIEW1.md`, `REVIEW2.md`, `REVIEW3.md`, `REVIEW4.md`
**Commit Reviewed:** `2b6ad42` (working tree, uncommitted v2 work; latest pipeline mtime 2026-05-06 18:09–18:10, README 18:14)

---

## Verdict: **PASS**

This is the first round where the pipeline cleanly meets the FSD on every
must-fix path, with no remaining major bugs and no critical findings. The
one-character key fix from REVIEW4 landed (`asr_engines.py:253` now reads
`segments[0].get("lang", "nl")`), so Engine B is finally skipped on
non-Dutch audio for the first time across five reviews. The empty-segments
guard, symmetric start-widening, dead-code cleanup, and glossary
malformed-line warning are all in place. Task 14 (README) is genuinely
done — comprehensive, accurate, and consistent with the implementation.
A handful of minor findings remain (one wrong README claim about `--help`
seeding the glossary; one cosmetic `__import__("sys")` in glossary.py;
one inconsistency in the README's "Pipeline summary" example output) but
none block proceeding. The code is converged enough to commit.

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
| `pipeline/asr_engines.py` | Yes — key fix, dead code cleanup | Engine A/B + gating |
| `pipeline/rover.py` | Yes — empty-segments guard, start widening | ROVER reconciliation |
| `pipeline/glossary.py` | Yes — malformed-line warning, `[term]` placeholder | Glossary loader |
| `transcribe` | Yes — minor: `--diarize-model` doc default `community-1 → 3.1` | Bash orchestrator |
| `README.md` | **Yes — Task 14 completed** | User-facing docs |
| `pipeline/postcorrect.py`, `diarize.py`, `preprocess.py`, `prompt_builder.py`, `render.py`, `artifacts.py`, `__init__.py`, `Dockerfile` | No — verified mtime unchanged | (unchanged this round) |

---

## 1. Coverage Analysis

### Items claimed by `codefixes4.md`

| Source | Claim | Verdict |
|--------|-------|---------|
| REVIEW4 Dev #1 | `_detect_nl_windows` reads `"lang"` not `"language"` | **DONE** — `asr_engines.py:253` reads `segments[0].get("lang", "nl")`. Engine B is now genuinely skipped on uniform non-nl audio for the first time. |
| REVIEW4 §4.1 | Empty-segments guard in `_rebuild_segments` | **DONE** — `rover.py:216-226` emits a single segment when Engine A is empty and B has unmatched words; returns early. |
| REVIEW4 §4.2 | Symmetric `start` widening on merged B-words | **DONE** — `rover.py:238-239` mirrors the existing `end` widening. |
| REVIEW4 §4.3 | Drop dead `result_b` assignment | **DONE** — `asr_engines.py:228-236` calls `run_engine_b` for its file-write side effect with a comment justifying it. |
| REVIEW4 §4.4 | Drop unused `window_s` parameter | **DONE** — `asr_engines.py:241` signature is now `_detect_nl_windows(segments)`. |
| REVIEW4 §4.5 | Commented `[term]` placeholder in seeded glossary | **DONE** — `glossary.py:164-166`. |
| REVIEW4 §6.2 | Glossary parser warns on malformed lines | **DONE** — `glossary.py:64`. (Implementation nit: uses `__import__("sys").stderr`; a top-of-file `import sys` would be cleaner.) |

### Item not claimed but verified done

- **Task 14 — README.md update.** Pending across all four prior reviews.
  Now comprehensive (~520 lines) and structurally accurate. Covers
  architecture, directory layout, all 24 CLI flags, quality presets,
  glossary format and locations, auto-prompt pipeline, post-correction,
  Z.ai setup, building from scratch, debugging, and troubleshooting.
  Three small accuracy issues — see Deviation #2.
- **`transcribe:26`** — `--diarize-model community-1 or 3.1 (default:
  3.1)` (docstring previously said `default: community-1`, contradicting
  the actual `DIARIZE_MODEL="3.1"` default). Quietly fixed; not in
  `codefixes4.md`. Net positive.

### Items still on the deferred list (acknowledged)

| Item | Status |
|------|--------|
| P1-3 (ASR subprocess refactor) | Deferred. |
| P1-16 / P1-17 (AudioSR / DeepFilterNet full installs) | Deferred — silently no-op. End-of-run summary correctly surfaces "skipped". |
| Per-30 s window language detection | Deferred — depends on P1-3. The simplified `_detect_nl_windows` handles uniform-language correctly; fine-grained windowing remains future work. |

---

## 2. Deviation Report

### Deviation #1 — `README.md` claims `transcribe --help` triggers glossary seeding

**Severity:** MINOR (doc bug, not behaviour bug)
**File:** `README.md:432`

```markdown
# The glossary is auto-seeded on first run, or create manually:
~/claudecode/projects/whisper/transcribe --help  # triggers seed
```

`transcribe --help` actually does `head -10 "$0"; exit 0` (`transcribe:90`).
It prints the first 10 lines of the script and exits. It never invokes
`pipeline.glossary --seed`. The glossary is also not auto-seeded by any
other path in `transcribe` — seeding only happens when the user
explicitly runs:

```bash
docker run --rm whisper-transcribe:2.0 pipeline.glossary --seed
```

(or copies a glossary file in by hand).

**Recommendation:** either replace the README line with the actual seed
command (`docker run --rm whisper-transcribe:2.0 pipeline.glossary
--seed`), or wire up `transcribe --seed-glossary` and have `--help`
print a longer usage block that lists it. The current state is a small
trap for new users.

### Deviation #2 — `README.md` "Pipeline summary" example shows contradictory output

**Severity:** MINOR (doc bug)
**File:** `README.md:200-205`

```
=== Pipeline summary ===
Ran:     preprocess asr (large-v3) ensemble (engine B + ROVER) diarize post-correct (12 segments fixed)
Skipped: ensemble (--ensemble off)
```

The bash logic at `transcribe:585-590` emits exactly one of three
`STAGE_SKIPPED` entries for ensemble (`engine B + ROVER` ran / `engine
B unavailable` / `--ensemble off`), so a real run cannot produce both
"ensemble (engine B + ROVER)" in `Ran:` and "ensemble (--ensemble off)"
in `Skipped:`. The example is a manually-constructed paste that mixes
two separate runs.

**Recommendation:** show the output for a single run consistently, or
add a "(example combined for illustration)" note. Two independent
examples would be clearer.

### Deviation #3 — README architecture diagram says ROVER is "(optional)"

**Severity:** INFO (already accurate enough)
**File:** `README.md:48`

The bash orchestrator runs the ROVER stage only when `--ensemble`
produced `asr_engine_b.json`; otherwise it copies `asr_engine_a.json`
as `rover.json`. The "optional" annotation is technically correct.
Listed only because a reader might wonder what happens to `rover.json`
in the non-ensemble case — a one-line note ("when `--ensemble` is off,
Engine A's output is used as-is") would prevent confusion.

---

## 3. Plan vs. Implementation

`codefixes4.md` is the plan for this round. Compared to that plan:

| Plan Item | Planned | Actual | Status |
|-----------|---------|--------|--------|
| Key fix in `_detect_nl_windows` | `"language"` → `"lang"` | `asr_engines.py:253` matches | ✅ |
| Empty-segments guard | Early-return single segment | `rover.py:216-226` matches | ✅ |
| Symmetric start widening | If `w["start"] < seg["start"]`, widen | `rover.py:238-239` matches | ✅ |
| Drop `result_b` and `window_s` | Both removed; side-effect comment added | `asr_engines.py:228-241` matches | ✅ |
| Glossary malformed-line warning | Add stderr print | `glossary.py:64` matches | ✅ |
| `[term]` placeholder | Commented-out lines in seeded default | `glossary.py:164-166` matches | ✅ |

**Undocumented changes:** `transcribe:26` docstring was also updated
this round (`(default: community-1)` → `(default: 3.1)`), which was a
correctness fix for the documentation but not listed in
`codefixes4.md`. Net positive, no concern.

**Plan status:** `codefixes4.md` reads "README / Task 14 — pending, to
be done next." The README *was* in fact completed in the same session
(timestamp 18:14 vs. `codefixes4.md` 18:12). The plan was honest about
the as-of-writing state; the README was finished after the doc was
written. No deviation.

---

## 4. Edge Cases & Safety

### 4.1 `_detect_nl_windows` defaults to `"nl"` when segments lack the `"lang"` key

**Severity:** INFO
**File:** `pipeline/asr_engines.py:253`

If for any reason `segments[0]` is missing `"lang"` (the inner script
always writes it, but a custom-built segment list could omit it), the
function defaults to `"nl"` and returns the full audio range, causing
Engine B to run. This is a permissive default that matches user intent
for the common case (Dutch transcription) and harmlessly degrades to
"Engine B runs once and writes a JSON file" for other cases. Not a
concern; flagged for transparency.

### 4.2 `_rebuild_segments` empty-segments path uses `lang: "nl"` unconditionally

**Severity:** INFO
**File:** `pipeline/rover.py:223`

When Engine A produced zero segments, the synthetic catch-all segment
is labelled `"lang": "nl"` regardless of what Engine B actually saw.
Engine B always runs with `language="nl"` (per `asr_engines.py:234`),
so this is consistent in practice — but conceptually the synthetic
segment should carry whatever language label is most accurate, not a
hardcoded one. Cosmetic.

### 4.3 Symmetric start widening can compress segment ranges

**Severity:** INFO
**File:** `pipeline/rover.py:238-241`

When a merged B-word's `[start, end]` spans more than the host
segment, both `start` and `end` widen. If multiple B-words are merged
into the same host, the host's effective range can grow significantly
beyond Engine A's original timing. Diarization speaker assignment uses
segment centre, so a segment that started as `[10, 30]` and grows to
`[5, 35]` may have its centre shift from 20 to 20 (unchanged in this
example), but in skewed cases the centre can drift and the speaker
overlap calculation in `render.py:_assign_speakers` may pick a
different speaker. Practical impact small — the merged words were
already orphans without a tight A-segment home — but worth noting if
diarization quality drops.

### 4.4 Glossary malformed-line warning fires for every load

**Severity:** INFO
**File:** `pipeline/glossary.py:62-65`

Every call to `Glossary.load` re-parses the file and re-prints the
warning for each malformed line. The orchestrator calls `Glossary.load`
in three places (prompt builder, ROVER, post-correct), so a
malformed-line user file will produce 3× warnings per run. Cosmetic;
deduplication would require state across calls.

---

## 5. Concurrency & Platform Issues

No new concurrency or platform issues this round. The pipeline is still
single-threaded per stage with Docker stages serialised by the bash
orchestrator. The pyannote monkeypatch (version-gated) is unchanged.

The `select_ollama_model` function in bash and `OllamaCorrector._auto_select`
in Python remain duplicated. Acknowledged as low priority since the
two execute in different host vs. container contexts.

---

## 6. Error Handling

### 6.1 `__import__("sys").stderr` works but is unidiomatic

**Severity:** INFO
**File:** `pipeline/glossary.py:64`

```python
print(f"glossary: skipped malformed line: {raw!r}", file=__import__("sys").stderr)
```

Functional, but `import sys` at the top of the module would be the
idiomatic Python and avoid the import-on-every-call overhead. The
overhead is negligible (`__import__` is cached after first call), and
the file already has `import re` at top — this is purely a style
choice. Listed only because the rest of the file is clean Python.

### 6.2 No regressions in existing error paths

`postcorrect.py:412-414` (Fix #2 from round 3) still logs swallowed
exceptions. `_retry` (round-3 Fix #7) still covers the right transient
types. `_run_pyannote` monkeypatch still version-gated. No paths
regressed this round.

---

## 7. Code Quality

Substantive observations only.

### 7.1 README inaccuracies

Already covered as Deviation #1, #2, #3.

### 7.2 README at line 290 lists `[term]` as a supported section, glossary doc at line 132 lists only three

`README.md:289`: "Sections: [brand], [person], [place], [term]" (4
sections).
`pipeline/glossary.py:132`: "Sections weight the entry: brand >
person > place" (3 sections).

These are now consistent in spirit (the glossary file's `[term]`
placeholder at lines 164-166 documents support for term entries), but
the inline comment at line 132 still doesn't mention `term`. Either
extend it ("brand > person > place > term") or drop "term" from the
README's section list.

### 7.3 `transcribe -h` / `--help` prints only 10 lines of header

`transcribe:90`: `head -10 "$0"; exit 0`. The first 10 lines of the
file are the shebang + 8 lines of header comment + 1 line. Of the ~24
flags documented in the comment block (lines 6-32), only the first 4
are visible. Combined with Deviation #1 (README pointing users at
`--help`), this is mildly user-hostile.

**Recommendation:** either bump to `head -33 "$0"` (or wherever the
header ends), or print a `cat <<EOF ... EOF` usage block. Two-line
change.

### 7.4 No new code-quality regressions

Round 4 was a cleanup round; no new structural patterns introduced.

---

## 8. Summary

| Category | Critical | Major | Minor | Info |
|----------|----------|-------|-------|------|
| Spec conformance | 0 | 0 | 0 | 1 (per-window deferred) |
| Plan conformance | 0 | 0 | 0 | 0 |
| Correctness | 0 | 0 | 0 | 3 (4.1, 4.2, 4.3) |
| Safety | 0 | 0 | 0 | 0 |
| Concurrency | 0 | 0 | 0 | 0 |
| Error handling | 0 | 0 | 0 | 2 (4.4 redundant warning, 6.1 idiom) |
| Code quality (incl. docs) | 0 | 0 | 3 (Dev #1 wrong help, Dev #2 contradictory example, §7.3 short help) | 1 (§7.2 wording) |

**Critical:** none.
**Major:** none.
**Minor (3):** all in user-facing documentation.
**Info (7):** scattered.

Compared to the trajectory:

| Round | Major | Minor | Info | Verdict |
|-------|-------|-------|------|---------|
| REVIEW1 | many P0/P1 | many | — | many issues |
| REVIEW2 | 3 NEW-P0 + others | many | — | regressions |
| REVIEW3 | 4 | 12 | — | PASS WITH NOTES |
| REVIEW4 | 1 | 4 | 9 | PASS WITH NOTES |
| **REVIEW5** | **0** | **3** | **7** | **PASS** |

---

## 9. Recommendation

**PASS** — proceed to commit and real-world re-verification.

The code path for `--quality perfect` end-to-end with both `--correct`
and `--cloud-correct` should now behave per the FSD. The five-round
review cycle has converged.

### Optional follow-ups (no blockers)

These would all fit in a single small commit:

1. **README fix:** replace `transcribe --help # triggers seed` with the
   actual seed command, or implement a real `transcribe --seed-glossary`
   wrapper. (Deviation #1)
2. **README example fix:** make the "Pipeline summary" example
   internally consistent. (Deviation #2)
3. **`transcribe --help` output:** print the full header (or a curated
   usage block) instead of `head -10`. (§7.3)
4. **Glossary inline comment:** mention `[term]` so the in-file doc
   matches the README. (§7.2)
5. **`glossary.py:64`:** replace `__import__("sys").stderr` with a top-
   of-file `import sys`. (§6.1)

### Re-verification matrix (recommended before declaring v2.0 stable)

The same checklist from REVIEW4 § 9 still applies; the new must-test
case is the language-gating fix:

- [ ] Run a known **non-Dutch** sample (e.g. an English meeting) with
      `--ensemble`. After the fix, `asr_engine_b.json` should **not**
      appear in the scratch dir, and the pipeline summary should report
      "ensemble (engine B unavailable)" or similar.
- [ ] Run the three test recordings at `--quality perfect` and
      `--quality perfect --cloud-correct`. Compare cleaned output to
      the legacy v1 transcripts on the eight known mishearings (Fluke,
      Anixter, CommScope, ConnectWise, wiremap, RustOleum, Bjorn,
      Microsoft Teams).
- [ ] Run with `~/.config/whisper/hf-token` removed. Pipeline should
      complete with a single SPEAKER_00 block and a stderr warning.
- [ ] Run with `--prompt /tmp/foo.txt` (path outside `INPUT_DIR`) to
      confirm `DOCKER_STAGE_EXTRA` plumbing.
- [ ] Run `--quality perfect --no-correct` to confirm `--no-*`
      overrides.
- [ ] Run with a deliberately-malformed glossary (one line missing the
      `->`) and confirm the new stderr warning fires.

### Items genuinely deferred (next phase)

- P1-3 — ASR subprocess refactor.
- P1-16 / P1-17 — AudioSR / DeepFilterNet full installs (need Rust
  toolchain in the builder image).
- Per-30 s window language detection (depends on P1-3).
- Audio-aware LLM second-pass on `[?]` segments — explicitly out of
  scope per HANDOFF §6.

End of review.
