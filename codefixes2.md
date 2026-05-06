# Code Fixes #2 — Session 2026-05-06 (REVIEW2.md corrections)

Fixes applied from the second code review (REVIEW2.md), covering new P0 regressions,
incomplete P1 fixes, and new issues introduced by the first round of fixes.

## NEW-P0 — Regressions / still broken

### NEW-P0-1. Auto-prompt path falls back to raw LLM output

**File**: `transcribe:383-397`

When `prompt_builder` fails silently (`2>/dev/null || true`), `REFINED_PROMPT` is empty
and `AUTO_PROMPT_TEXT` retains the raw Ollama output (ANSI-stripped only). This is written
to `final_prompt.txt` and fed to Whisper as `initial_prompt` — the exact anti-pattern the
v2 rewrite was meant to eliminate.

**Fix**: When `REFINED_PROMPT` is empty, set `AUTO_PROMPT=false` and log a warning.
The fall-through path that wrote unsanitised text is removed entirely.

### NEW-P0-2. Per-chunk language gating does not actually gate

**Files**: `pipeline/asr_engines.py`, `pipeline/rover.py`

Two problems:
1. `_detect_nl_windows` reads `seg.get("language", "nl")` per segment, but WhisperX
   always populates this with the forced language — every segment reports "nl", so the
   function returns `[(0.0, total_end)]` covering the whole audio.
2. The `rover_eligible` flag set on `language_chunks` is never read by `pipeline/rover.py`.

**Fix**:
- Inner WhisperX script now uses `top_lang = result.get("language", language or "nl")`
  and passes per-segment language through correctly (faster-whisper does detect per-segment
  language when `language=None`).
- `_detect_nl_windows` now returns `[]` when the detected language is uniformly non-nl,
  causing Engine B to be skipped entirely for non-Dutch audio.
- For uniform nl audio, returns the full audio (Engine B runs as before).
- For mixed language, builds 30s windows and checks majority language.

Full per-window Engine B execution (splitting audio and running Engine B per nl window)
is deferred pending the P1-3 subprocess refactor.

### NEW-P0-3. Pin vs patch contradiction in diarize stack

**File**: `pipeline/diarize.py:48-75`

`huggingface_hub==0.30.2` (pre-1.0) still has `use_auth_token`, so the monkeypatch
is dead code. Reading the code, you can't tell whether the pin or the patch is doing
the work.

**Fix**: Gate the monkeypatch on the installed version:
```python
_hf_version = tuple(int(x) for x in _hf.__version__.split('.')[:2])
if _hf_version >= (1, 0):
    # apply monkeypatch
```
With the pinned 0.30.2, the patch is skipped. If the pin drifts (user runs
`pip install -U`), the patch activates. Clear comments explain the relationship.

---

## P1 — Incomplete fixes

### P1-6 (partial). Unmatched B words out of chronological order

**File**: `pipeline/rover.py:210-228`

Previous fix appended unmatched B words as a single trailing segment. Words from
second 12 of a 30-minute recording appeared at the end of the transcript.

**Fix**: Each unmatched B word is now inserted as its own segment, then all segments
are sorted by `start` time and re-numbered:
```python
segments.sort(key=lambda s: s["start"])
for i, seg in enumerate(segments):
    seg["id"] = i
```

### P1-10. Prompt caching against Z.ai

**File**: `pipeline/postcorrect.py:200-215`

Same system prompt (including full glossary) was re-sent per batch with no caching.

**Fix**: Added `cache_control: {"type": "ephemeral"}` to both system and user messages
in ZaiCorrector. Z.ai mirrors the OpenAI/Anthropic caching pattern.

### P1-11. Token usage to sidecar JSON

**File**: `pipeline/postcorrect.py:196,223-226,414-418`

Token usage was logged to stdout via `print()`, mixed with stage output and not
machine-readable.

**Fix**: `ZaiCorrector` now stores `self.last_usage` dict after each call. The `run()`
function writes this to `cleaned_usage.json` alongside the main output and includes
it in the result dict.

---

## NEW-P1 — Issues introduced by round-1 fixes

### NEW-P1-1. `_score_word` reads `avg_logprob` that `_flatten_words` never sets

**File**: `pipeline/rover.py:125`

`_flatten_words` stores `logprob` (from `seg["avg_logprob"]`), but `_score_word` did
`w.get("avg_logprob", w.get("logprob", 0.0))` — always falling through to the second
key. Functionally correct but misleading.

**Fix**: Simplified to `w.get("logprob", 0.0)` — single consistent key.

### NEW-P1-2. Glossary tiebreak is dead code

**File**: `pipeline/rover.py:103-127`

Glossary weight was applied multiplicatively (`prob *= (1 + weight)`), which meant
two candidates with different glossary membership always had different scores.
The third sort key (`not is_glossary`) could never fire.

**Fix**: Changed to additive (`prob += weight`). Now two candidates with the same
base probability get a small glossary bonus that differentiates them, and the
tie-break cascade's step 3 (glossary membership) can actually engage.

---

## NEW-P2 — Robustness issues from round-1 fixes

### NEW-P2-1. `--no-correct` chain is order-sensitive

**File**: `transcribe:148`

```bash
$NO_CORRECT && CORRECT=false && CLOUD_CORRECT=false
```

Relies on `VAR=value` always returning 0. Anyone "improving" this to use `;` would
unconditionally clear `CLOUD_CORRECT`.

**Fix**: Changed to explicit `if` blocks:
```bash
if $NO_ENHANCE;  then ENHANCE=false;  fi
if $NO_DENOISE;  then DENOISE=false;  fi
if $NO_ENSEMBLE; then ENSEMBLE=false; fi
if $NO_CORRECT;  then CORRECT=false; CLOUD_CORRECT=false; fi
```

### NEW-P2-2. `_retry` retries non-transient errors

**File**: `pipeline/postcorrect.py:224-234`

Caught all `Exception`, including `json.JSONDecodeError`, `KeyError`, `TypeError`.
A malformed LLM response would retry 3x, wasting API credits.

**Fix**: Only catch transient error types:
- `urllib.error.URLError`, `socket.timeout`, `ConnectionError`, `TimeoutError`
- `openai.RateLimitError`, `openai.APIConnectionError` (imported lazily)

Non-transient errors (parse failures, type errors) propagate immediately.

### NEW-P2-3. `_retry` log line to stdout mixes with stage output

**File**: `pipeline/postcorrect.py:233`

**Fix**: Changed to `print(..., file=sys.stderr)`.

---

## NEW-P3 — Polish

### NEW-P3-1. Empty `[term]` section in seeded glossary

**File**: `pipeline/glossary.py:164`

After removing `sales -> fails`, the `[term]` section had no entries — cosmetically
odd for users reading the file as a template.

**Fix**: Removed the empty `[term]` header from `_DEFAULT_GLOSSARY`.

### NEW-P3-2. Manual `--prompt` block duplicates `docker run` invocation

**File**: `transcribe:413-428`

Manual prompt block had its own `docker run --rm --gpus all ...` because `docker_stage()`
didn't support extra `-v` mounts.

**Fix**: Extended `docker_stage` to read from a `DOCKER_STAGE_EXTRA` array.
The manual prompt block now uses `docker_stage` with `DOCKER_STAGE_EXTRA=("-v" "$PROMPT_FILE_REAL:/run/prompt.txt:ro")`,
then resets it to `()`.

---

## Files changed

| File | Changes |
|------|---------|
| `transcribe` | NEW-P0-1, NEW-P2-1, NEW-P3-2 |
| `pipeline/asr_engines.py` | NEW-P0-2 |
| `pipeline/diarize.py` | NEW-P0-3 |
| `pipeline/rover.py` | P1-6 sort, NEW-P1-1, NEW-P1-2 |
| `pipeline/postcorrect.py` | P1-10, P1-11, NEW-P2-2, NEW-P2-3 |
| `pipeline/glossary.py` | NEW-P3-1 |

## Remaining from REVIEW2 (acknowledged, deferred)

- **NEW-P0-2 (full)**: Per-window Engine B execution (split audio, run Engine B only on nl
  windows) requires the P1-3 subprocess refactor. Current fix gates at the top level.
- **NEW-P2-4**: Docker build progress polling — polish, no functional impact.
- **NEW-P3-3**: Ollama picker duplication (bash awk + Python) — kept separate for now.
- **P1-3**: ASR subprocess refactor — deferred as large change.
- **P1-16 / P1-17**: AudioSR / DeepFilterNet silently no-op — needs Rust toolchain.
