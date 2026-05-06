# Code Review #2 — after `codefixes1.md`

Reviewer: Claude (read-only re-review against `REVIEW1.md` and the spec
`HANDOFF.md`).

`codefixes1.md` claims to address every P0 and most P1/P2/P3 from
`REVIEW1.md`. This pass verifies each claim against the on-disk code and
flags anything that is **still broken**, **incompletely fixed**, or **a new
bug introduced by the fix**.

Severity legend (same as REVIEW1.md):

- **P0** — broken: code path will raise or silently produce wrong output.
- **P1** — important divergence from the approved plan.
- **P2** — robustness / correctness concerns.
- **P3** — polish.

---

## Verified fixes (good — no further action)

All confirmed by reading the post-fix code:

| ID | Fix | Verdict |
|----|-----|---------|
| P0-1 | `postcorrect.py` calls `_format_batch` / `_parse_corrections` as plain functions | ✅ correct |
| P0-2 | auto-prompt scan now passes input audio as positional arg | ✅ correct |
| P0-4 | `--prompt FILE` mounts via `-v "$PROMPT_FILE_REAL:/run/prompt.txt:ro"` | ✅ correct |
| P0-5 | `OLLAMA_WAS_ACTIVE=false` hoisted above `trap` | ✅ correct |
| P0-6 | cross-correlation reads the post-conversion `current` path | ✅ correct |
| P0-7 | `MODEL` propagates to `--model-path` via case statement | ✅ correct |
| P1-4 | Levenshtein cluster threshold raised to 0.6 | ✅ correct |
| P1-5 | tie-break cascade (`score, logprob, glossary, engine A`) | ✅ correct in spirit |
| P1-7 | diarize default model unified at `3.1` across all three layers | ✅ correct |
| P1-8 | `ENV PREBAKE_ALL=${PREBAKE_ALL}` added | ✅ correct |
| P1-9 | postcorrect uses `Glossary.load(...).to_prompt_block()` | ✅ correct |
| P1-12 | warning printed when HF token absent | ✅ correct |
| P1-13 | `OLLAMA_MODEL_EXPLICIT` flag gates pass-through to postcorrect | ✅ correct |
| P1-15 | stereo-split L/R wavs now resampled to 16 kHz mono PCM | ✅ correct |
| P2-1 | `sales -> fails` removed from default glossary | ✅ correct |
| P2-2 | `_retry()` with 3 attempts × 2× exponential backoff added | ✅ correct |
| P2-8 | ffmpeg `pan=mono\|c0=c0` / `c0=c1` replaces deprecated `-map_channel` | ✅ correct |
| P2-9 | "loudnorm skipped" note added | ✅ correct |
| P2-11 | `_auto_select` boosts ≥32 B / ≥14 B, penalises <8 B | ✅ correct |
| P3-3 | `import openai` added to Dockerfile smoke step | ✅ correct |
| P3-4 | docker-build banner mentions ~20 min on first run | ✅ correct |
| P3-7 | GLM regex now accepts `glm-X` (no minor version) | ✅ correct |
| `--no-*` | flags added and applied after preset block | ✅ correct |
| End-of-run summary | "Pipeline summary" block added | ✅ correct |

---

## Still broken / regressed (new P0)

### NEW-P0-1. `transcribe`: auto-prompt path can still poison Whisper with raw LLM output

`transcribe:359-401`. The auto-prompt flow is now:

```bash
AUTO_PROMPT_TEXT="$(curl … ollama … | sed 's/\033\[…//g')"   # ANSI strip only
…
docker_stage prompt_builder … > "$SCRATCH/prompt_build.json" 2>/dev/null || true
REFINED_PROMPT="$(jq … prompt)"
if [ -n "$REFINED_PROMPT" ]; then
    AUTO_PROMPT_TEXT="$REFINED_PROMPT"          # good path
fi
…
if $AUTO_PROMPT && [ -n "$AUTO_PROMPT_TEXT" ]; then
    echo "$AUTO_PROMPT_TEXT" > "$SCRATCH/final_prompt.txt"
    PROMPT_FILE_MOUNT=…final_prompt.txt
fi
```

When `prompt_builder` errors silently (the `2>/dev/null || true` swallows
any failure), `prompt_build.json` is missing or malformed → `REFINED_PROMPT`
ends up empty → `AUTO_PROMPT_TEXT` retains the **raw Ollama output**, which
has only had ANSI stripped by sed. `<think>…</think>` blocks, ChatML
residue, multi-line verbiage, and any other "thinking-mode" output are
written verbatim to `final_prompt.txt` and then fed to Whisper as
`initial_prompt`.

Mitigation present today: the curl call sets `system: "/no_think"` and
`think: false`, which qwen3 *usually* honours. But:

- Not every model honours the `/no_think` directive.
- If the user passes `-o some-other-model`, the directive is meaningless.
- An Ollama model bug or reasoning leak still produces poison.

The whole point of the v2 rewrite was to make sanitisation **mandatory**,
not best-effort. The current code makes sanitisation conditional on
`prompt_builder` succeeding silently.

**Fix**: don't fall back. If `prompt_builder` fails, treat that as a hard
failure and disable `--auto-prompt` for this run (set `AUTO_PROMPT=false`,
log a warning). Never write unsanitised LLM output to `final_prompt.txt`.

### NEW-P0-2. `pipeline/asr_engines.py`: per-chunk language gating still does not work

`codefixes1.md` claims P1-1 and P1-2 are fixed by `_detect_nl_windows`. The
actual code:

1. Calls `_detect_nl_windows(result_a.get("segments", []))` — but the
   segments in `result_a` carry `seg.get("language", "nl")` per
   `_run_whisperx_engine` line 121, where the value is **always** the
   default `language or "nl"` because WhisperX returns top-level `language`,
   not a per-segment one. Every segment has `lang: "nl"` by construction.
2. `_detect_nl_windows` then computes `all_same and detected_lang == "nl"`,
   which is always true → returns `[(0.0, total_end)]` covering the whole
   audio.
3. Engine B is invoked on the **entire audio** (line 232-237), not per-window.
4. The `rover_eligible` flag added to `language_chunks` (line 244) is
   **never read** — `pipeline/rover.py` does not look for it
   (`grep "rover_eligible" pipeline/*.py` returns only the line that sets it).

Net effect: behaviour is unchanged from REVIEW1.md. For an English /
French / mixed meeting the Dutch fine-tune still transcribes the whole
audio and ROVER still has to outvote Engine B on every English/French
word. **The "fix" is shape only.**

**Fix**: implement actual chunked language detection (`whisperx.detect_language`
on 30-s windows of the audio array), then split the audio into nl/non-nl
windows and call Engine B per-nl-window. Have ROVER consume the
`rover_eligible` flag and pass through Engine A on non-eligible windows.

### NEW-P0-3. `Dockerfile` pin makes the diarize monkeypatch dead code (but it's still load-bearing)

`codefixes1.md` P2-7 claim: "pinned `huggingface_hub==0.30.2` in both builder
and runtime stages." Confirmed in `Dockerfile:35,119`.

But the diarize monkeypatch (`pipeline/diarize.py:48-75`) was added because
**`huggingface_hub >= 1.0` removed `use_auth_token`** (per `HANDOFF.md`
resume notes). `huggingface_hub==0.30.2` still has `use_auth_token`, so the
monkeypatch is now patching nothing. That's not a bug today, but:

- If a future build picks a different `huggingface_hub` version (e.g. via a
  transitive bump from `pyannote.audio`), the monkeypatch may be the only
  thing keeping diarization working.
- Conversely, if the user runs `pip install -U` inside the container, the
  pin is bypassed and the patch may not work because `pyannote.audio
  3.4.0`'s import chain may have changed.

In other words: the version pin and the monkeypatch are now load-bearing
**and contradict each other**. Reading the code, you can't tell whether
the pin or the patch is actually doing the work.

**Fix**: pick one. Either pin a `huggingface_hub` version that *needs* the
patch and keep the patch, or pin a version that doesn't need the patch and
delete the patch (or comment it out with a clear "kept for future
compatibility" note).

---

## Incompletely fixed (P1 → still P1)

### P1-6 (partial). Unmatched B words appear at end of transcript out of chronological order

`codefixes1.md`: "fixed `_rebuild_segments` to append unmatched B words past
Engine A's word count."

`pipeline/rover.py:210-221` does append them — as a single trailing segment
spanning `remaining[0]["start"]` to `remaining[-1]["end"]`. But these
timestamps can be anywhere in the audio (Engine A could have missed a word
near the start). The render layer then prints this segment **after**
everything else, even when its `start` is at second 12 of a 30-minute
recording.

The spec called for "insert unmatched B-singletons into the timeline by
their start time and re-segment by speaker turn or fixed window." The
fix-as-shipped puts them at the end of the segment list, which the
renderer iterates in list order, not time order.

**Fix**: after `_rebuild_segments`, sort the resulting `segments` by `start`,
or merge unmatched B words into the temporally-nearest Engine-A segment
during cluster construction.

### P1-10. Prompt caching against Z.ai is missing

`REVIEW1.md` flagged this as P1-10. `codefixes1.md` mentions it nowhere —
not in fixes, not in the deferred list. It is silently absent.

`pipeline/postcorrect.py:200-208` rebuilds the same large system prompt
(includes the entire glossary) for every batch. With ~40 segments per
batch and 25–48 segments on the test files, that's 1–2 wasted system-prompt
sends per file. Easy ~30 % token-cost reduction.

**Fix**: send the system prompt with `cache_control: {"type": "ephemeral"}`
on the system message (Z.ai mirrors the OpenAI / Anthropic pattern for
compatible models), or use Anthropic-style pre-cached system blocks.

### P1-11 (cosmetic). Token-usage logging goes to stdout, not a sidecar JSON

`pipeline/postcorrect.py:212-213` prints the usage line directly to stdout:

```python
print(f"Z.ai ({self.model}): {response.usage.prompt_tokens} prompt + …")
```

Verification check 4 in `HANDOFF.md` §5 task 13 implied a structured record
("log token usage and the resolved GLM model id"). The current `print` is
mixed in with the rest of the run log; the orchestrator can't grep it
deterministically. Workable but fragile — recommend writing
`{model: …, usage: {…}}` to `cleaned.json` alongside the segments, or to a
sidecar `postcorrect_usage.json`.

---

## New issues introduced by the fixes

### NEW-P1-1. `_score_word` reads `avg_logprob` that `_flatten_words` never sets

`pipeline/rover.py:125`:

```python
logprob = w.get("avg_logprob", w.get("logprob", 0.0))
```

`_flatten_words` (line 46-61) populates `logprob`, not `avg_logprob`. The
`avg_logprob` lookup always falls through to `w["logprob"]`. Functionally
correct, but the dual key suggests confusion about the schema. If a future
change copies `avg_logprob` straight from segments without copying
`logprob`, the tie-break silently degrades to a 0.0 default and
ranks-by-logprob becomes a coin flip.

**Fix**: pick one key consistently. Either store `avg_logprob` in
`_flatten_words` and read `avg_logprob` here, or drop the `avg_logprob`
fallback.

### NEW-P1-2. `_score_word` glossary tiebreak is dead code

`pipeline/rover.py:103-127` returns `(score, logprob, is_glossary, engine)`
where `score = prob * (1 + glossary_weight)` already incorporates glossary
membership multiplicatively. The third sort key `not is_glossary` therefore
only fires when two candidates have **identical** scores, **identical**
logprobs, but different glossary membership — and that's impossible because
the score already differs for them by the glossary multiplier.

Not a correctness bug — the tiebreak just never engages. Worth cleaning up
or making the glossary contribution additive instead of multiplicative so
the cascade actually exercises step (3) of the spec.

### NEW-P2-1. `--no-correct` works but the bash chain is order-sensitive

`transcribe:148`:

```bash
$NO_CORRECT  && CORRECT=false && CLOUD_CORRECT=false
```

This works today because `VAR=value` always returns 0. Anyone "improving"
this to `$NO_CORRECT && CORRECT=false; CLOUD_CORRECT=false` would
unconditionally clear `CLOUD_CORRECT` — silent foot-gun. Use an `if` block
or `&& { … }`:

```bash
if $NO_CORRECT; then
    CORRECT=false
    CLOUD_CORRECT=false
fi
```

### NEW-P2-2. `_retry` retries non-transient errors

`pipeline/postcorrect.py:224-234` catches **any** Exception (including
`json.JSONDecodeError`, `KeyError`, `TypeError`). A malformed LLM response
will retry 3× before giving up — spending API credits for no chance of
success.

**Fix**: filter the caught exception types to network/transient ones
(`urllib.error.URLError`, `socket.timeout`, `openai.RateLimitError`,
`openai.APIConnectionError`, etc.). Re-raise immediately on parse / type
errors.

### NEW-P2-3. `_retry` log line uses `print` to stdout — mixes with stage output

`pipeline/postcorrect.py:233`:

```python
print(f"  retry {attempt}/{_MAX_RETRIES} after {e} (waiting {wait}s)")
```

Same issue as NEW-P1-1's stdout-log mixing. Send to stderr or to a logger.

### NEW-P2-4. `--auto-prompt` scratch dir is created before image is built

`transcribe:213-214` creates `$SCRATCH` before the docker-build check at
line 207-210 is verified. Order today is: docker-build check → scratch
mkdir → trap. Fine. But the scratch is also referenced by the auto-prompt
flow that runs before stage 1, where the docker_stage helper assumes the
image already exists. The build step at line 207-210 only fires when the
image is missing — fine — but it has no progress polling or timeout, so
any user starting on a fresh machine waits ~20 min staring at one log line.
Combined with the ANSI-stripping curl chain, troubleshooting first-run
timeouts is hard.

Lower-priority polish, listed for completeness.

### NEW-P3-1. Glossary `[term]` section header has no entries after `sales -> fails` removal

`pipeline/glossary.py:164` declares `[term]` and the heredoc ends
immediately after. An empty section is harmless (the parser skips it) but
is cosmetically odd in the seeded output and may confuse a user reading the
file as a template.

**Fix**: drop the empty `[term]` header from `_DEFAULT_GLOSSARY`, or add a
commented placeholder so the user knows what to put there.

### NEW-P3-2. Manual-`--prompt` block re-implements `docker_stage` instead of using the helper

`transcribe:413-421` does its own `docker run --rm --gpus all "${DOCKER_ARGS[@]}"
-v "$PROMPT_FILE_REAL:/run/prompt.txt:ro" "$IMAGE" pipeline.prompt_builder …`
because `docker_stage()` does not accept extra `-v` mounts. The duplication
will rot.

**Fix**: extend `docker_stage` to accept extra docker args (e.g. via a
prefix array), or factor the common `docker run` invocation into a
`run_with_extra_mounts()` helper.

### NEW-P3-3. Ollama-model picker exists in two places (bash awk + Python) with divergent rules

`select_ollama_model` (transcribe lines 249-269, awk) penalises `qwen3`,
`coder`; boosts `instruct`. `OllamaCorrector._auto_select`
(`postcorrect.py:118-153`, Python) does the same plus a parameter-count
size heuristic. The two will drift.

`codefixes1.md` P3-1 says "keeping separate" because they're "different
scoring logic (bash vs Python)" — they're not, they're nearly identical
just expressed in two languages.

**Fix**: have the bash side call the Python module:

```bash
OLLAMA_MODEL=$(python3 -c "from pipeline.postcorrect import OllamaCorrector; print(OllamaCorrector()._auto_select())")
```

(Run inside the container, so the pipeline package is on `PYTHONPATH`.)

---

## Items still on the deferred list (acknowledged in `codefixes1.md`)

These are documented as deferred and so are not flagged as bugs, but
they are worth knowing about when planning the next iteration:

- **P1-3**: ASR subprocess refactor — the `subprocess.run([sys.executable, "-c", script])`
  pattern is still in `asr_engines.py`. It's a regression toward the
  embedded-Python anti-pattern that motivated the rewrite. Revisit when the
  per-chunk language gating (NEW-P0-2) is implemented; both should land in
  the same change.
- **P1-16 / P1-17**: AudioSR / DeepFilterNet — these silently no-op. The
  end-of-run summary block now correctly notes them as "skipped" via the
  preprocess.json notes, which is a clear improvement on the silent state.
  Real fix needs Rust toolchain + dep installs.
- **P2-3, P2-4, P2-5, P2-6**: cosmetic / future improvements.
- **P2-10**: per-chunk alignment (depends on NEW-P0-2 landing first).
- **P3-1, P3-2, P3-5, P3-6, P3-8, P3-9**: noted, kept.

---

## Cross-cutting observations

1. **Two of the four claimed P1 fixes for the multi-engine path are
   ineffective.** P1-1 and P1-2 ("per-chunk language gating") look fixed
   but don't actually gate anything — the new `_detect_nl_windows`
   short-circuits to "whole audio" because the per-segment language field
   is always populated with the same default value. Combined with the
   intact subprocess-driven Engine B at line 233-237 (which still runs over
   the entire audio), the ensemble path remains the same as
   pre-fix. **The "multi-engine ROVER for Dutch chunks only" promise from
   the spec is still unmet.**

2. **The auto-prompt sanitisation guarantee is now conditional, not
   absolute.** The path through `prompt_builder` is the *good* one, but
   the bash logic falls back to raw Ollama output when `prompt_builder`
   silently fails. Whisper can still receive un-sanitised LLM output via
   that fall-through. Unconditional sanitisation was the load-bearing claim
   of the v2 rewrite — that should not be a fall-through-friendly path.

3. **The pin-vs-patch contradiction in the diarize stack should be resolved
   one way or the other.** Right now `huggingface_hub==0.30.2` makes the
   monkeypatch redundant; if the pin ever drifts, the patch silently does
   the work. Reviewing this code in 6 months will be confusing.

4. **The end-of-run summary is a real upgrade.** Combined with the warning
   for missing HF token (P1-12), users will now see clearly which stages
   ran and which were skipped. This eliminates one of the worst aspects of
   the original v2 — the "silently degraded `--quality perfect`" problem.

5. **Verification matrix from REVIEW1.md hasn't been re-run** (or at least,
   the results aren't documented in `codefixes1.md`). Recommend running:
   - each `--quality` preset on one sample
   - `--auto-prompt` on / off
   - `--cloud-correct` on / off
   - `--prompt FILE` from outside `INPUT_DIR`
   - manual run without `~/.config/whisper/hf-token`
   - `--no-correct` overriding `--quality perfect`

   Before re-confirming "tasks 1–13 complete." Several of the new findings
   (NEW-P0-1, NEW-P0-2) would surface immediately on a non-Dutch sample
   plus `--auto-prompt` with a non-qwen3 model.

---

## Suggested fix order

1. **NEW-P0-1**: harden the auto-prompt path so unsanitised LLM output
   cannot reach Whisper. This is the regression of the v2 rewrite's
   founding promise.
2. **NEW-P0-2**: actually implement per-chunk language gating, and have
   ROVER honour `rover_eligible`. Without this, the multi-engine claim is
   structural, not real.
3. **P1-6 sort**: sort `_rebuild_segments` output by `start`.
4. **NEW-P0-3**: pick one of pin or patch in the diarize stack.
5. **NEW-P1-1, NEW-P1-2**: clean up the score / tie-break implementation
   so it matches what the spec describes.
6. **NEW-P2-2, NEW-P2-3**: tighten `_retry` (catch only transient
   exceptions, log to stderr).
7. **P1-10, P1-11**: prompt caching against Z.ai + sidecar usage JSON.
8. Polish (NEW-P2-1, NEW-P3-*) and the deferred items as a follow-up.
9. README (still pending from REVIEW1.md).

End of review.
