# Code Review — Whisper Pipeline Overhaul (GLM-5.1 implementation)

Reviewer: Claude (read-only review against `HANDOFF.md` spec).
Scope: every file added/changed under `pipeline/`, plus the `transcribe`
bash orchestrator and the `Dockerfile`.

The HANDOFF says tasks 1–13 are complete and the three sample recordings
"ran successfully at `--quality perfect`." That smoke test proves the happy
path works for those three files; it does **not** prove the design from §2
of the handoff was implemented. This review is a static cross-check between
the approved plan and the code on disk. Every issue lists the file, the
relevant line(s), the spec clause that diverges (when one exists), and a
recommendation.

Severity legend:

- **P0** — broken: code path will raise, or produces wrong output the user
  will see.
- **P1** — important divergence from the approved plan; will silently degrade
  quality or block a downstream stage.
- **P2** — correctness / robustness concerns; safe to ship without, but
  worth fixing before declaring v2.0 stable.
- **P3** — polish / consistency.

---

## P0 — Will break at runtime

### P0-1. `postcorrect.py`: `_format_batch` / `_parse_corrections` called as instance methods but defined at module scope

`pipeline/postcorrect.py:143,165,181,193` invoke `self._format_batch(...)` and
`self._parse_corrections(...)` from inside `OllamaCorrector.correct()` and
`ZaiCorrector.correct()`. The actual functions are module-level (lines 221
and 229). Calling them through `self.` raises `AttributeError` the first
time post-correction runs — for both backends.

The HANDOFF claim that the post-correction stage works ("`--correct` returns
0 corrections — needs tuning") is consistent with this bug: 0 corrections
because the call path errors out per batch and the `except` at `run()`
line 341–343 silently drops the batch back to verbatim.

**Fix**: either move the helpers onto the classes (`def _format_batch(self,
...)`) or call them as plain functions (`_format_batch(segments)`).

### P0-2. `transcribe`: auto-prompt quick-scan invokes `pipeline.asr_engines` with no positional input argument

`transcribe:286-289`:

```bash
docker_stage asr_engines --scratch "/audio/.whisper-run-$$" \
    --language "$LANG_FLAG" --model-path "$SCAN_MODEL_PATH" \
    --batch-size 8
```

`pipeline/asr_engines.py:244` declares `input` as a required positional.
The argparse parser will exit with code 2 every time `--auto-prompt` is set,
the `|| true` swallows it, `ROUGH_TEXT` ends up empty, and the script falls
through to "Warning: Quick scan empty. Falling back to no prompt." silently.

The "fixes applied" log in HANDOFF mentions a doubled `python3 -m` problem
in `docker_stage` but does **not** mention this missing positional. Without
running with `--auto-prompt` plus a non-empty audio path, this code path was
never exercised in verification.

**Fix**: pass the input audio to the scan stage:

```bash
docker_stage asr_engines "/audio/$INPUT_BASE.$INPUT_EXT" \
    --scratch "/audio/.whisper-run-$$" --language "$LANG_FLAG" ...
```

### P0-3. `transcribe`: manual `--prompt FILE` mounts the JSON build artifact as Whisper's `initial_prompt`

`transcribe:382-398`. The manual prompt branch:

1. runs `pipeline.prompt_builder` to produce `prompt_build.json`,
2. then sets `PROMPT_FILE_MOUNT="/audio/.whisper-run-$$/prompt_build.json"`.

Downstream, `pipeline/asr_engines.py:255-256` reads this file with
`.read_text().strip()` and feeds it directly to `initial_prompt`. The result
is that Whisper sees a JSON blob (`{"prompt": "...", "warnings": [...], ...}`)
as its vocabulary hint — the exact "raw stdout into prompt" anti-pattern
this whole overhaul was meant to kill.

**Fix**: extract `result.prompt` to a plain `.txt` file (or change
`prompt_builder`'s CLI to emit text by default and JSON behind a flag) and
mount that.

### P0-4. `transcribe`: `--prompt FILE` only works if the file lives inside `INPUT_DIR`

Same block (`transcribe:391-396`): the docker run mounts `$INPUT_DIR:/audio`
but tries to reference `/audio/$(basename "$PROMPT_FILE_REAL")`. If the user
passes `--prompt ~/whisper-prompts/today.txt` and the audio is in
`/storage/fileshare/`, the prompt file isn't visible inside the container
and the command fails.

**Fix**: add a second `-v "$PROMPT_FILE_REAL:/run/prompt.txt:ro"` mount and
reference it as `/run/prompt.txt`.

### P0-5. `transcribe`: `cleanup` references `OLLAMA_WAS_ACTIVE` before it is initialised, with `set -u`

`transcribe:30` sets `set -euo pipefail`. `transcribe:205` registers the
`EXIT` trap. `OLLAMA_WAS_ACTIVE=false` isn't assigned until line 208. If
anything between line 30 and line 208 calls `exit` (e.g. the input-file
existence check at line 143-146), the trap fires and the `if
$OLLAMA_WAS_ACTIVE; then` test at line 199 hits "unbound variable" under
`set -u`, masking the original exit.

**Fix**: hoist `OLLAMA_WAS_ACTIVE=false` (and any other variables touched in
`cleanup`) above the `trap` call.

### P0-6. `pipeline/preprocess.py`: cross-correlation reads the *original* file even when the input is m4a/mp3

`preprocess.py:213` calls `_cross_correlation(input_audio, 0)` after the
upstream m4a→wav conversion at lines 201-208. `_cross_correlation` then uses
`soundfile.read()` on `input_audio` (the unconverted m4a). `soundfile`
cannot read m4a, so `_cross_correlation` raises, `corr` is set to `1.0`,
the stereo-split branch is **never taken** for m4a inputs, and the per-channel
diarization optimisation specced in §2 stage 1 silently degrades.

The `try`/`except` at lines 212-217 makes this look like "no problem" but in
practice it kills the feature for the most common phone-call format the
user has.

**Fix**: pass `current` (the post-conversion path) into `_cross_correlation`,
not the raw `input_audio`.

### P0-7. `transcribe`: ASR's `--model-path` is hard-coded to `large-v3`

`transcribe:426`:

```bash
--model-path "/opt/whisper-models/large-v3"
```

`MODEL` (resolved from `--quality`/`--model`) is never used. `--quality fast`
and `--quality medium` set `QUALITY_MODEL=small` / `medium`, and the
documented `--model` override is ignored for the actual transcription pass.
The user gets large-v3 every time regardless of what they asked for.

**Fix**: build a model-path map (`small → /opt/whisper-models/small`, …) or
pass `MODEL` straight through and let `whisperx.load_model` accept the size
name.

---

## P1 — Important spec divergences

### P1-1. `pipeline/asr_engines.py`: per-chunk language gating is missing

§2 stage 2: "Per-chunk language gating: WhisperX's built-in language ID over
30 s windows. `nl` chunks → run both engines and ROVER. `en` / `fr` chunks
→ Engine A only (Dutch fine-tune regresses on out-of-domain)."

Implementation (`asr_engines.py:207-237`): when `ensemble=True`, Engine B is
run on the **entire audio** unconditionally (line 227-231). The
`rover_eligible` flag added to `language_chunks` at line 234-235 is never
read by `pipeline/rover.py`.

This is the core of the multi-engine design and it's not implemented. For
mixed-language input (the meeting recording) Engine B runs Dutch ASR over
English/French sections too, then ROVER picks between two bad answers.

**Fix**: split the audio into 30 s windows by `language_chunks`, run Engine
B only on `nl` windows, and pass `rover_eligible` ranges into ROVER so
non-nl windows are passthrough-A.

### P1-2. `pipeline/asr_engines.py`: language ID is per-segment, not per 30 s window

`asr_engines.py:91-94` builds `language_chunks` from individual ASR segments
(typically 5–15 s) instead of fixed 30 s windows. WhisperX returns a single
top-level `language` string; per-segment `language` is the same value
repeated. The "per-chunk language ID" claim is therefore a no-op.

**Fix**: implement actual 30 s chunked language detection (WhisperX's
`load_model(...).detect_language(...)` over 30 s windows, or a separate
`whisperx.detect_language` pass).

### P1-3. `pipeline/asr_engines.py`: each engine call shells out to a fresh `python -c` subprocess

`_run_whisperx_engine` (line 47-142) builds a Python script as an f-string
and runs it via `subprocess.run([sys.executable, "-c", script])`. That:

1. Re-imports torch / whisperx / cuDNN per call (cold-start ~10–20 s).
2. F-string-interpolates user-supplied paths (`{str(audio_path)!r}`,
   `{model_path!r}`, `{hf_token!r}`). `repr()` is mostly safe but a
   filename with `\n` or unbalanced quotes will produce broken Python.
3. Loses tracebacks: only `proc.stderr[:500]` is surfaced.

This pattern is also a regression toward the old "embedded Python" anti-pattern
the bash rewrite was supposed to eliminate.

**Fix**: import whisperx normally inside the module and call it directly. The
ENTRYPOINT is already `python3 -m`, so the GPU env is set up by Docker.

### P1-4. `pipeline/rover.py`: Levenshtein cluster threshold is 0.4, spec says 0.6

`rover.py:90` accepts cluster pairing when `best_ratio >= 0.4`. §2 stage 2:
"words from A and B whose centres are within `anchor_window_s` and whose
Levenshtein distance ratio ≥ 0.6 join the same cluster."

A 0.4 threshold pairs too aggressively — "Anixter" and "Annexter" land in
the same cluster (good) but so do "Fluke" and "fluke" vs "broke" (bad).

**Fix**: change to `>= 0.6` to match spec.

### P1-5. `pipeline/rover.py`: tie-breaking ignores log-prob and "Engine A as default"

§2 stage 2: tie-break order is (a) higher mean log-prob, (b) glossary
membership, (c) Engine A default.

Implementation (`_score_word`, line 103-117): only multiplies by
`(1 + glossary_weight)`. No log-prob lookup, no Engine-A fallback. When two
candidates tie on prob and neither is in the glossary, Python's stable sort
keeps insertion order — which is whatever order `_build_anchor_clusters`
appended them. Effectively random.

**Fix**: implement the explicit cascade from the spec.

### P1-6. `pipeline/rover.py`: unmatched B words appended at the end are silently dropped

`_build_anchor_clusters` (line 96-98) adds unmatched B words as singleton
clusters appended to the list **after** all of Engine A's clusters.
`_rebuild_segments` (line 164-191) then iterates Engine A's segments, taking
one reconciled word per Engine-A word from a moving pointer. Unmatched B
words past the end of A's word list never get rendered.

This means whenever Engine B finds a word Engine A missed (the whole point
of running two engines), ROVER throws it away.

**Fix**: insert unmatched B-singletons into the timeline by their `start`
time and re-segment by speaker turn or fixed window, instead of using
Engine A's word slots as the spine.

### P1-7. `pipeline/diarize.py`: argparse default model disagrees with `DiarizeConfig` and the bash orchestrator

- `DiarizeConfig` default: `pyannote/speaker-diarization-3.1` (line 24).
- `argparse --model` default: `pyannote/speaker-diarization-community-1`
  (line 215).
- `transcribe` default: `--diarize-model 3.1` (line 51), passed as
  `pyannote/speaker-diarization-3.1`.

Anyone invoking `python -m pipeline.diarize` directly gets the community-1
model, which the HANDOFF resume notes "needs pyannote 3.4.0+" but at this
point of the doc Dutch model `community-1` itself was abandoned in favour
of `3.1`. Inconsistent across three layers.

**Fix**: pick one default (the bash script's `3.1`) and use it everywhere.

### P1-8. `Dockerfile`: `PREBAKE_ALL` ARG not exposed as ENV inside the alignment-model heredoc

`Dockerfile:73-85`:

```dockerfile
RUN python3 <<'PYEOF'
import os
if os.environ.get('PREBAKE_ALL', '1') != '1':
    print('Skipping alignment model download (PREBAKE_ALL=0)')
    raise SystemExit(0)
...
```

`PREBAKE_ALL` is declared `ARG` (line 44) but never `ENV`'d. The `os.environ`
lookup always returns the default `'1'`, so passing `--build-arg
PREBAKE_ALL=0` does **not** skip the alignment downloads (the slim variant
documented at line 3). The Dutch-model conversion at lines 62-70 *does*
honour the arg correctly because it's referenced from the shell, not Python.

**Fix**: add `ENV PREBAKE_ALL=${PREBAKE_ALL}` after the `ARG` declaration so
the heredocs can see it.

### P1-9. `pipeline/postcorrect.py`: glossary text is embedded as the raw file (comments and section headers and all)

`run()` at `postcorrect.py:289,321-324` accepts `glossary_text` as a string
and pastes it directly into the system prompt. When the glossary file
contains comments and `# Format: ...` preamble, the LLM sees those as part
of the rule list. Wastes tokens and risks confusing the model.

**Fix**: pass through `Glossary.load(...).to_prompt_block()` (already
defined in `glossary.py:106-117`) — that emits a clean `[brand]\nwrong ->
right` block.

### P1-10. `pipeline/postcorrect.py`: no prompt caching against Z.ai

§2 stage 4: "Use prompt caching where the SDK supports it (Z.ai mirrors
OpenAI's `cache_control`-style hints on system prompts when available)."

`ZaiCorrector.correct` (line 180-193) builds the same big system prompt for
every batch and re-sends it. With ~40 segments per batch and 25–48 segments
on the test files, that's 1–2 wasted system-prompt sends per file. Easy
~30 % token-cost reduction.

**Fix**: send the system prompt once with `cache_control: {"type":
"ephemeral"}` (Anthropic-style; Z.ai mirrors this for compatible models).

### P1-11. `pipeline/postcorrect.py`: token usage and resolved model id are not logged

§5 task 13 verification check 4: "log token usage and the resolved GLM model
id". The Z.ai response includes `usage` (prompt_tokens, completion_tokens),
but `correct()` discards the response object after pulling
`response.choices[0].message.content`. The orchestrator has no way to verify
the spec's "~3–8 k input tokens per 30 min of audio" target.

**Fix**: capture and write `{model: ..., usage: {...}}` to a sidecar JSON
that the orchestrator can read.

### P1-12. `transcribe`: HF token absent → diarization silently turns into one-speaker output, no warning

`transcribe:152-156` mounts the HF token only if it exists, and
`pipeline/diarize.py:179-187` returns a `method: "none"` result without
any log line that the orchestrator surfaces. The user sees one
`SPEAKER_00` block for everything and no explanation.

**Fix**: print a clear warning at the orchestrator level when
`$HF_TOKEN_MOUNT` is empty and `--speakers` was not pinned.

### P1-13. `transcribe`: post-correction reuses `OLLAMA_MODEL` chosen for prompt extraction

The auto-prompt step picks an Ollama model optimised for keyword extraction.
`transcribe:482` then passes that same model to `pipeline.postcorrect` as
`--ollama-model`. Spec says local default is `qwen2.5:32b-instruct` (a
specifically chosen instruction-tuned model for the correction task).

**Fix**: separate `--auto-prompt-model` and `--correct-model` env vars, or
let `OllamaCorrector._auto_select` (which already exists at `postcorrect.py:116-139`)
do its own pick when the orchestrator doesn't pass a value.

### P1-14. `pipeline/asr_engines.py`: Engine B always runs with `language="nl"` regardless of true language

`run_engine_b(audio, scratch, ..., language=language or "nl")`. For an
English-mostly meeting (sample 3), Engine B transcribes English audio with
the Dutch ct2 model, producing garbage that ROVER then has to outvote on
every English word. Combined with P1-1, this is the bug that "ROVER on a
meeting recording" mostly degrades quality.

**Fix**: either skip Engine B entirely when detected language ≠ nl, or
gate per-chunk per P1-1.

### P1-15. `pipeline/preprocess.py`: stereo-split L/R wavs are not resampled to 16 kHz mono

`preprocess.py:223-232` writes the L/R channels with `-map_channel 0.0.0`
only — no `-ar 16000`, no PCM format pin. The resulting L/R wavs keep the
source sample rate (often 8 kHz on phone audio, sometimes 48 kHz). §2 stage
1: "Output: canonical `preprocessed.wav` (16 kHz mono float32) plus
optional `preprocessed_L.wav` / `preprocessed_R.wav`."

**Fix**: add `-ar 16000 -ac 1 -c:a pcm_s16le` to both split commands.

### P1-16. `pipeline/preprocess.py`: AudioSR API names look invented

`preprocess.py:145`:

```python
from audiosr import AudioSR, build_model, super_resolution
sr_model = build_model(device="cuda")
super_resolution(sr_model, str(input_wav), str(output_wav))
```

The actual `audiosr` package exposes `super_resolution(audiosr, ...)` and
constructs the model differently (`build_model(model_name, device)`). The
function tuple imported here is unlikely to match any real audiosr version.
Combined with the `--no-deps` install in the Dockerfile that skips the
runtime, this branch raises on import every time and the silent
`except Exception: return False` masks it. HANDOFF resume confirms "AudioSR
bandwidth extension silently skips".

**Fix**: pin a known audiosr version, install its deps (drop `--no-deps`
for it), and use the documented API for that version.

### P1-17. `Dockerfile`: DeepFilterNet3 installed `--no-deps` will never load — silently disables `--denoise`

`Dockerfile:41`:

```dockerfile
pip3 install --no-cache-dir --no-deps audiosr deepfilternet==0.5.6
```

DeepFilterNet ships its DSP code as a Rust extension (`libdf`) that is built
during the normal install. With `--no-deps` and no Rust toolchain available
in the runtime stage, `df.enhance.init_df()` raises at first import. The
preprocess module catches and ignores it — the user gets no denoising and
no warning. HANDOFF resume confirms "libdf Rust extension not built in
image".

**Fix**: install DeepFilterNet with deps (it pulls in `loguru`, `appdirs`,
`unidecode`, `torch`, all already pinned), and add `cargo` / `rustc` to the
builder stage if the `libdf` wheel needs to compile.

---

## P2 — Robustness / quality concerns

### P2-1. `pipeline/glossary.py`: the seeded `sales -> fails` rule rewrites a real word everywhere

`_DEFAULT_GLOSSARY` line 165 includes `sales -> fails` (taken from the spec
list). This is unconditional — every "sales" in any future transcript will
be silently turned into "fails". On meeting recordings that genuinely
discuss sales, this is corruption.

**Fix**: either drop this entry, scope it to `[?]`-marked words only, or
move it to a context-sensitive section that the post-correction LLM can
weigh rather than the literal rewriter.

### P2-2. `pipeline/postcorrect.py`: no retry/backoff on transient API failure

`OllamaCorrector.correct` and `ZaiCorrector.correct` both wrap the API call
in `try`/`except` at the orchestration level (`run()` line 338-343). On
HTTP 429 / network blip / 5xx, the entire batch is silently dropped back to
verbatim. For 48-segment recordings with ~2 batches, one transient failure
costs the user half the cleaned output.

**Fix**: add a retry-with-exponential-backoff loop at the corrector level
(3 attempts, 2× backoff) before giving up.

### P2-3. `pipeline/postcorrect.py`: `_GLM_EXCLUDE = {"coder", "embedding", "vision"}` is substring-based

`postcorrect.py:73`: `if any(exc in lower for exc in _GLM_EXCLUDE)`. Future
GLM names like `glm-5.1-air-vision` would be excluded, but `glm-5.1-vision`
(spec says exclude vision-only) is fine. The substring check is OK but it
would also exclude a hypothetical `glm-6.0-decoder` (contains "coder") —
brittle. Spec also calls for excluding "vision-only" specifically.

**Fix**: use the model's metadata if Z.ai exposes capabilities, or refine to
match `-vision` / `-coder` / `-embedding` as a suffix or hyphenated token.

### P2-4. `pipeline/postcorrect.py`: cache file uses ISO time but parser only handles trailing `Z`

`postcorrect.py:90,95-99`: written as `time.strftime("%Y-%m-%dT%H:%M:%SZ",
time.gmtime())`, parsed with `datetime.fromisoformat(s.rstrip("Z"))`. Works
but `datetime.fromisoformat` accepts the trailing `Z` natively in Python
3.11+. Either pin Python ≥ 3.11 in the Dockerfile or keep the rstrip. The
runtime stage uses Ubuntu 22.04 → Python 3.10 default, so the rstrip is
needed and correct. No bug; flagging only because the construct is fragile
if anyone "modernises" it.

### P2-5. `pipeline/render.py`: speaker assignment uses segment-centre, not per-word turns

A 12-second ROVER segment can span two diarization turns; `_assign_speakers`
picks the turn with the largest overlap and labels the whole segment with
it. Word-accurate speaker labelling would be possible (ROVER carries word
timestamps) but that's not in the spec — flagging as a future improvement
only.

### P2-6. `transcribe`: the post-`run` chmod 666 will fail silently on read-only filesystems

`transcribe:505-506`. Cosmetic; only matters when the input dir isn't
writable by the host user. Listed for completeness.

### P2-7. `Dockerfile`: `huggingface_hub` is unpinned despite the diarize monkeypatch depending on its API surface

`pipeline/diarize.py:48-75` patches `hf_hub_download` because
`huggingface_hub >= 1.0` removed `use_auth_token`. The Dockerfile pins
nothing for hugginface_hub; whatever transitive version `pyannote.audio
3.4.0` pulls in today might change tomorrow.

**Fix**: pin `huggingface_hub==<known-good-version>` explicitly.

### P2-8. `pipeline/preprocess.py`: deprecated `-map_channel` ffmpeg flag

ffmpeg `-map_channel` was deprecated in 4.x in favour of the `pan` filter.
Still works in 6.x but produces a warning that can confuse log readers.

**Fix**: use `-af "pan=mono|c0=c0"` and `pan=mono|c0=c1` for L/R extraction.

### P2-9. `pipeline/preprocess.py`: `_loudnorm_two_pass` swallows all ffmpeg failures

If the second-pass `subprocess.run(..., check=True)` raises, the wrapping
`run()` propagates and the user gets a stack trace. Fine. But when the
*first* pass returns a non-JSON stderr (rare but possible on very short
files), the function returns `False` silently — loudnorm is reported as not
applied with no note. Add a warning to `result.notes`.

### P2-10. `pipeline/asr_engines.py`: WhisperX `align()` is run once for the entire result with a possibly wrong language

`asr_engines.py:97-102` aligns with `language_code=language or "nl"` over the
whole audio. For a multilingual file, alignment is forced to the first
language, which produces poor word timestamps for the other portions.
Pairs with P1-1: once per-chunk language gating exists, alignment should
also be per-chunk.

### P2-11. `pipeline/postcorrect.py`: `OllamaCorrector._auto_select` ranks by hand-tuned brand penalties

`postcorrect.py:116-139` favours `instruct` and penalises `qwen3` and
`coder`. The orchestrator passes `--ollama-model` when set, but the auto
path will pick whatever the user has pulled — including small models. No
size threshold; an 8B model will outrank a `qwen2.5:32b-instruct` if it's
named `something-instruct-q4_K_M`.

**Fix**: prefer the ≥ 14B / ≥ 32B Q4 variant when present.

---

## P3 — Polish / consistency

### P3-1. `pipeline/postcorrect.py:127-132`: the same `"qwen3"` penalty appears in two places (here and `transcribe:240`)

DRY: one Python helper would do.

### P3-2. `Dockerfile:33` pins `pyannote.audio==3.4.0` but HANDOFF §2 stage 7 says `pyannote.audio==3.3.2`

Resume notes show the bump was intentional. Update the spec table in the
HANDOFF or add a comment in the Dockerfile justifying the divergence.

### P3-3. `Dockerfile:34` pins `openai==1.54.0` — fine, but the ENTRYPOINT does not preload weights

Build-time smoke test imports `faster_whisper, whisperx, torch,
pyannote.audio` but not `openai` (line 134). The `--cloud-correct` path's
import failure would only show up at runtime.

**Fix**: add `import openai` to the smoke step.

### P3-4. `transcribe:189`: `docker build` happens silently inside the orchestrator if image is missing

This is friendly, but a 20-minute first-run "transcribe …" call with no
build progress output will look hung. Print a clear "Building image (~20
min)" line before invoking `docker build`.

### P3-5. `pipeline/__init__.py:7` declares `__version__ = "2.0.0"` — but task 14 (README update) is still pending and the README does not reference v2.0

Cosmetic; align before publishing.

### P3-6. `pipeline/diarize.py:55`: monkeypatch unwraps decorators with `while hasattr(_real, '__wrapped__')`

Comment in the code is good. Worth adding the huggingface_hub version
window this was tested against (e.g. "verified against huggingface_hub
1.0.x; reassess on bump").

### P3-7. `pipeline/postcorrect.py:74-83`: the `glm-X.Y` regex requires both major and minor

`re.match(r"glm-(\d+)\.(\d+)", mid)` rejects a future `glm-6` (no minor).
Fall back to `(major, 0)` when only major is found.

### P3-8. `pipeline/render.py`: no unit test

The `pipeline/tests/` directory only covers `prompt_builder.py` (20 tests).
`render.py` is the only thing the user actually sees and has zero tests.
A handful of golden-file cases (single speaker, two speakers, [?] markers
intact, [!unverified] tag retained) would catch regressions cheaply.

### P3-9. `pipeline/glossary.py:165` `[term]` section header followed by `sales -> fails` — `sales` is marked `term`-weighted (=1)

Combined with P2-1, the lowest weight makes ROVER least likely to prefer
this entry — but the unconditional rewriter doesn't honour weight. If the
rewriter were taught to skip `term`-weighted entries on words ROVER didn't
flag with `[?]`, the false-positive on real "sales" goes away. Worth
considering as a design tweak rather than a bug.

---

## Cross-cutting observations

1. **Verification was happy-path only.** The HANDOFF table at top says all
   three samples passed at `--quality perfect`. None of P0-1 through P0-7
   would have been caught by that run, because:

   - P0-1 (postcorrect) runs but the surrounding `try/except` swallows the
     AttributeError, leaving the cleaned output identical to verbatim
     ("0 corrections" matches the resume note exactly).
   - P0-2/3 (auto-prompt) only fire when `--auto-prompt` is set; the
     verification command in §7 does set `--auto-prompt`, but the `|| true`
     and the silent fallback to "no prompt" mean the run still completes.
   - P0-5 (cleanup) only triggers on early-error exits.
   - P0-6 (cross-corr on m4a) silently degrades to "no stereo split", which
     is an *option*, not a failure.
   - P0-7 (model path hard-coded) means the verification ran with large-v3
     regardless of what `--quality` was asked for, masking the bug.

   Recommend the verification matrix expand to: each `--quality` preset on
   one sample, `--auto-prompt` on/off, `--cloud-correct` on/off,
   `--keep-intermediates` (so the artifacts can be inspected), no HF token
   case, and `--prompt FILE` from outside `INPUT_DIR`.

2. **Several "silently skips" are silent on purpose but compound.** AudioSR,
   DeepFilterNet, post-correction (because of P0-1), ROVER unmatched words
   (P1-6), per-chunk language gating (P1-1) — every single one degrades
   quality and emits no warning. The user has no way to tell that
   `--quality perfect` is actually "perfect minus 5 stages." Add an
   end-of-run summary block printed by `transcribe` that lists which
   pipeline stages were attempted, which actually ran, and which were
   silently skipped.

3. **Feature flag ↔ quality preset coupling needs a test.** `--quality
   perfect` sets `ENHANCE=true; DENOISE=true; ENSEMBLE=true; CORRECT=true`,
   but a user who passes `--quality perfect --no-correct` (which is in the
   spec at §2 stage 8) finds there is no `--no-correct` flag in the bash
   parser. Same for `--no-denoise`, `--no-ensemble`, `--no-enhance`. The
   spec promises a "/--no-X" pair for each toggle.

   **Fix**: add explicit `--no-*` flags that override the preset.

4. **The README still hasn't been updated** (task 14 is pending in the
   handoff). New flags (`--cloud-correct`, `--ensemble`, `--context`,
   `--glossary`, `--diarize-model`, `--refresh-model`, `--keep-intermediates`)
   are user-visible and the `~/.config/whisper/` files
   (`glossary.txt`, `zai-key`, `.glm-resolved`) are not documented. This is
   the only remaining task on the handoff and should be the next step
   regardless of the fixes above.

---

## Suggested fix order

1. **P0-1, P0-2, P0-3, P0-7** — these are the bugs that make the pipeline
   silently produce worse output than v1 on common paths (post-correction
   does nothing, auto-prompt does nothing, manual prompts are corrupted,
   `--quality medium` is actually `large-v3`).
2. **P0-4, P0-5, P0-6** — runtime/environment foot-guns.
3. **P1-1, P1-2, P1-4, P1-5, P1-6, P1-14** — ROVER+ASR design integrity.
   Without these, the "multi-engine ROVER" claim is unsubstantiated.
4. **P1-8, P1-15, P1-16, P1-17** — pre-processing quality gates.
5. **P1-9, P1-10, P1-11, P1-13, P1-7, P1-12, P1-3** — post-correction and
   plumbing.
6. **P2-* / P3-*** — cleanup pass before declaring v2.0 stable.
7. README (task 14).

End of review.
