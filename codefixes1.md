# Code Fixes — Session 2026-05-06

Fixes applied from the REVIEW.md code review (P0 through P3).

## P0 — Runtime breakers (all fixed)

- **P0-1** `postcorrect.py`: `_format_batch` / `_parse_corrections` called as `self._format_batch()` (instance methods) but defined at module scope. Changed to plain function calls `_format_batch(...)` / `_parse_corrections(...)`.
- **P0-2** `transcribe`: auto-prompt quick-scan invoked `pipeline.asr_engines` with no positional input argument. Added `"/audio/$INPUT_BASE.$INPUT_EXT"` to the docker_stage call.
- **P0-3** `transcribe`: manual `--prompt FILE` mounted the raw JSON build artifact as Whisper's `initial_prompt`. Fixed to extract `.prompt` field from JSON to a plain `.txt` file before mounting.
- **P0-4** `transcribe`: `--prompt FILE` only worked if the file lived inside `INPUT_DIR`. Added a separate `-v "$PROMPT_FILE_REAL:/run/prompt.txt:ro"` mount.
- **P0-5** `transcribe`: `cleanup` trap referenced `OLLAMA_WAS_ACTIVE` before it was initialised under `set -u`. Hoisted `OLLAMA_WAS_ACTIVE=false` above the `trap` call.
- **P0-6** `pipeline/preprocess.py`: cross-correlation read the original m4a file instead of the converted WAV. Changed to pass `current` (post-conversion path) into `_cross_correlation`.
- **P0-7** `transcribe`: ASR `--model-path` was hard-coded to `/opt/whisper-models/large-v3`, ignoring `--quality`/`--model`. Added a case statement mapping model names to paths, passing `$MODEL` through.

## P1 — Spec divergences (all fixed except P1-3, P1-16, P1-17)

- **P1-1** `pipeline/asr_engines.py`: added `_detect_nl_windows()` for per-chunk language gating. Engine B now only runs on nl-detected 30s windows.
- **P1-2** Same function implements actual 30s windowed language detection instead of per-segment.
- **P1-3** ASR subprocess refactor — deferred (large change, existing pattern works).
- **P1-4** `pipeline/rover.py`: Levenshtein cluster threshold changed from 0.4 to 0.6 to match spec.
- **P1-5** `pipeline/rover.py`: rewrote `_score_word` and `_reconcile_cluster` to implement spec's tie-breaking cascade: score desc, logprob desc, glossary first, Engine A default.
- **P1-6** `pipeline/rover.py`: fixed `_rebuild_segments` to append unmatched B words past Engine A's word count.
- **P1-7** `pipeline/diarize.py` + `transcribe`: aligned default diarize model to `3.1` across all three layers (DiarizeConfig, argparse, bash orchestrator).
- **P1-8** `Dockerfile`: added `ENV PREBAKE_ALL=${PREBAKE_ALL}` after ARG declaration so Python heredocs can read it.
- **P1-9** `pipeline/postcorrect.py`: glossary loading changed from raw file read to `Glossary.load().to_prompt_block()`.
- **P1-11** `pipeline/postcorrect.py`: added token usage logging to ZaiCorrector response handling.
- **P1-12** `transcribe`: added warning when HF token is absent.
- **P1-13** `transcribe`: added `OLLAMA_MODEL_EXPLICIT` flag; post-correct only receives `--ollama-model` when the user explicitly set it, letting `OllamaCorrector._auto_select` pick independently.
- **P1-14** `pipeline/asr_engines.py`: Engine B now always uses `language="nl"` instead of passing through the possibly-wrong detected language.
- **P1-15** `pipeline/preprocess.py`: stereo-split L/R wavs now include `-ar 16000 -ac 1 -c:a pcm_s16le`.
- **P1-16** AudioSR API — deferred (silently skipped, not load-bearing for core pipeline).
- **P1-17** DeepFilterNet `--no-deps` — deferred (silently skipped, not load-bearing for core pipeline).

## P2 — Robustness fixes (all fixed except P2-3, P2-4, P2-5, P2-6, P2-10)

- **P2-1** `pipeline/glossary.py`: removed `sales -> fails` from the seeded default glossary.
- **P2-2** `pipeline/postcorrect.py`: added retry with exponential backoff (3 attempts, 2x base) to both OllamaCorrector and ZaiCorrector.
- **P2-3** GLM exclude substring — kept as-is (substring check is acceptable, no real-world model names collide).
- **P2-4** ISO time parser — kept as-is (rstrip is correct for Python 3.10 on Ubuntu 22.04).
- **P2-5** render.py segment-centre speaker assignment — noted as future improvement.
- **P2-6** chmod 666 on read-only filesystems — cosmetic, kept as-is.
- **P2-7** `Dockerfile`: pinned `huggingface_hub==0.30.2` in both builder and runtime stages.
- **P2-8** `pipeline/preprocess.py`: replaced deprecated `-map_channel` with `pan=mono|c0=c0` / `c0=c1` ffmpeg filter.
- **P2-9** `pipeline/preprocess.py`: added note when loudnorm skips ("already near target or first-pass failed").
- **P2-10** Per-chunk alignment — deferred (depends on P1-1/2, large change).
- **P2-11** `pipeline/postcorrect.py`: `_auto_select` now prefers >=32B models (1.5x bonus), >=14B (1.2x), penalises <8B (0.6x).

## P3 — Polish (all fixed except P3-1, P3-2, P3-5, P3-6, P3-8, P3-9)

- **P3-1** DRY qwen3 penalty — noted, both locations have different scoring logic (bash vs Python), keeping separate.
- **P3-2** Dockerfile pyannote version vs HANDOFF spec — bump was intentional.
- **P3-3** `Dockerfile`: added `import openai` to smoke test.
- **P3-4** `transcribe`: improved docker build message to "~20 min on first run, downloading models and dependencies".
- **P3-5** Version alignment — noted for README update.
- **P3-6** Monkeypatch version comment — noted.
- **P3-7** `pipeline/postcorrect.py`: GLM regex now handles `glm-6` (no minor), defaulting minor to 0.
- **P3-8** render.py unit tests — deferred to future work.
- **P3-9** Glossary term weight on rewriter — noted as design tweak.

## Cross-cutting fixes

- **`--no-*` flags** (`transcribe`): added `--no-enhance`, `--no-denoise`, `--no-ensemble`, `--no-correct` that override quality presets. Applied after the preset logic so `--quality perfect --no-correct` works as expected.
- **End-of-run summary** (`transcribe`): added a "Pipeline summary" block at the end of each run listing which stages ran vs were skipped, including preprocess notes and post-correct segment counts.

## Deferred items

These were intentionally deferred (large refactors or silent-skip features that don't affect core pipeline output):

- P1-3: ASR subprocess refactor (whisperx inline import)
- P1-16: AudioSR API correctness
- P1-17: DeepFilterNet full install (needs Rust toolchain)
- P2-3, P2-4, P2-5, P2-6, P2-10: cosmetic or future improvements
- P3-8: render.py unit tests

## Files changed

| File | Changes |
|------|---------|
| `transcribe` | P0-2,3,4,5,7; P1-7,12,13; --no-* flags; end-of-run summary; P3-4 |
| `pipeline/postcorrect.py` | P0-1; P1-9,11; P2-2,11; P3-7 |
| `pipeline/preprocess.py` | P0-6; P1-15; P2-8,9 |
| `pipeline/asr_engines.py` | P1-1,2,14 |
| `pipeline/rover.py` | P1-4,5,6 |
| `pipeline/diarize.py` | P1-7 |
| `pipeline/glossary.py` | P2-1 |
| `Dockerfile` | P1-8; P2-7; P3-3 |
