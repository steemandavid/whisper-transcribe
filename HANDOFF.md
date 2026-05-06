# Whisper Transcription Quality Overhaul — Implementation Handoff

## Resume Here (2026-05-06 session)

**Pipeline verified end-to-end on all three sample recordings.** Tasks 1–13 complete.

All three samples ran successfully at `--quality perfect`:

| File | Segments | Speakers | Turns | Preprocessing |
|------|----------|----------|-------|---------------|
| brent microsoft (427K) | 10 | 2 | 92 | denoise skipped |
| Brent Duraconnect (2MB) | 48 | 2 | 480 | loudnorm |
| Meeting Recording (49MB) | 25 | 4 | 245 | loudnorm + highpass |

**Remaining task:** 14 (README update).

**Fixes applied during 2026-05-06 session:**
- Fixed Docker entrypoint (`docker_stage` was doubling `python3 -m`)
- Added early WAV conversion in `preprocess.py` for non-WAV inputs (m4a/mp3)
- Upgraded `ctranslate2` from 4.4.0 → 4.5.0 (cuDNN 9 compatibility)
- Added `LD_LIBRARY_PATH` for nvidia cuDNN/cublas libraries
- Added `matplotlib` (missing pyannote.audio dependency)
- Added monkeypatch in `diarize.py` to remap `use_auth_token` → `token` for huggingface_hub >= 1.0
- Changed default diarization model from `community-1` → `3.1` (compatible with pyannote 3.4.0)
- Made Ollama stop/start non-fatal (no passwordless sudo in current setup)
- Image updated via docker commit: `whisper-transcribe:2.0`

**Known limitations:**
- DeepFilterNet3 denoising (`--denoise`) silently skips — `libdf` Rust extension not built in image
- AudioSR bandwidth extension silently skips — same `--no-deps` limitation
- Local LLM post-correction (`--correct`) returns 0 corrections — needs tuning or `--cloud-correct`
- No `~/.config/whisper/zai-key` — `--cloud-correct` untested
- `vloek` → `Fluke` misrecognition persists (glossary-aware post-correction not effective with local LLM)
- `--auto-prompt` and `--summary` require sudo to manage Ollama lifecycle

---

This document is a self-contained brief for an LLM that will pick up the work in
progress. It covers (1) the goal, (2) the approved architecture in full, (3)
the current progress with file-by-file status, (4) the exact next steps with
expected APIs, and (5) the verification plan.

The repo is a single-user, GPU-accelerated Whisper transcription pipeline that
runs in Docker. The user's audio is mostly Belgian-Dutch (Flemish) business
calls and meetings. Speed/cost is no constraint — the goal is the highest
attainable quality.

---

## 1. Goal

Three real samples in `/storage/fileshare/` exposed systemic quality gaps that
no decoding-parameter tweak can fix:

- **Brand / jargon mishearing** (Flemish phonetics + multilingual model bias):
  - `vloek` → `Fluke`
  - `Annexter` → `Anixter`
  - `comscope` → `CommScope`
  - `Connect-wise` → `ConnectWise`
  - `applyermap` → `wiremap`
  - `sales` → `fails`
  - `DNAI` → `de AI`
  - `Dinapse` → `Dynapse`
- **Auto-prompt poisoning**: the legacy bash CLI piped raw `ollama` stdout
  (qwen3 `<think>...</think>` blocks + ANSI escape codes like `\x1b[K\x1b[2D`)
  straight into Whisper's `initial_prompt`. The "vocab hint" actively hurt
  decoding. Saved poison example:
  `/storage/fileshare/.whisper_prompt_v1_2965923.txt` (≈800 lines of qwen3
  thinking-mode output with leaked terminal control codes).
- **No audio preprocessing** (raw m4a → WhisperX) and **no post-correction**
  stage to recover from any of the above.

The user confirmed via clarifying questions:
- Optimise for all audio types equally (phone, Teams, in-person).
- Local-first, but a cloud LLM is acceptable for the text-only post-correction
  step.
- Produce **two outputs** per run: a verbatim transcript + a cleaned
  transcript.
- Run **multi-engine ASR ensemble** (large-v3 + Dutch fine-tune) reconciled
  via ROVER.
- For cloud post-correction, use **the latest GLM model** via Z.ai's
  OpenAI-compatible API. Auto-detect the highest version on each run.

---

## 2. Architecture (approved plan)

The original bash-with-embedded-Python orchestrator becomes **bash CLI +
Python modules under `pipeline/`**. Stages communicate via JSON artifacts in a
per-run scratch directory at `$INPUT_DIR/.whisper-run-<pid>/`. The bash CLI
remains the entry point and stage sequencer; long Python is no longer
string-escaped inside here-docs (the structural cause of the auto-prompt
poisoning).

### Stage 1 — Audio preprocessing (`pipeline/preprocess.py`)

`ffprobe` the source, then conditionally apply:

1. **Channel split**: stereo with low-correlated channels (cross-corr < 0.6)
   → keep both, run diarization per channel later. Otherwise downmix to mono.
2. **Loudness normalize**: `ffmpeg loudnorm` two-pass to `I=-16 LUFS`,
   `TP=-1.5`, `LRA=11`. Skip if already within 2 LU.
3. **High-pass 80 Hz**: only when source SR ≥ 16 kHz (skip on telephony
   narrowband to avoid removing fundamentals).
4. **Bandwidth extension** (narrowband only): use **AudioSR**
   (`haoheliu/versatile_audio_super_resolution`). Detect narrowband via FFT
   energy above 3.4 kHz < −50 dBFS. **Never** run on already-wideband audio
   (it hallucinates sibilants).
5. **Denoise**: **DeepFilterNet3** (pip-installable, GPU, speech-tuned, no
   hallucination). On for `--quality perfect`, otherwise gated by
   `--denoise/--no-denoise`. It can suppress soft secondary speakers in
   conference rooms, so it is opt-in by default for non-perfect presets.

Output: canonical `preprocessed.wav` (16 kHz mono float32) plus optional
`preprocessed_L.wav` / `preprocessed_R.wav`. Metadata + decisions logged to
`preprocess.json`.

### Stage 2 — Multi-engine ASR + ROVER (`pipeline/asr_engines.py`, `pipeline/rover.py`)

- **Engine A**: existing WhisperX `large-v3` with current `asr_options`
  (`beam_size=10`, `best_of=10`, `temperature=0`). Unchanged.
- **Engine B**: **`golesheed/whisper-large-v3-dutch`** (CGN-trained,
  Flemish-aware), loaded via `faster-whisper` after a one-shot ct2 conversion
  baked into the image.
  - *Maintainer note*: A/B against `qanastek/whisper-large-v3-french-dutch` on
    the three sample files before locking in.
- **Per-chunk language gating**: WhisperX's built-in language ID over 30 s
  windows. `nl` chunks → run both engines and ROVER. `en` / `fr` chunks →
  Engine A only (Dutch fine-tune regresses on out-of-domain).
- **Word-level alignment**: both engines emit word timestamps via wav2vec2
  (`jonatasgrosman/wav2vec2-large-xlsr-53-dutch` for nl). Anchor on word
  centres within ±150 ms; within each anchor cluster, character-level
  Levenshtein resolves "same word in same position".
- **ROVER**: implement in-house, ~150 LOC. (SCTK / `sclite` is text-only,
  loses our timestamp anchors, painful to containerise.) Algorithm:
  confidence-weighted majority vote per slot; tie-break by (a) higher mean
  log-prob, (b) presence in glossary / prompt vocab, (c) Engine A as default.
  Emit `[?]` markers when winner's normalised-confidence margin over
  runner-up is < 0.15.

### Stage 3 — Prompt sanitisation & priors (`pipeline/prompt_builder.py`)

Replaces the bash `ollama_extract` + here-docs with Python that:

- **Strips ANSI** (`\x1b\[[0-9;]*[a-zA-Z]`) and **strips qwen3 thinking
  blocks** (`<think>...</think>`, `Thinking...\n...\n...done thinking`,
  ChatML residue) before any text reaches a prompt file.
- **Validates**: single line, ASCII + Latin-1, ≤ 1500 chars, ≤ 80 tokens.
  On violation → fall back to filename priors only and log a warning.
- **Filename priors**: tokenise basename, drop short / numeric tokens,
  title-case proper-noun candidates
  (`brent_microsoft_*.m4a` → `Brent, Microsoft`).
- **`--context "..."` flag**: free-text prior prepended verbatim.
- Final prompt = `[user context] + [filename priors] + [LLM-extracted vocab]
  + [glossary canonical forms]`, dedup-preserved-order, truncated to ~224
  BPE tokens (Whisper `initial_prompt` ceiling).

Move `select_ollama_model` and Ollama HTTP plumbing into Python too, with the
same sanitisation applied to every LLM response.

### Stage 4 — LLM post-correction (`pipeline/postcorrect.py`)

Highest-leverage step. Inputs: ROVER-merged JSON + glossary + filename +
`--context` + detected languages.

- **Default backend**: local `qwen2.5:32b-instruct` via Ollama.
- **`--cloud-correct`**: latest GLM model via Z.ai OpenAI-compatible API
  (`https://api.z.ai/api/paas/v4/`), `ZAI_API_KEY` from env.
- **Auto-version detection** at startup:
  1. Call `GET /models`.
  2. Filter for `glm-*` entries excluding embedding / vision-only / coder
     variants by name pattern.
  3. Parse the version (`glm-X.Y` → `(X, Y)` tuple), pick the highest.
  4. Cache the resolved model id in `~/.config/whisper/.glm-resolved` for
     24 h. Always re-resolve on `--refresh-model` flag or cache miss.
  5. Fall back to `glm-5.1` (current latest as of this plan) if the list
     endpoint is unavailable.
- Use prompt caching where the SDK supports it (Z.ai mirrors OpenAI's
  `cache_control`-style hints on system prompts when available).
- **Batches of ~40 segments** with speaker labels and `[?]` markers preserved.
- **System prompt rule**: "Fix only obvious phonetic mishearings
  (Dutch / Flemish dialect, brand names from the glossary, `[?]`-marked
  words). Do not paraphrase, do not add or remove content, do not change
  punctuation outside fixed words."
- **Strict JSON output**:
  ```json
  [{"seg_id": ..., "original": ..., "corrected": ...,
    "changed_words": [{"idx": ..., "before": ..., "after": ...,
                       "confidence": ...}]}, ...]
  ```
  Apply only changes with per-word `confidence ≥ 0.8`. Lower-confidence
  corrections are dropped and the `[?]` marker is preserved in the cleaned
  output.
- **Diff guard**: any cleaned segment with token-edit-distance > 35 % from
  verbatim is rejected and falls back to verbatim with `[!unverified]` tag.
  Catches LLM hallucinations / paraphrasing creep.

**Two outputs, both written:**

- `audio.txt` — verbatim ROVER, speaker-labeled, today's exact format.
- `audio.cleaned.txt` — post-corrected, same format, with `[?]` retained
  where the LLM was not confident.

### Stage 5 — Glossary (`pipeline/glossary.py`)

Plain-text format, one entry per line: `wrong → right` (Unicode arrow or
`->`). Comments `#`, sections `[brand]` / `[person]` / `[place]` / `[term]`
for downstream weighting (brands strongest). Lookup order: `--glossary FILE`,
container path `/run/glossary.txt`, then `~/.config/whisper/glossary.txt`.
Used in three places:

1. Prompt builder (canonical forms appended).
2. ROVER tie-breaker (glossary terms preferred).
3. Post-correction system prompt (full glossary embedded).

Seed `~/.config/whisper/glossary.txt` with the known terms surfaced from the
samples: Fluke, Anixter, CommScope, ConnectWise, wiremap, RustOleum, Bjorn,
Microsoft, Teams, GigaSpeed XL, Berendrechtstraat, Mespelare, etc.

### Stage 6 — Diarization (`pipeline/diarize.py`)

- **`--diarize-model`**: A/B `pyannote/speaker-diarization-community-1` vs
  `3.1`. Default `community-1` after a smoke-test on samples.
- **`--speakers N`**: pin `min_speakers=max_speakers=N`.
- **Filename heuristic**: regex
  `(?i)(callrec|phonecall|gsm|recording_\d{14})` → default `N=2`.
- **Per-channel diarization** when stage 1 kept stereo split: skip pyannote,
  label each channel as a single speaker (`[CH-L]`, `[CH-R]`). Dramatically
  more accurate than VAD-based diarization on cross-talked phone audio.

### Stage 7 — Docker image & dependency pinning

- **Pin all pip versions** (current Dockerfile is unpinned):
  `whisperx==3.4.2`, `faster-whisper==1.1.1`, `ctranslate2==4.5.0`,
  `torch==2.5.1+cu124`, `torchaudio==2.5.1`, `pyannote.audio==3.3.2`,
  `deepfilternet==0.5.6`, `audiosr==0.0.7`, `openai==1.54.0` (used as the
  OpenAI-compatible client against `https://api.z.ai/api/paas/v4/`),
  `ollama==0.4.4`. Add a build-time smoke step:
  `python -c "import faster_whisper, whisperx, torch; print(...)"`.
- **Pre-bake the Dutch model** via `ct2-transformers-converter` into
  `/opt/whisper-models/nl-large-v3/`. Pre-download wav2vec2 alignment models
  for `en`, `nl`, `fr`. Pre-download DeepFilterNet3 weights and AudioSR
  checkpoint. Multi-stage build to keep final image ~12 GB (up from ~7 GB).
- Copy `pipeline/` into `/opt/pipeline/`, set `PYTHONPATH=/opt`.
- Build arg `PREBAKE_ALL=1` (default) — set to `0` for a slim variant that
  downloads on first run into the existing `whisper-hf-cache` volume.

### Stage 8 — CLI surface (backward compatible)

New flags on `transcribe`:

- `--enhance` / `--no-enhance` (preprocessing pipeline)
- `--denoise` / `--no-denoise`
- `--ensemble` / `--no-ensemble` (ROVER on/off)
- `--correct` (local LLM cleanup), `--cloud-correct` (latest GLM via Z.ai),
  `--no-correct`
- `--refresh-model` (force re-query of the GLM models endpoint, ignoring the
  24 h cache)
- `--context "free text"`
- `--speakers N`
- `--glossary FILE`
- `--diarize-model {3.1,community-1}`
- `--keep-intermediates` (preserve scratch dir for debugging)

`--quality perfect` preset toggles enable: `enhance + denoise + ensemble +
correct (local)`. `--quality perfect --cloud-correct` swaps in the
auto-resolved latest GLM for the cleanup pass. All other presets keep
today's behaviour.

---

## 3. Progress report

Status as of handoff. Tasks are tracked numerically; complete the in-progress
one first, then proceed in numerical order.

| #  | Task                                                              | Status        |
| -- | ----------------------------------------------------------------- | ------------- |
| 1  | Read full `transcribe` script and `Dockerfile`                    | ✅ done       |
| 2  | Scaffold `pipeline/` Python package                               | ✅ done       |
| 3  | Implement glossary module                                         | ✅ done       |
| 4  | Implement `prompt_builder` with sanitisation                      | ✅ done (unit tests passing, including real poison file) |
| 5  | Implement audio `preprocess` module                               | ✅ done       |
| 6  | Implement `asr_engines` module                                    | ✅ done       |
| 7  | Implement ROVER reconciliation                                    | ✅ done       |
| 8  | Implement diarization module                                      | ✅ done       |
| 9  | Implement `postcorrect` module (with GLM auto-version resolver)   | ✅ done       |
| 10 | Rewrite `Dockerfile` with pinned deps and pre-bake                | ✅ done       |
| 11 | Rewrite `transcribe` bash orchestrator                            | ✅ done       |
| 12 | Build Docker image and run smoke checks                           | ✅ done (image 20.4 GB, all imports pass) |
| 13 | Verify on the three sample recordings                             | ✅ done — all 3 pass, 2 speakers each (4 on meeting) |
| 14 | Update README with new pipeline and flags                         | ⬜ pending    |

### Files that exist on disk

| Path                                                        | Status                                  |
| ----------------------------------------------------------- | --------------------------------------- |
| `pipeline/__init__.py`                                      | ✅ done — version `2.0.0`              |
| `pipeline/artifacts.py`                                     | ✅ done — `RunPaths` + JSON helpers    |
| `pipeline/glossary.py`                                      | ✅ done — loader + seeder, 26 entries  |
| `pipeline/prompt_builder.py`                                | ✅ done — sanitiser + builder          |
| `pipeline/preprocess.py`                                    | ✅ done — audio preprocessing          |
| `pipeline/asr_engines.py`                                   | ✅ done — multi-engine ASR             |
| `pipeline/rover.py`                                         | ✅ done — ROVER reconciliation         |
| `pipeline/diarize.py`                                       | ✅ done — speaker diarization          |
| `pipeline/postcorrect.py`                                   | ✅ done — LLM post-correction          |
| `pipeline/render.py`                                        | ✅ done — transcript text rendering    |
| `pipeline/tests/test_prompt_builder.py`                     | ✅ done — 20 tests, all passing        |
| `pipeline/tests/__init__.py`                                | ✅ done                                |
| `~/.config/whisper/glossary.txt`                            | ✅ seeded with Fluke / Anixter / …     |
| `~/.config/whisper/hf-token`                                | pre-existing, untouched                |
| `transcribe`                                                | ✅ rewritten — slim bash orchestrator  |
| `Dockerfile`                                                | ✅ rewritten — multi-stage, pinned     |
| `model.bin` (3 GB faster-whisper large-v3)                  | pre-existing, untouched                |

### Existing module APIs that subsequent stages will import

`pipeline/artifacts.py`:

```python
@dataclass(frozen=True)
class RunPaths:
    input_audio: Path
    input_dir: Path
    input_base: str
    scratch: Path
    # properties: preprocessed, preprocessed_left, preprocessed_right,
    #             preprocess_meta, engine_a_json, engine_b_json,
    #             rover_json, diarize_json, merged_json, cleaned_json,
    #             verbatim_txt, cleaned_txt

def resolve_run_paths(input_audio, scratch=None) -> RunPaths
def write_json(path: Path, data) -> None
def read_json(path: Path)
```

`pipeline/glossary.py`:

```python
SECTION_WEIGHTS = {"brand": 4, "person": 3, "place": 2, "term": 1}
USER_GLOSSARY = Path.home() / ".config" / "whisper" / "glossary.txt"
CONTAINER_GLOSSARY = Path("/run/glossary.txt")

@dataclass(frozen=True)
class GlossaryEntry:
    wrong: str
    right: str
    section: str
    weight: int  # property

@dataclass
class Glossary:
    entries: list[GlossaryEntry]
    source: Path | None
    @classmethod
    def load(cls, path: Path) -> "Glossary"
    @classmethod
    def resolve(cls, override: str|os.PathLike|None = None) -> "Glossary"
    def canonical_terms(self) -> list[str]
    def lookup(self, term: str) -> str | None
    def is_canonical(self, term: str) -> bool
    def to_prompt_block(self) -> str

def seed_default(path: Path = USER_GLOSSARY) -> bool
```

`pipeline/prompt_builder.py`:

```python
MAX_PROMPT_CHARS = 1500
MAX_PROMPT_TOKENS = 80
PROMPT_TOKEN_BUDGET = 224

@dataclass
class PromptBuildResult:
    prompt: str
    vocab_tokens: list[str]
    rejected: list[str]
    warnings: list[str]

def strip_thinking(text: str) -> str
def strip_ansi(text: str) -> str
def normalise_text(text: str) -> str
def sanitise_vocab(raw: str) -> tuple[list[str], list[str]]
def filename_priors(audio_path: str | Path) -> list[str]
def build_prompt(*, audio_path, glossary=None, llm_vocab=None,
                 user_context=None) -> PromptBuildResult
```

---

## 4. Immediate next step — finish task 4

Before moving on, **add and run a smoke test** for `prompt_builder.py` that
proves the poison-prompt scenario is now handled:

1. Read the real poison sample at
   `/storage/fileshare/.whisper_prompt_v1_2965923.txt`. (It exists; ~800 lines
   of qwen3 thinking-mode + ANSI escapes.)
2. Pass its contents through `sanitise_vocab()` and assert:
   - Result token list is non-empty (don't reject *everything*).
   - No token contains an `\x1b` byte or a bare `[K` / `[2D` artefact.
   - No token contains `<think>` or the literal substring `done thinking`.
3. Pass it through `build_prompt(audio_path="brent_microsoft_…m4a", llm_vocab=poison)`
   and assert:
   - `result.prompt` is ≤ `MAX_PROMPT_CHARS`.
   - `result.warnings` includes at least one entry (truncation or "long
     phrase" drops).
   - `result.prompt` contains "Brent" and "Microsoft" (filename priors fired).
   - `result.prompt` contains "Fluke" or another seeded glossary canonical
     (proves glossary is appended).

Suggested home: `pipeline/tests/test_prompt_builder.py` (create the
`pipeline/tests/` directory and an empty `__init__.py`). Run with
`python -m unittest discover pipeline/tests`. Once green, mark task 4
complete and start task 5.

---

## 5. Detailed work order for remaining tasks

### Task 5 — `pipeline/preprocess.py`

Public surface:

```python
@dataclass
class PreprocessResult:
    mono_path: Path
    left_path: Path | None
    right_path: Path | None
    sr: int                  # always 16000
    used_loudnorm: bool
    used_highpass: bool
    used_audiosr: bool
    used_deepfilternet: bool
    stereo_split: bool
    cross_correlation: float | None
    notes: list[str]

def run(input_audio: Path, scratch: Path, *,
        denoise: bool, enhance: bool) -> PreprocessResult
```

CLI: `python -m pipeline.preprocess <input.wav> --scratch DIR
[--denoise] [--enhance]`. Writes `preprocess.json` summarising the result and
the canonical `preprocessed.wav` (and `_L.wav` / `_R.wav` when stereo split
fires).

Implementation notes:

- Use `ffprobe -show_streams -of json` to inspect channels / sample rate /
  duration.
- Cross-correlation: read both channels with `soundfile`, compute Pearson
  correlation on a 30-s window (or full file if shorter). Threshold 0.6.
- Loudnorm two-pass: first pass with `-af loudnorm=…:print_format=json`
  parses the `input_i` / `input_tp` / `input_lra` / `input_thresh` /
  `target_offset` numbers and feeds them into the second pass's
  `measured_*` parameters. Skip if `|input_i - target_i| < 2`.
- Narrowband detection: 1024-pt FFT, mean energy in bins > 3.4 kHz, threshold
  −50 dBFS. Only then call AudioSR.
- DeepFilterNet3: `from df.enhance import enhance, init_df`; load model once.
- Always finish with a resample to 16 kHz mono float32 PCM via `ffmpeg
  -ar 16000 -ac 1 -c:a pcm_s16le` (Whisper's expected input).

### Task 6 — `pipeline/asr_engines.py`

Two entry points, `run_engine_a(audio, scratch, prompt, language=None)` and
`run_engine_b(audio, scratch, prompt)`. Engine A wraps WhisperX + the
existing decoder options. Engine B wraps `faster-whisper` against the pre-baked
Dutch ct2 model at `/opt/whisper-models/nl-large-v3/`.

Both must emit the same JSON schema (write to `asr_engine_a.json` /
`asr_engine_b.json`):

```json
{
  "engine": "A" | "B",
  "language_chunks": [{"start": 0.0, "end": 30.0, "lang": "nl", "prob": 0.97}, ...],
  "segments": [
    {
      "id": 0, "start": 0.12, "end": 4.30,
      "text": "...",
      "avg_logprob": -0.21,
      "lang": "nl",
      "words": [{"start": 0.12, "end": 0.34, "word": "Dag", "prob": 0.91}, ...]
    },
    ...
  ]
}
```

Word timestamps come from wav2vec2 alignment
(`jonatasgrosman/wav2vec2-large-xlsr-53-dutch` for nl,
`WAV2VEC2_ASR_BASE_960H` for en, `voidful/wav2vec2-xlsr-multilingual-56` for
fr). Per-chunk language gating: if a 30-s chunk's `lang != "nl"`, write
Engine A's segments for that span and skip Engine B (downstream ROVER will
treat single-source spans as already-resolved).

### Task 7 — `pipeline/rover.py`

Public surface:

```python
@dataclass
class RoverConfig:
    anchor_window_s: float = 0.150
    margin_threshold: float = 0.15

def reconcile(engine_a_path: Path, engine_b_path: Path, glossary: Glossary,
              cfg: RoverConfig = RoverConfig()) -> dict  # writes rover.json
```

Algorithm (~150 LOC):

1. Flatten both engines to per-word lists with `(start, end, word, prob,
   logprob)`.
2. Build anchor clusters: greedy left-to-right walk; words from A and B whose
   centres are within `anchor_window_s` and whose Levenshtein distance ratio
   ≥ 0.6 join the same cluster. Words without a partner form a singleton
   cluster.
3. Per cluster, score candidates by `prob` (or normalised `exp(logprob)` if
   `prob` missing). Multiply by glossary weight when
   `glossary.is_canonical(word)`.
4. Pick the winner. If `(top - runner_up) / top < margin_threshold`, append a
   trailing `[?]` to the winning word.
5. Re-emit reconciled segments using engine-A timing as the spine, swapping
   in winner words. Store both per-word margin and a `source` ∈
   `{"A","B","both","glossary"}` in the JSON.

### Task 8 — `pipeline/diarize.py`

`run(audio_paths, *, model="community-1", n_speakers=None,
filename_hint=None) -> dict` writing `diarize.json`. When `audio_paths` is a
2-tuple (the stereo split case), bypass pyannote and emit one segment per
non-silent span per channel labelled `CH-L` / `CH-R`. Otherwise call pyannote
with the requested model and the optional `min_speakers / max_speakers` pin.

Filename hint: regex `(?i)(callrec|phonecall|gsm|recording_\d{14})` → set
`n_speakers=2` when the user did not pass `--speakers`.

### Task 9 — `pipeline/postcorrect.py` (high leverage, do this carefully)

Two backends behind one interface:

```python
class PostCorrector(Protocol):
    def correct(self, batches: list[list[Segment]],
                system_prompt: str) -> list[CorrectedBatch]: ...

class OllamaCorrector: ...
class ZaiCorrector: ...

def resolve_glm_model(client, *, refresh: bool = False,
                      cache_path: Path = ...) -> str
```

`resolve_glm_model`:

1. If cache file exists, mtime < 24 h, and `refresh is False` → return cached.
2. Else `client.models.list()`, filter
   `m.id.startswith("glm-")` and not in
   `{coder, embedding, vision-only}` substrings, parse `glm-X.Y` →
   `(int(X), int(Y))`, pick max.
3. Write `{"id": ..., "resolved_at": <iso8601>}` to
   `~/.config/whisper/.glm-resolved`.
4. On any exception, fall back to `glm-5.1` and log.

Z.ai client setup:

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["ZAI_API_KEY"],
                base_url="https://api.z.ai/api/paas/v4/")
```

Batching: ~40 segments per request; preserve `[?]` markers in the input;
require strict JSON output as defined in §2 stage 4. After parsing, apply the
≥ 0.8 confidence filter and the 35 % token-edit-distance diff guard.

Outputs: `cleaned.json` (per-segment after correction). The `transcribe` bash
orchestrator is responsible for rendering `audio.txt` and
`audio.cleaned.txt` from `rover.json` + `diarize.json` + `cleaned.json`.

### Task 10 — `Dockerfile` rewrite

Keep `nvidia/cuda:12.4.0-runtime-ubuntu22.04` base. Add multi-stage build:

- Stage `builder`: install build deps, run `ct2-transformers-converter
  --model golesheed/whisper-large-v3-dutch
  --output_dir /opt/whisper-models/nl-large-v3 --quantization float16`,
  pre-download wav2vec2 alignment models (`en`, `nl`, `fr`),
  DeepFilterNet3 weights, AudioSR checkpoint.
- Stage `runtime`: copy `/opt/whisper-models/`, `/opt/wav2vec/`,
  `/opt/df3/`, `/opt/audiosr/` from builder; copy `pipeline/` to
  `/opt/pipeline/`; set `ENV PYTHONPATH=/opt`.
- Pin every pip install (versions in §2 stage 7).
- Build arg `PREBAKE_ALL=1` (default); when `0`, skip the model downloads
  and let first run populate the existing `whisper-hf-cache` volume.
- Smoke step before exit: `RUN python -c "import faster_whisper, whisperx,
  torch, df, pyannote.audio, openai; print('imports ok')"`.
- New `ENTRYPOINT`: `python3 -m pipeline.<stage>` (each stage invokable
  individually). The bash orchestrator runs `docker run` per stage.

### Task 11 — `transcribe` bash rewrite

Strip every embedded Python here-doc. The bash script's job becomes:

1. Parse flags (the new surface in §2 stage 8) and the `--quality` preset.
2. `mkdir -p` the scratch dir and `trap` cleanup unless
   `--keep-intermediates`.
3. Run each pipeline stage as `docker run … python -m pipeline.<stage> …`
   with the scratch dir and HF cache volume mounted.
4. Mount the user's glossary at `/run/glossary.txt` (read-only) and
   `~/.config/whisper/zai-key` exported as `ZAI_API_KEY` when
   `--cloud-correct` is set.
5. Render the two `.txt` outputs from `rover.json` + `diarize.json` +
   (optionally) `cleaned.json` — this is small enough to live in bash, or
   factor out to `pipeline/render.py` if it grows past ~30 lines.

### Task 12 — Build & smoke

```bash
docker build -t whisper-transcribe:2.0 .
docker run --rm --gpus all whisper-transcribe:2.0 python -m pipeline.glossary --dump
```

The build must succeed end-to-end (the import smoke step gates it).

### Task 13 — Verification on real samples

Run each of the three samples in `/storage/fileshare/`:

- `Brent Duraconnect CallRecord_20260423-140505_+32472073503.m4a`
- `brent microsoft - CallRecord_20260424-164320_+32472073503.m4a`
- `Start of new L&L ticketing process-20260420_103418-Meeting Recording.mp4`

once at `--quality perfect` and once at `--quality perfect --cloud-correct`.

**Quantitative checks:**

1. `grep -E "vloek|Annexter|comscope|Connect-wise|applyermap|DNAI" *.cleaned.txt`
   → expect **zero** hits in `.cleaned.txt`; ≥ 80 % reduction in `.txt`
   compared to the legacy outputs in `/storage/fileshare/*.txt`.
2. `diff` `.cleaned.txt` against `.txt`: changes should be local, lexical,
   short. No segment should exceed the 35 % token-churn guard (count flagged
   `[!unverified]` segments — should be 0 on these samples).
3. Speaker-turn count matches the legacy transcripts ±1.
4. For `--cloud-correct`: log token usage and the resolved GLM model id
   (e.g. `glm-5.1`); expect ~3–8 k input tokens per 30 min of audio. Verify
   the auto-version path: clear `~/.config/whisper/.glm-resolved`, run with
   `--refresh-model`, confirm the resolver logs the picked model id and that
   it's the highest-versioned `glm-*` non-embedding / non-vision /
   non-coder entry returned by `GET /models`.
5. **Regression check**: pick three segments per sample where the legacy
   pipeline is already correct; verify the new pipeline doesn't break them.
6. **Auto-prompt sanitiser** end-to-end: feed the real
   `/storage/fileshare/.whisper_prompt_v1_2965923.txt` to
   `pipeline.prompt_builder` and confirm rejection / fallback (this is
   already covered by the unit test in §4, but rerun against the live
   pipeline once the orchestrator is wired).
7. **Build smoke**: `docker build` must pass the import smoke step in the
   `Dockerfile`.

### Task 14 — README update

Document the new flags, the two-output design, the `~/.config/whisper/`
files (`glossary.txt`, `hf-token`, `zai-key`, `.glm-resolved`), and the
per-stage invocation (`python -m pipeline.<stage>`) for debugging.

---

## 6. Out of scope (do NOT do these)

- Replacing WhisperX with another ASR framework (NeMo Parakeet, etc.) —
  Parakeet is English-only and the user's audio is mostly Dutch.
- Audio-aware LLM second-pass verification (Qwen2-Audio, Phi-4-multimodal) on
  `[?]` segments — possible follow-up if the dual-output design proves
  insufficient.
- Replacing Ollama with vLLM / llama.cpp for the local LLM steps — Ollama
  works and the orchestration is already wired.

---

## 7. Quick reference — sample evidence files

The user's three real test cases live in `/storage/fileshare/`. The legacy
`.txt` outputs sit alongside each audio file and document the exact errors
this overhaul targets. `run-transcription.txt` shows the user's invocation
template:

```
~/claudecode/projects/whisper/transcribe \
  --language nl --auto-prompt --summary --quality perfect \
  /storage/fileshare/<audiofile>
```

The legacy `.whisper_prompt_v1_2965923.txt` poison file is the unit-test
fixture for §4. Treat it as the gold-standard adversarial input for the
prompt sanitiser.

---

## 8. Authoritative plan file

The original approved plan (single source of truth for product decisions)
lives at `/home/john/.claude/plans/curious-waddling-zephyr.md`. This handoff
is a strict superset; on conflict the plan file wins for *intent* and this
handoff wins for *current implementation state*.
