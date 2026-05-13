# Whisper Transcription Pipeline v2

Self-hosted, GPU-accelerated speech-to-text with speaker diarization, multi-engine ASR, and LLM post-correction. Runs entirely locally in Docker — no audio data leaves the machine.

## What's New in v2

- **Multi-stage pipeline** — audio preprocessing, dual-engine ASR with ROVER reconciliation, LLM post-correction
- **Audio preprocessing** — loudness normalisation, high-pass filtering, channel splitting for phone calls
- **Two outputs** — verbatim transcript + LLM-cleaned transcript with corrected brand names and jargon
- **Glossary support** — `wrong → right` mappings used in ASR, ROVER, and post-correction
- **Python pipeline modules** — all heavy logic in `pipeline/`, bash script is just the orchestrator

## Features

- **High-quality transcription** — Whisper large-v3 model with beam_size=10, best_of=10
- **Speaker diarization** — identifies who said what using pyannote/speaker-diarization-3.1
- **Word-level timestamps** — precise alignment via wav2vec2
- **Auto-prompt** — iterative pipeline that uses a local LLM to extract domain-specific vocabulary
- **Manual prompts** — pass a vocabulary file for known domains
- **Summaries** — LLM-generated summary, decisions, and action items appended to the transcript
- **LLM post-correction** — fix brand names and phonetic mishearings (local Ollama or cloud GLM via Z.ai Anthropic-compatible API)
- **GPU-accelerated** — ~2 minutes for a 42-minute recording on RTX 3090
- **Privacy-first** — all processing happens locally, no cloud APIs (unless `--cloud-correct`)

## Architecture

```
audio.m4a
     |
     v
+---------------------------------------------+
|  Stage 1: Audio Preprocessing               |
|  - Loudness normalisation (ffmpeg loudnorm)  |
|  - High-pass 80 Hz filter                    |
|  - Stereo channel splitting for phone calls  |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
|  Stage 2: ASR (WhisperX large-v3)            |
|  - Engine A: large-v3 with max quality       |
|  - Engine B: Dutch fine-tune (nl audio only) |
|  Output: asr_engine_a.json                   |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
|  Stage 3: ROVER Reconciliation (optional)    |
|  - Merges Engine A + Engine B results        |
|  - Without --ensemble, Engine A used as-is   |
|  - Glossary-weighted majority vote           |
|  Output: rover.json                          |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
|  Stage 4: Speaker Diarization                |
|  - pyannote/speaker-diarization-3.1          |
|  - Per-channel mode for stereo phone calls   |
|  Output: diarize.json                        |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
|  Stage 5: LLM Post-Correction (optional)     |
|  - Fixes brand names and phonetic errors     |
|  - Local (Ollama) or cloud (GLM via Z.ai)    |
|  - Cloud uses Anthropic Messages API         |
|  Output: cleaned.json                        |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
|  Stage 6: Render                             |
|  - audio.txt (verbatim, speaker-labeled)     |
|  - audio.cleaned.txt (post-corrected)        |
+---------------------------------------------+
     |
     v
NVIDIA GPU (CUDA, ~8 GB peak VRAM)
```

## Project Structure

```
~/claudecode/projects/whisper/
+-- Dockerfile            Multi-stage build (CUDA runtime + WhisperX + pyannote)
+-- transcribe            Bash orchestrator (entry point)
+-- pipeline/             Python pipeline modules
|   +-- __init__.py
|   +-- artifacts.py      RunPaths + JSON helpers
|   +-- glossary.py       Glossary loader + seeder
|   +-- prompt_builder.py Prompt sanitiser + builder
|   +-- preprocess.py     Audio preprocessing
|   +-- asr_engines.py    Multi-engine ASR
|   +-- rover.py          ROVER reconciliation
|   +-- diarize.py        Speaker diarization
|   +-- postcorrect.py    LLM post-correction
|   +-- render.py         Transcript text rendering
+-- model.bin             Pre-downloaded large-v3 model (2.9 GB)
+-- large-v3-support/     Tokenizer, vocab, config files
+-- .gitignore
+-- README.md

~/.config/whisper/
+-- hf-token              HuggingFace API token (read-only mount)
+-- glossary.txt          Default glossary (wrong → right mappings)
+-- zai-key               Z.ai API key (optional, for --cloud-correct)
+-- .glm-resolved         Cached GLM model id (auto-managed, 24h TTL)

~/Documents/transcribe nl/
+-- prompt-medisch.txt    Example vocabulary prompt file
```

### Docker Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Image | `whisper-transcribe:2.0` | ~20 GB with pre-baked models |
| Volume | `whisper-hf-cache` | Persistent HuggingFace model cache |

## Usage

### Quick start with quality preset

```bash
# Recommended for most use cases
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality good --language nl

# Maximum quality with all enhancements
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality perfect

# Quick draft
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality fast --language nl
```

### With auto-prompt and summary

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --quality perfect --language nl --auto-prompt --summary
```

### With LLM post-correction

```bash
# Local LLM correction (Ollama)
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --quality perfect --language nl --auto-prompt --summary --correct

# Cloud correction (GLM via Z.ai, requires zai-key)
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --quality perfect --language nl --cloud-correct
```

### With manual vocabulary prompt

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --prompt prompt-file.txt
```

### Output format

Two files are produced (verbatim and cleaned):

```
# audio.txt (verbatim)
[SPEAKER_00]
  Goedemorgen, ik heb hier net mijn man met de vloek.
  Ik weet niet of we al een gigantisch veel werk...

[SPEAKER_01]
  Ja, kom maar binnen. Waar kan ik u mee helpen?
```

```
# audio.cleaned.txt (post-corrected)
[SPEAKER_00]
  Goedemorgen, ik heb hier net mijn man met de Fluke.
  Ik weet niet of we al een gigantisch veel werk...
```

With `--summary`, a structured summary is appended to the verbatim file:

```
---

## Samenvatting
Het team besprak de voortgang van het bouwproject...

## Beslissingen
- De fundering wordt volgende week afgerond...

## Actiepunten
- [ ] Offerte aanvragen bij de nieuwe leverancier...
```

### Pipeline summary

At the end of each run, the script prints a summary of which stages ran vs were skipped:

```
=== Pipeline summary ===
Ran:     preprocess asr (large-v3) diarize post-correct (12 segments fixed)
Skipped: ensemble (engine B unavailable)

=== Pipeline summary ===
Ran:     preprocess asr (large-v3) ensemble (engine B + ROVER) diarize post-correct (8 segments fixed)
Skipped: (none)
```

This makes it clear which stages were actually active for a given run, especially when using `--quality perfect` where some stages may silently skip (e.g. DeepFilterNet or AudioSR if not installed).

## CLI Reference

```
transcribe <audio_file> [options]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `audio_file` | Path to audio/video file (m4a, mp3, wav, mp4, etc.) |

### Flags

| Flag | Short | Values | Default | Description |
|------|-------|--------|---------|-------------|
| `--quality` | `-Q` | fast, medium, good, perfect | custom | Quality preset (see below) |
| `--model` | `-m` | tiny, base, small, medium, large-v3 | medium | Whisper model size |
| `--language` | `-l` | en, nl, fr, de, auto | auto-detect | Language of the audio |
| `--prompt` | `-p` | path to text file | none | Vocabulary/context hints file |
| `--auto-prompt` | `-a` | (boolean flag) | off | Auto-generate vocabulary via Ollama |
| `--summary` | `-s` | (boolean flag) | off | Append LLM summary to transcript |
| `--enhance` | | (boolean flag) | off | Enable audio preprocessing |
| `--denoise` | | (boolean flag) | off | Enable DeepFilterNet3 denoising |
| `--ensemble` | | (boolean flag) | off | Run multi-engine ASR with ROVER |
| `--correct` | | (boolean flag) | off | LLM post-correction (local Ollama) |
| `--cloud-correct` | | (boolean flag) | off | LLM post-correction (cloud GLM via Z.ai) |
| `--no-enhance` | | (boolean flag) | off | Override preset: disable audio preprocessing |
| `--no-denoise` | | (boolean flag) | off | Override preset: disable denoising |
| `--no-ensemble` | | (boolean flag) | off | Override preset: disable multi-engine ASR |
| `--no-correct` | | (boolean flag) | off | Override preset: disable post-correction |
| `--refresh-model` | | (boolean flag) | off | Force re-query of GLM models endpoint |
| `--context` | | free text | none | Context for prompt building |
| `--speakers` | | number | auto | Pin speaker count for diarization |
| `--glossary` | | path | `~/.config/whisper/glossary.txt` | Glossary file |
| `--diarize-model` | | 3.1, community-1 | 3.1 | Diarization model variant |
| `--ollama-model` | `-o` | model name | auto-select | Override Ollama model |
| `--keep-intermediates` | | (boolean flag) | off | Preserve scratch dir for debugging |

### Mutually exclusive options

- `--auto-prompt` and `--prompt` cannot be used together

## Quality Presets

The `--quality` flag sets a preset that configures models and enhancement flags:

| Preset | Scan model | Final model | Enhance | Denoise | Ensemble | Correct | Use case |
|--------|-----------|-------------|---------|---------|----------|---------|----------|
| `fast` | base | small | off | off | off | off | Quick drafts |
| `medium` | base | medium | off | off | off | off | Good balance |
| `good` | medium | large-v3 | off | off | off | off | High quality |
| `perfect` | large-v3 | large-v3 | on | on | on | on (local) | Maximum quality |

- `--model` overrides the final transcription model from the preset
- `--no-enhance`, `--no-denoise`, `--no-ensemble`, `--no-correct` override individual preset flags (e.g. `--quality perfect --no-correct`)
- Without `--quality`, the default is medium model with max quality parameters

### Performance by quality level

For a ~42-minute audio file on RTX 3090:

| Quality | Time | Notes |
|---------|------|-------|
| fast | ~30s | Small model |
| medium | ~1 min | Medium model |
| good | ~2 min | Large-v3, no enhancements |
| perfect | ~3-5 min | Large-v3 + all enhancements + post-correction |

## Glossary

The glossary maps common misrecognitions to their correct forms. It's used in three places:
1. **Prompt builder** — canonical terms appended to vocabulary hints
2. **ROVER tie-breaker** — glossary terms preferred in reconciliation
3. **Post-correction** — full glossary embedded in LLM system prompt

### Format

```
# Comments start with #
# Sections: [brand], [person], [place], [term]
# Brands have highest weight, terms lowest

[brand]
vloek -> Fluke
annexter -> Anixter
comscope -> CommScope
connect-wise -> ConnectWise

[person]
brent -> Brent

[place]
berendrechtstraat -> Berendrechtstraat
```

### Locations

The glossary is loaded from the first found:
1. `--glossary FILE` (explicit override)
2. `/run/glossary.txt` (container mount)
3. `~/.config/whisper/glossary.txt` (default)

## Auto-Prompt Pipeline

The `--auto-prompt` flag enables a multi-step pipeline that generates vocabulary automatically:

1. **Quick scan** — Whisper base/medium model produces rough transcript
2. **Keyword extraction** — Ollama LLM extracts domain terms from rough text
3. **Refined scan** — Re-run with extracted vocabulary for cleaner text
4. **Keyword refinement** — LLM refines keywords against cleaner text
5. **Final transcription** — Full large-v3 with refined prompt

The prompt builder sanitises all LLM output — stripping ANSI codes, thinking blocks, and enforcing length limits to prevent the "poisoned prompt" issue from v1.

### Ollama model selection

When `--ollama-model` is not specified, the script auto-selects the best available model, penalising qwen3 (thinking mode issues) and coder-specific models.

### VRAM management

Ollama and WhisperX share the GPU. The pipeline manages the lifecycle:
- Ollama stopped → WhisperX uses GPU
- Ollama started → LLM keyword extraction (CPU)
- On exit: Ollama restarted only if it was running before

## Post-Correction

The `--correct` or `--cloud-correct` flags run LLM post-correction on the transcript. The LLM receives the verbatim transcript, glossary, and context, and corrects phonetic mishearings while preserving the original meaning.

- **Local** (`--correct`): uses Ollama on the host
- **Cloud** (`--cloud-correct`): uses the latest GLM model via Z.ai Anthropic-compatible API (`/api/anthropic/v1/messages`)

Both produce a `.cleaned.txt` file alongside the verbatim `.txt`.

### Z.ai setup (optional, for cloud correction)

```bash
# Get an API key from Z.ai
echo "your-key" > ~/.config/whisper/zai-key
```

The model is auto-detected from the Z.ai models endpoint (OpenAI-compatible `/api/paas/v4/models`) and cached for 24 hours in `~/.config/whisper/.glm-resolved`. Use `--refresh-model` to force re-detection. Completions use the Anthropic-compatible endpoint (`/api/anthropic/v1/messages`) which shares the same API key.

When using `--cloud-correct`, token usage is logged to stderr and a `cleaned_usage.json` sidecar file is written to the scratch directory with accumulated totals across all batches.

## Vocabulary Prompt Files

A vocabulary prompt file is a plain text file containing domain-specific terms passed via Whisper's `initial_prompt` parameter.

### Example

```
Dit is een medisch consult. De gesprekken gaan over hypertensie,
diabetes mellitus, cholesterol, bloeddruk, receptuur, huisarts.
```

### Guidelines

- Keep it under ~1500 characters
- Include proper nouns and domain-specific jargon
- Use the same language as the audio
- Natural sentences work better than bare word lists

## Prerequisites

### Hardware

- NVIDIA GPU with at least 8 GB VRAM
- ~20 GB disk space for Docker image + models

### Software

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `jq` (for Ollama API JSON processing)
- Optional: Ollama with a model for `--auto-prompt`, `--summary`, and `--correct`

### HuggingFace account setup

1. Create account at https://huggingface.co/
2. Generate a read token at https://huggingface.co/settings/tokens
3. Accept terms for these gated models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Save token to `~/.config/whisper/hf-token`

The token is required for speaker diarization. Without it, transcription and alignment still work but speakers won't be identified.

## Building from Scratch

### 1. Clone and download model

```bash
git clone https://github.com/steemandavid/whisper-transcribe.git ~/claudecode/projects/whisper
cd ~/claudecode/projects/whisper

# Download model weights (2.9 GB)
wget -O model.bin https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin

# Download supporting files
mkdir -p large-v3-support
for f in config.json preprocessor_config.json tokenizer.json vocabulary.json; do
    wget -O "large-v3-support/$f" "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/$f"
done
```

### 2. Build Docker image

```bash
docker build -t whisper-transcribe:2.0 .
```

Build takes ~20 minutes on first run. The image pins specific versions of all dependencies (`ctranslate2==4.4.0`, `whisperx==3.4.2`, `pyannote.audio==3.4.0`, etc.). Pipeline code is volume-mounted at runtime, so the image does not need rebuilding for pipeline changes.

### 3. Create HuggingFace cache volume

```bash
docker volume create whisper-hf-cache
```

### 4. Seed the glossary

```bash
# Seed the default glossary (safe to re-run — won't overwrite an existing file):
docker run --rm whisper-transcribe:2.0 pipeline.glossary --seed
```

### 5. Run

```bash
~/claudecode/projects/whisper/transcribe "./audio.m4a" --quality perfect
```

## Debugging

### Per-stage invocation

Each pipeline stage can be run independently inside the container:

```bash
# Preprocessing only
docker run --rm --gpus all -v "$(pwd)":/audio \
    -v ~/claudecode/projects/whisper/pipeline:/opt/pipeline:ro \
    whisper-transcribe:2.0 \
    pipeline.preprocess /audio/file.m4a --scratch /audio/.scratch --enhance

# ASR only
docker run --rm --gpus all -v "$(pwd)":/audio \
    -v ~/claudecode/projects/whisper/pipeline:/opt/pipeline:ro \
    whisper-transcribe:2.0 \
    pipeline.asr_engines /audio/.scratch/preprocessed.wav --scratch /audio/.scratch

# Diarization only
docker run --rm --gpus all -v "$(pwd)":/audio \
    -v ~/claudecode/projects/whisper/pipeline:/opt/pipeline:ro \
    -v ~/.config/whisper/hf-token:/run/secrets/hf-token:ro \
    whisper-transcribe:2.0 \
    pipeline.diarize /audio/.scratch/preprocessed.wav --output /audio/.scratch/diarize.json

# Dump glossary
docker run --rm -v ~/claudecode/projects/whisper/pipeline:/opt/pipeline:ro \
    whisper-transcribe:2.0 pipeline.glossary --dump
```

### Intermediate files

Use `--keep-intermediates` to preserve the scratch directory (`.whisper-run-<pid>/`) for debugging. It contains JSON artifacts from each stage:

```
.whisper-run-12345/
+-- preprocessed.wav          Stage 1 output
+-- preprocess.json           Stage 1 metadata
+-- asr_engine_a.json         Stage 2 Engine A output
+-- diarize.json              Stage 4 diarization output
+-- cleaned.json              Stage 5 post-correction output
```

## Troubleshooting

### GPU not used / CUDA out of memory

- Check VRAM: `nvidia-smi`
- Ollama uses ~20 GB — the script manages stopping/restarting it
- Verify GPU passthrough: `docker run --rm --gpus all whisper-transcribe:2.0 python3 -c "import torch; print(torch.cuda.is_available())"`

### Post-correction returns 0 corrections

- **Local (`--correct`):** Ollama models may not be effective at correcting Flemish phonetics
- **Cloud (`--cloud-correct`):** uses the Z.ai Anthropic-compatible endpoint — requires a valid API key with credits at `~/.config/whisper/zai-key`
- If you see "Insufficient balance" errors, the Z.ai account needs credits (the same key may work for other Z.ai services on different endpoints)
- Ensure the glossary at `~/.config/whisper/glossary.txt` contains relevant entries

### Ollama sudo prompts

The script uses `sudo systemctl` to manage Ollama. If your user doesn't have passwordless sudo, Ollama will remain running. This works fine if Ollama isn't actively using the GPU (models load on demand).

### `libcudnn_ops_infer.so.8` warning

This is a harmless warning from a secondary code path. ctranslate2 4.5.0 uses cuDNN 9 correctly.

### Docker image not building

- Network timeouts during `pip install` are common — retry the build
- Ensure model files are in place: `ls -la model.bin large-v3-support/`
- Check disk space — the build needs ~25 GB temporarily

## Backup Coverage

| What | Location | Backup |
|------|----------|--------|
| Docker image data | `/storage/docker` | Daily Borg → USB, Weekly → NAS |
| Docker volume (HF cache) | Docker-managed | Included in Docker data backups |
| Project files | `~/claudecode/projects/whisper/` | Daily system backup |
| HF token | `~/.config/whisper/hf-token` | Daily system backup |
| Glossary | `~/.config/whisper/glossary.txt` | Daily system backup |
| Prompt files | `~/Documents/transcribe nl/` | Daily system backup |
| Model weights | `model.bin` + `large-v3-support/` | Re-downloadable from HuggingFace |
