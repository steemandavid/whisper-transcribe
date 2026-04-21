# Local GPU-Accelerated Whisper Transcription with Speaker Diarization

Self-hosted speech-to-text tool for transcribing audio recordings with speaker identification. Runs entirely locally inside a Docker container with NVIDIA GPU passthrough — no audio data leaves the machine.

Built around [WhisperX](https://github.com/m-bain/whisperX) (batched Whisper inference), [wav2vec2](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch) (word-level alignment), and [pyannote](https://github.com/pyannote/pyannote-audio) (speaker diarization). Supports automatic vocabulary prompt generation via a local LLM and post-transcription summarization.

## Features

- **High-quality transcription** — Whisper large-v3 model with beam_size=10, best_of=10
- **Speaker diarization** — identifies who said what using pyannote/speaker-diarization-3.1
- **Word-level timestamps** — precise alignment via wav2vec2
- **Auto-prompt** — iterative pipeline that uses a local LLM to extract domain-specific vocabulary from the audio itself
- **Manual prompts** — pass a vocabulary file for known domains
- **Summaries** — LLM-generated summary, decisions, and action items appended to the transcript
- **GPU-accelerated** — ~2 minutes for a 42-minute recording on RTX 3090
- **Privacy-first** — all processing happens locally, no cloud APIs

## Architecture

```
audio.m4a
     |
     v
+---------------------------------------------+
|  Docker container: whisper-transcribe        |
|                                              |
|  1. WhisperX (large-v3 model, pre-loaded)    |
|     -> Transcription with vocabulary prompt  |
|     -> beam_size=10, best_of=10, temp=0      |
|                                              |
|  2. wav2vec2 alignment (language-specific)   |
|     -> Word-level timestamp alignment        |
|                                              |
|  3. pyannote speaker diarization             |
|     -> Identifies SPEAKER_00, SPEAKER_01...  |
|                                              |
|  Output: .txt with speaker labels            |
+---------------------------------------------+
     |
     v
NVIDIA GPU (CUDA, ~8 GB peak VRAM)
```

## Project Structure

```
~/claudecode/projects/whisper/
+-- Dockerfile            Container definition (CUDA runtime + WhisperX + pyannote)
+-- transcribe            Main bash wrapper script (entry point)
+-- model.bin             Pre-downloaded large-v3 model (2.9 GB)
+-- large-v3-support/     Tokenizer, vocab, config files for large-v3
|   +-- config.json
|   +-- preprocessor_config.json
|   +-- tokenizer.json
|   +-- vocabulary.json
+-- .gitignore            Excludes model.bin and large-v3-support/ from git
+-- README.md             This file

~/.config/whisper/
+-- hf-token              HuggingFace API token (read-only mount into container)

~/Documents/transcribe nl/
+-- prompt-medisch.txt            Example vocabulary prompt file
```

### Docker Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Image | `whisper-transcribe` | ~7 GB with baked-in large-v3 model |
| Volume | `whisper-hf-cache` | Persistent HuggingFace model cache |
| Token mount | `~/.config/whisper/hf-token` | HuggingFace credentials (read-only) |

## Usage

### Basic transcription

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" --model large-v3 --language nl
```

### Quick transcription with quality preset

```bash
# Fast — small model, lower accuracy, quickest results
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality fast --language nl

# Good — large-v3 model with iterative auto-prompt (recommended for most use cases)
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality good --language nl --auto-prompt

# Perfect — large-v3 for all stages including scans, maximum quality
~/claudecode/projects/whisper/transcribe "./file.m4a" --quality perfect --language nl --auto-prompt
```

### With a manual vocabulary prompt

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --prompt prompt-gemeentediensten.txt
```

### With auto-generated vocabulary prompt

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --auto-prompt
```

### With summary appended to transcript

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --summary
```

### Combined: auto-prompt + summary

```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --auto-prompt --summary
```

### Output format

The transcript is written to a `.txt` file alongside the audio file:

```
[SPEAKER_00]
  Goedemorgen, ik heb een afspraak om 9 uur.
  Ik wilde het projectoverzicht bespreken.

[SPEAKER_01]
  Ja, kom maar binnen.
  Waar kan ik u mee helpen?
```

With `--summary`, a structured summary is appended:

```
---

## Samenvatting
Het team besprak de voortgang van het bouwproject en de
leveringsproblemen bij de hoofdaannemer...

## Beslissingen
- De fundering wordt volgende week afgerond conform planning...
- Er wordt overgestapt op een alternatieve leverancier voor staal...

## Actiepunten
- [ ] Offerte aanvragen bij de nieuwe leverancier voor staalprofelen...
- [ ] Planning update delen met de opdrachtgever uiterlijk vrijdag...
```

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
| `--ollama-model` | `-o` | Ollama model name | auto-select | Override Ollama model selection |
| `--help` | `-h` | | | Show usage |

### Mutually exclusive options

- `--auto-prompt` and `--prompt` cannot be used together

### Notes

- The script requires `sudo` to start/stop Ollama for VRAM management
- Audio files are mounted into the container from their current directory — no copying needed
- Output is always written to the same directory as the input file

## Quality Presets

The `--quality` flag sets a preset that configures the scan model (used in auto-prompt quick scans), the final transcription model, and quality parameters. Use it instead of manually specifying `--model` and quality params.

### Presets

| Preset | Scan model | Final model | beam_size | best_of | batch_size | Use case |
|--------|-----------|-------------|-----------|---------|------------|----------|
| `fast` | base | small | 2 | 2 | 16 | Quick drafts, testing |
| `medium` | base | medium | 5 | 5 | 12 | Good balance of speed and quality |
| `good` | medium | large-v3 | 10 | 10 | 8 | High quality (recommended) |
| `perfect` | large-v3 | large-v3 | 10 | 10 | 8 | Maximum quality, slower scans |

### Interaction with `--model`

- `--model` overrides the final transcription model from the quality preset
- `--quality good --model medium` uses medium scan + medium final (overriding large-v3)
- Without `--quality`, the default is medium model with max quality parameters (backward compatible)

### Examples

```bash
# Recommended for most use cases
~/claudecode/projects/whisper/transcribe "audio.m4a" --quality good --language nl

# Quick draft to check what's in a recording
~/claudecode/projects/whisper/transcribe "audio.m4a" --quality fast --language nl

# Maximum quality with auto-prompt and summary
~/claudecode/projects/whisper/transcribe "audio.m4a" --quality perfect --language nl --auto-prompt --summary

# Override just the final model while keeping preset quality params
~/claudecode/projects/whisper/transcribe "audio.m4a" --quality fast --model large-v3 --language nl
```

### Performance by quality level

For a ~42-minute audio file on RTX 3090:

| Quality | Without auto-prompt | With auto-prompt |
|---------|-------------------|-----------------|
| fast | ~30s | ~1 min |
| medium | ~1 min | ~1.5 min |
| good | ~2 min | ~2.5 min |
| perfect | ~2 min | ~4 min |

The "perfect" preset is slower with auto-prompt because the quick scan stages use large-v3 instead of medium (~42s per scan instead of ~10s).

## Auto-Prompt Pipeline

The `--auto-prompt` flag enables a 5-step iterative pipeline that generates a vocabulary prompt automatically, using a local LLM running via Ollama.

### How it works

```
audio.m4a
  |
  +-- Step 1: Quick scan (Whisper medium model, ~10s on GPU)
  |    -> rough transcript text
  |
  +-- Step 2: Keyword extraction (Ollama LLM)
  |    -> comma-separated vocabulary list (prompt v1)
  |
  +-- Step 3: Refined scan (Whisper medium + prompt v1, ~10s on GPU)
  |    -> cleaner transcript text
  |
  +-- Step 4: Keyword refinement (Ollama LLM)
  |    -> refined vocabulary list (prompt v2)
  |
  +-- Step 5: Full transcription (Whisper large-v3 + prompt v2)
       -> aligned, speaker-identified transcript
```

The iteration matters. Step 1 produces rough text with misrecognized words. The LLM extracts keywords from this rough text (step 2), which may include some errors. But feeding those keywords back into a second scan (step 3) produces significantly cleaner text, because the vocabulary hints correct the worst misrecognitions. The LLM then refines the keyword list against the cleaner text (step 4), removing false positives and adding missed terms. The refined prompt goes into the final large-v3 transcription.

### Ollama model selection

When `--ollama-model` is not specified, the script auto-selects the best available model using `select_ollama_model()`:

- Scans `ollama list` for installed models
- Excludes embedding models
- Penalizes coder-specific models (0.7x multiplier)
- Penalizes qwen3 models (0.5x — aggressive thinking mode cannot be reliably disabled)
- Bonuses instruct variants (1.3x — better at following extraction prompts)
- Applies a generation bonus based on version number (e.g. qwen3 > qwen2.5)

### VRAM management

Ollama normally consumes ~20 GB of the 24 GB VRAM on an RTX 3090. The pipeline manages the lifecycle:

1. Ollama stopped -> Step 1 (scan model, GPU — varies by `--quality`)
2. Ollama started -> Step 2 (LLM keyword extraction, CPU)
3. Ollama stopped -> Step 3 (scan model with prompt v1, GPU)
4. Ollama started -> Step 4 (LLM keyword refinement, CPU)
5. Ollama stopped -> Step 5 (final model, GPU)
6. On exit: Ollama restarted only if it was running before the script

### Fallback behavior

| Condition | Behavior |
|-----------|----------|
| Quick scan produces no output | Warn, fall back to no prompt |
| Ollama not installed or not running | Warn, fall back to no prompt |
| Model not found in `ollama list` | Warn, show available models, fall back to no prompt |
| Ollama returns empty result | Warn, fall back to no prompt |
| Any error in steps 1-4 | Script continues with full transcription (no prompt) |

## Summary Feature

The `--summary` flag generates a structured summary after transcription using a local LLM via Ollama. The summary includes:

1. **Samenvatting** — concise overview (3-5 sentences)
2. **Beslissingen** — key decisions or conclusions
3. **Actiepunten** — action items with checkboxes

The LLM is instructed to respond in the same language as the transcript and to include specific names, places, and numbers.

### Combining with auto-prompt

Both flags can be used together. The auto-prompt pipeline runs first to produce an optimal vocabulary, then the final transcription uses that vocabulary, and the summary is generated from the resulting transcript.

## Vocabulary Prompt Files

A vocabulary prompt file is a plain text file containing domain-specific terms that improve Whisper's recognition accuracy. It is passed via Whisper's `initial_prompt` parameter, which primes the model before transcription begins.

### Example (medical consultation context)

```
Dit is een medisch consult. De gesprekken gaan over hypertensie,
diabetes mellitus, cholesterol, bloeddruk, receptuur, huisarts,
specialist, verwijzing, bloedonderzoek, recept, dosering,
bijwerkingen, chronische aandoening, preventief onderzoek,
vaccinatie, longfunctie, physiotherapie, orthopedie, revalidatie.
```

### Guidelines for writing prompt files

- Keep it under ~1500 characters (Whisper's initial_prompt has limits)
- Include proper nouns that the model might misrecognize
- Include domain-specific jargon and technical terms
- Use the same language as the audio
- Natural sentences work better than bare word lists
- Create separate files for different contexts (medical, legal, business, etc.)

## Quality Settings

Quality parameters are set via `--quality` presets (see Quality Presets above). The parameters that vary between presets:

| Parameter | fast | medium | good / perfect | Purpose |
|-----------|------|--------|----------------|---------|
| `beam_size` | 2 | 5 | 10 | Searches more candidate translations |
| `best_of` | 2 | 5 | 10 | Samples more candidates before beam search |
| `batch_size` | 16 | 12 | 8 | Smaller batches for more careful processing |

These parameters are always enabled regardless of preset:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperatures` | [0] | Deterministic greedy decoding |
| `condition_on_previous_text` | True | Uses prior context for consistency |
| `compression_ratio_threshold` | 2.4 | Filter unlikely segments |
| `no_speech_threshold` | 0.6 | Skip silence |
| `log_prob_threshold` | -1.0 | Filter low-confidence segments |

## Performance

For a ~42-minute audio file on an NVIDIA RTX 3090 (with `--quality good`, i.e. large-v3 model):

### Standard transcription (no auto-prompt)

| Stage | Time |
|-------|------|
| Model load | ~5s |
| Transcription | ~42s (58x realtime) |
| Alignment | ~5s |
| Speaker diarization | ~60s |
| **Total** | **~2 min** |

### With auto-prompt enabled

| Stage | Time |
|-------|------|
| Quick scan (medium model) | ~10s |
| Keyword extraction (LLM) | ~15s |
| Refined scan (medium model) | ~10s |
| Keyword refinement (LLM) | ~15s |
| Final transcription + alignment + diarization | ~2 min |
| **Total** | **~2.5 min** |

See the Quality Presets section for timing at different quality levels.

### With summary

Adds ~30s for the LLM summary generation (Ollama start + inference + stop).

### VRAM usage

| Component | VRAM |
|-----------|------|
| WhisperX + large-v3 | ~4.5 GB |
| Alignment model | ~1 GB (loaded then unloaded) |
| Diarization model | ~2 GB (loaded then unloaded) |
| **Peak** | **~8 GB** |

Comfortable fit with 24 GB RTX 3090. Ollama (~20 GB) is stopped during GPU operations.

## Prerequisites

### Hardware

- NVIDIA GPU with at least 8 GB VRAM
- Sufficient disk space: ~7 GB for Docker image, ~3 GB for HuggingFace cache volume

### Software

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- `jq` (for Ollama API JSON processing)
- Optional: Ollama with a suitable model for `--auto-prompt` and `--summary`

Verify GPU passthrough:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi
```

### HuggingFace account setup

1. Create account at https://huggingface.co/
2. Generate a read token at https://huggingface.co/settings/tokens
3. Accept terms for these gated models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Save token to `~/.config/whisper/hf-token`

The token is required for speaker diarization. Without it, transcription and alignment still work but speakers won't be identified.

### Ollama setup (optional, for auto-prompt and summary)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download a model (recommended)
ollama pull qwen2.5:32b-instruct-q4_K_M
```

The auto-prompt and summary features work with any Ollama model. The script auto-selects the best available model. Recommended: a 30B+ instruct model for best keyword extraction quality.

## Building from Scratch

### 1. Clone the repository

```bash
git clone https://github.com/steemandavid/whisper-transcribe.git ~/claudecode/projects/whisper
cd ~/claudecode/projects/whisper
```

### 2. Download the large-v3 model

```bash
# Download model weights (2.9 GB)
wget -O model.bin https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin

# Download supporting files
mkdir -p large-v3-support
for f in config.json preprocessor_config.json tokenizer.json vocabulary.json; do
    wget -O "large-v3-support/$f" "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/$f"
done
```

These files are excluded from git via `.gitignore` due to their size.

### 3. Build the Docker image

```bash
docker build -t whisper-transcribe .
```

The image is ~7 GB (includes CUDA runtime, WhisperX, pyannote, and the large-v3 model). The build takes 5-10 minutes depending on download speed.

### 4. Create the HuggingFace cache volume

```bash
docker volume create whisper-hf-cache
```

This stores alignment and diarization models after their first download, so they don't need to be re-downloaded on every run.

### 5. Run

```bash
~/claudecode/projects/whisper/transcribe "./audio.m4a" --model large-v3 --language nl
```

## Troubleshooting

### GPU not used / CUDA out of memory

- Check what's consuming VRAM: `nvidia-smi`
- Ollama typically uses ~20 GB — the script handles stopping/restarting it automatically
- If another process is using the GPU, stop it before running transcription
- Verify GPU passthrough: `docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi`

### `libcublas.so.12` not found

The Dockerfile uses `nvidia/cuda:12.4.0-runtime-ubuntu22.04` (runtime variant). If you modify the Dockerfile, do **not** use the `base` variant — it's missing the required CUDA runtime libraries.

### Model download hangs at build time

- Without a HuggingFace token, downloads are rate-limited
- Pass `--build-arg HF_TOKEN=$(cat ~/.config/whisper/hf-token)` to `docker build` for faster downloads
- Alternatively, download `model.bin` and supporting files manually (see Building from Scratch)

### Alignment model download hangs at runtime

- Ensure the HF token is saved at `~/.config/whisper/hf-token`
- The `whisper-hf-cache` Docker volume persists downloaded models across runs
- If a previous download failed, a `.no_exist` marker file may block future downloads:
  ```bash
  # Clean stale markers from the volume
  docker run --rm -v whisper-hf-cache:/cache alpine find /cache -name '.no_exist' -delete
  ```

### `GatedRepoError: 403`

Accept the model terms on the HuggingFace website (see Prerequisites). There are three separate models to accept — the error message doesn't tell you which one is missing.

### Ollama returns empty results for auto-prompt

- qwen3 models have aggressive thinking mode that can't be reliably disabled — the script penalizes them in model selection
- Use `--ollama-model qwen2.5:32b-instruct-q4_K_M` to force a specific model
- Check that Ollama is running: `systemctl status ollama`
- Check available models: `ollama list`

### Docker image not building

- Ensure the model files are in place: `ls -la model.bin large-v3-support/`
- Ensure `model.bin` is ~2.9 GB (a truncated download will cause build failures)
- Check disk space — the build needs ~10 GB temporarily

### Transcript file is owned by root

The container runs as root, so output files are created as root. The script handles this with `os.chmod(0o666)` inside the container to make the file writable by all users.

## Backup Coverage

| What | Location | Backup |
|------|----------|--------|
| Docker image data | `/storage/docker` | Daily Borg -> USB, Weekly -> NAS |
| Docker volume (HF cache) | Docker-managed at `/storage/docker/volumes/whisper-hf-cache/` | Included in Docker data backups |
| Project files (script, Dockerfile) | `~/claudecode/projects/whisper/` | Daily system backup |
| HF token | `~/.config/whisper/hf-token` | Daily system backup |
| Prompt files | `~/Documents/transcribe nl/` | Daily system backup |
| Model weights | `model.bin` + `large-v3-support/` | Excluded from git; re-downloadable from HuggingFace |
