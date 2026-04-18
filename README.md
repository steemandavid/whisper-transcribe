# Local GPU-Accelerated Whisper Transcription with Speaker Diarization

## Overview

This document describes a self-hosted speech-to-text setup for transcribing Dutch/Flemish audio recordings with speaker identification. It runs entirely locally on an NVIDIA RTX 3090 GPU inside a Docker container, with no data leaving the machine.

The setup was built to transcribe conversations recorded at a local Flemish municipal council (mobiliteitsdienst, stedenbouwdienst), where accurate recognition of regional dialect, place names, and government jargon is critical.

---

## Architecture

```
Audio file (.m4a)
     │
     ▼
┌─────────────────────────────────────────────┐
│  Docker container: whisper-transcribe        │
│                                              │
│  1. WhisperX (large-v3 model, pre-loaded)    │
│     → Transcription with vocabulary prompt   │
│     → beam_size=10, best_of=10, temp=0       │
│                                              │
│  2. wav2vec2 alignment (Dutch)               │
│     → Word-level timestamp alignment         │
│                                              │
│  3. pyannote speaker diarization             │
│     → Identifies SPEAKER_00, SPEAKER_01...   │
│                                              │
│  Output: .txt with speaker labels            │
└─────────────────────────────────────────────┘
     │
     ▼
RTX 3090 (CUDA, ~8 GB peak VRAM)
```

---

## Files & Locations

| File | Purpose |
|------|---------|
| `~/claudecode/projects/whisper/Dockerfile` | Container definition |
| `~/claudecode/projects/whisper/transcribe` | Main bash wrapper script |
| `~/claudecode/projects/whisper/model.bin` | Pre-downloaded large-v3 model (2.9 GB) |
| `~/claudecode/projects/whisper/large-v3-support/` | Tokenizer, vocab, config files |
| `~/.config/whisper/hf-token` | HuggingFace API token (read-only mount) |
| `~/Documents/transcribe nl/prompt-gemeentediensten.txt` | Flemish municipal vocabulary prompt |

---

## Usage

### Basic transcription
```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" --model large-v3 --language nl
```

### With vocabulary prompt and speaker diarization
```bash
~/claudecode/projects/whisper/transcribe "./file.m4a" \
    --model large-v3 --language nl --prompt prompt-gemeentediensten.txt
```

### Output format
```
[SPEAKER_00]
  Goedemorgen, ik ben David Steeman.
  Ik heb een afspraak om 9 uur.

[SPEAKER_01]
  Ja, kom maar binnen.
  Wat kan ik voor u doen?
```

---

## Key Design Decisions

### 1. Pre-loaded model in Docker image
The large-v3 model (2.9 GB) is baked into the Docker image. This avoids slow/unreliable downloads at runtime. The image is ~7 GB but builds once and is cached.

### 2. Persistent Docker volume for runtime model cache
A Docker volume `whisper-hf-cache` stores alignment and diarization models after their first download. Without this, every run would re-download ~3 GB of models since the container is ephemeral (`--rm`).

### 3. Credentials outside the container
The HuggingFace token lives at `~/.config/whisper/hf-token` and is mounted read-only into the container. No secrets are baked into the Docker image.

### 4. Ollama auto-stop/start
Ollama normally consumes ~20 GB of the 24 GB VRAM. The script auto-detects Ollama, stops it, runs transcription, and restarts it on exit (even on error) using `trap cleanup EXIT`.

### 5. Vocabulary prompts
A plain text file with domain-specific terms is passed via `--prompt`. Whisper uses this as an initial prompt to guide recognition. This significantly improves accuracy for proper nouns and jargon. Different prompt files can be created for different transcription contexts.

### 6. Max quality settings
- `beam_size=10` — searches more candidate translations
- `best_of=10` — samples more candidates before beam search
- `temperature=0` — deterministic greedy decoding
- `condition_on_previous_text=True` — uses prior context for consistency
- `batch_size=8` — smaller batches for more careful processing

---

## Performance

For a ~42-minute audio file on RTX 3090:

| Stage | Time |
|-------|------|
| Model load | ~5s |
| Transcription | ~42s (58x realtime) |
| Alignment | ~5s |
| Speaker diarization | ~60s |
| **Total** | **~2 min** |

---

## Prerequisites

### Docker with NVIDIA GPU support
```bash
# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi
```

### HuggingFace account setup
1. Create account at https://huggingface.co/
2. Generate token at https://huggingface.co/settings/tokens
3. Accept terms for gated models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Save token to `~/.config/whisper/hf-token`

---

## Building from Scratch

### 1. Download the model
```bash
# Download from browser or wget
wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin
# Also download: config.json, preprocessor_config.json, tokenizer.json, vocabulary.json
```

### 2. Place files
```bash
mkdir -p ~/claudecode/projects/whisper/large-v3-support
mv model.bin ~/claudecode/projects/whisper/
mv config.json preprocessor_config.json tokenizer.json vocabulary.json ~/claudecode/projects/whisper/large-v3-support/
```

### 3. Build and run
```bash
cd ~/claudecode/projects/whisper
docker build -t whisper-transcribe .
~/claudecode/projects/whisper/transcribe "./audio.m4a" --model large-v3 --language nl
```

---

## Troubleshooting

### "RTX not spinning up" / GPU not used
- Check VRAM: `nvidia-smi` — Ollama may be consuming it
- Verify GPU passthrough: `docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi`

### "libcublas.so.12 not found"
- Use `nvidia/cuda:12.4.0-runtime-ubuntu22.04` (runtime variant), not `base`

### Alignment model download hangs
- Ensure HF token is saved at `~/.config/whisper/hf-token`
- Clean stale cache markers: remove `.no_exist` from Docker volume
- Pre-download: run the manual download command from the changelog

### "GatedRepoError: 403"
- Accept model terms on HuggingFace website (see Prerequisites)

### "DiarizeOutput has no attribute 'itertracks'"
- New pyannote API uses `.speaker_diarization` property instead of direct iteration

---

## Session Context (2026-04-18)

This setup was built to transcribe two conversations recorded at a Flemish municipal council:
1. **Mobiliteit 20260413** (~42 min) — Discussion about requesting a bicycle street (fietsstraat) and information about Aquafin sewer collector works and parking construction in Mespelare
2. **Stedenbouw 20260413** (~30 min) — Discussion about expropriation (onteigening) related to the same works, zoning questions about parcel 149B, and historical subdivision (verkaveling) from 1983

The audio quality was challenging: informal spoken Flemish with dialect words, spoken in a room with background noise and paper shuffling. The vocabulary prompt was crafted from a review of the existing (lower quality) transcription to capture the key terms.
