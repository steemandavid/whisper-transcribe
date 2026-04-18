Write a blog post about setting up a local, GPU-accelerated Whisper transcription pipeline with speaker diarization, running entirely in Docker on a home server.

## Context

I built this solution to transcribe Dutch/Flemish audio recordings of conversations with local municipal council departments. The recordings contain informal spoken Flemish with dialect, background noise, and specialized government jargon — making accurate transcription challenging. The setup runs on a home server with an NVIDIA RTX 3090 and processes audio entirely locally (no data leaves the machine).

## Source material

Reference these files for technical details, architecture, and the full story of how the solution was built:

- **Project README**: `/home/john/claudecode/projects/whisper/README.md` — full architecture, usage, design decisions, and troubleshooting
- **Changelog**: `/home/john/claudecode/changelogs/2026-04-18-whisper-transcription-container.md` — step-by-step evolution of the project, including all issues encountered and fixes
- **GitHub**: The project lives at `https://github.com/steemandavid` (account: steemandavid)

## What the blog post should cover

1. **The problem**: Why I needed local transcription (privacy, Flemish dialect, government jargon) and why cloud services weren't suitable
2. **The solution architecture**: WhisperX + word-level alignment + pyannote speaker diarization, all in a Docker container with GPU passthrough
3. **Key design decisions**:
   - Pre-loading the large-v3 model into the Docker image to avoid runtime downloads
   - Using vocabulary prompt files to improve recognition of domain-specific terms
   - Auto-stopping Ollama to free VRAM for transcription
   - Persistent Docker volume for caching alignment and diarization models
   - Credentials stored outside the container
4. **The journey**: Briefly touch on the iterative debugging process — CUDA library issues, HuggingFace gated models, API changes in pyannote, the m4a decoding problem. Not exhaustive, but enough to show it wasn't trivial.
5. **Performance**: A 42-minute audio file transcribed in ~2 minutes with speaker labels on an RTX 3090
6. **Results**: Show example output format with speaker labels and how the vocabulary prompt improved accuracy
7. **Reusability**: How the setup is generic — swap the prompt file and it works for any transcription context

## Tone and style

- Technical but accessible — aimed at someone comfortable with Docker and Linux
- Practical and hands-on — show real commands and real output
- Honest about the bumps along the way
- Written in first person
- Blog is at https://www.steeman.be/ — this would be a technical blog post

## Language

Write the blog post in English.
