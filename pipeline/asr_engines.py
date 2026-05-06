"""Multi-engine ASR: Engine A (WhisperX large-v3) and Engine B (Dutch fine-tune).

Both engines emit a uniform JSON schema with segments, word timestamps, and
per-chunk language identification. Per-chunk language gating: nl chunks run
both engines, non-nl chunks run Engine A only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ASR_OPTIONS = {
    "beam_size": 10,
    "best_of": 10,
    "temperatures": [0],
    "condition_on_previous_text": True,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
}

NL_MODEL_PATH = "/opt/whisper-models/nl-large-v3"
ALIGN_MODELS = {
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "voidful/wav2vec2-xlsr-multilingual-56",
}


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    lang: str
    words: list[dict]


def _run_whisperx_engine(
    audio_path: Path,
    output_json: Path,
    *,
    model_path: str,
    language: str | None = None,
    prompt: str | None = None,
    batch_size: int = 8,
    align_model: str | None = None,
    hf_token: str | None = None,
    engine_label: str = "A",
) -> dict:
    """Run WhisperX inside a Python subprocess (designed for Docker container)."""
    asr_opts = dict(ASR_OPTIONS)
    if prompt:
        asr_opts["initial_prompt"] = prompt

    script = f"""
import json, sys, os, logging
logging.getLogger().setLevel(logging.WARNING)
import whisperx

audio_path = {str(audio_path)!r}
output_path = {str(output_json)!r}
model_path = {model_path!r}
language = {language!r}
batch_size = {batch_size}
asr_options = {asr_opts!r}
engine_label = {engine_label!r}
align_model_name = {align_model!r}
hf_token = {hf_token!r}

device = "cuda"
compute_type = "float16"

model = whisperx.load_model(
    model_path, device, compute_type=compute_type,
    language=language, asr_options=asr_options,
)

audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio, batch_size=batch_size, language=language)

# Language chunks from WhisperX's built-in language ID
# faster-whisper populates per-segment 'language' when language=None (auto-detect).
# When language is forced, all segments get that same language.
top_lang = result.get("language", language or "nl")
lang_chunks = []
for seg in result.get("segments", []):
    seg_lang = seg.get("language", top_lang)
    lang_chunks.append({{"start": seg["start"], "end": seg["end"], "lang": seg_lang, "prob": seg.get("avg_logprob", 0.0)}})

# Align if we have an alignment model
if align_model_name:
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language or "nl", model_name=align_model_name, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    except Exception as e:
        print(f"ALIGN WARNING: {{e}}", file=sys.stderr)

# Build output
segments_out = []
for i, seg in enumerate(result.get("segments", [])):
    words_out = []
    for w in seg.get("words", []):
        words_out.append({{
            "start": w.get("start", 0.0),
            "end": w.get("end", 0.0),
            "word": w.get("word", "").strip(),
            "prob": w.get("probability", w.get("score", 0.0)),
        }})
    segments_out.append({{
        "id": i,
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"].strip(),
        "avg_logprob": seg.get("avg_logprob", 0.0),
        "lang": seg.get("language", top_lang),
        "words": words_out,
    }})

output = {{
    "engine": engine_label,
    "language_chunks": lang_chunks,
    "segments": segments_out,
}}
with open(output_path, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"Wrote {{len(segments_out)}} segments to {{output_path}}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"Engine {engine_label} stderr: {proc.stderr}", file=sys.stderr)
        raise RuntimeError(f"Engine {engine_label} failed: {proc.stderr[:500]}")

    return json.loads(output_json.read_text())


def run_engine_a(
    audio_path: Path,
    scratch: Path,
    *,
    prompt: str | None = None,
    language: str | None = None,
    model_path: str = "/opt/whisper-models/large-v3",
    batch_size: int = 8,
    hf_token: str | None = None,
) -> dict:
    """Run Engine A: WhisperX large-v3 with current max-quality settings."""
    output_json = scratch / "asr_engine_a.json"
    align_name = ALIGN_MODELS.get(language or "nl", ALIGN_MODELS["nl"]) if language else ALIGN_MODELS["nl"]

    return _run_whisperx_engine(
        audio_path, output_json,
        model_path=model_path,
        language=language,
        prompt=prompt,
        batch_size=batch_size,
        align_model=align_name,
        hf_token=hf_token,
        engine_label="A",
    )


def run_engine_b(
    audio_path: Path,
    scratch: Path,
    *,
    prompt: str | None = None,
    language: str = "nl",
    batch_size: int = 8,
    hf_token: str | None = None,
) -> dict | None:
    """Run Engine B: Dutch fine-tuned whisper-large-v3-dutch.

    Returns None if the model is not available (graceful skip).
    """
    model_path = Path(NL_MODEL_PATH)
    if not model_path.exists():
        return None

    output_json = scratch / "asr_engine_b.json"
    align_name = ALIGN_MODELS.get(language, ALIGN_MODELS["nl"])

    try:
        return _run_whisperx_engine(
            audio_path, output_json,
            model_path=str(model_path),
            language=language,
            prompt=prompt,
            batch_size=batch_size,
            align_model=align_name,
            hf_token=hf_token,
            engine_label="B",
        )
    except Exception as e:
        print(f"Engine B failed, skipping: {e}", file=sys.stderr)
        return None


def run(
    audio_path: Path,
    scratch: Path,
    *,
    prompt: str | None = None,
    language: str | None = None,
    ensemble: bool = False,
    model_path: str = "/opt/whisper-models/large-v3",
    batch_size: int = 8,
    hf_token: str | None = None,
) -> dict:
    """Run the ASR stage. Always runs Engine A; optionally runs Engine B for ensemble."""
    result_a = run_engine_a(
        audio_path, scratch,
        prompt=prompt, language=language,
        model_path=model_path, batch_size=batch_size,
        hf_token=hf_token,
    )

    if ensemble:
        nl_windows = _detect_nl_windows(result_a.get("segments", []))
        if nl_windows:
            # Side effect: writes asr_engine_b.json consumed by downstream ROVER stage
            run_engine_b(
                audio_path, scratch,
                prompt=prompt, language="nl",
                batch_size=batch_size, hf_token=hf_token,
            )

    return result_a


def _detect_nl_windows(segments: list[dict]) -> list[tuple[float, float]]:
    """Determine whether Engine B (Dutch fine-tune) should run.

    WhisperX detects language once for the whole audio and propagates it to all
    segments, so we only get a uniform language. This function returns the full
    audio range when language is nl, or an empty list (skip Engine B) otherwise.

    Per-window language detection is deferred pending the P1-3 subprocess refactor.
    """
    if not segments:
        return []

    detected_lang = segments[0].get("lang", "nl")
    if detected_lang == "nl":
        total_end = max(s.get("end", 0.0) for s in segments)
        return [(0.0, total_end)]
    return []


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ASR engine stage.")
    ap.add_argument("input", type=Path, help="Input audio file (preprocessed WAV).")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--language", default=None)
    ap.add_argument("--prompt-file", type=Path, default=None, help="File containing initial_prompt text.")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--model-path", default="/opt/whisper-models/large-v3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hf-token-file", type=Path, default=None)
    args = ap.parse_args()

    prompt_text = None
    if args.prompt_file:
        prompt_text = args.prompt_file.read_text().strip()

    hf_token = None
    if args.hf_token_file and args.hf_token_file.exists():
        hf_token = args.hf_token_file.read_text().strip()

    result = run(
        args.input, args.scratch,
        prompt=prompt_text,
        language=args.language,
        ensemble=args.ensemble,
        model_path=args.model_path,
        batch_size=args.batch_size,
        hf_token=hf_token,
    )
    print(f"Engine A: {len(result.get('segments', []))} segments")
