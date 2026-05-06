"""Speaker diarization using pyannote, with per-channel and heuristic support.

When stereo-split audio is provided, bypasses pyannote entirely and labels each
channel as a single speaker. Otherwise calls pyannote with the requested model.
Filename heuristics auto-detect 2-speaker phone recordings.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_json


_PHONE_RE = re.compile(r"(?i)(callrec|phonecall|gsm|recording_\d{14})")


@dataclass
class DiarizeConfig:
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None


def _filename_speaker_hint(filename: str) -> int | None:
    """Return 2 if filename looks like a phone call recording."""
    if _PHONE_RE.search(filename):
        return 2
    return None


def _run_pyannote(
    audio_path: Path,
    hf_token: str,
    *,
    model_name: str = "pyannote/speaker-diarization-3.1",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Run pyannote diarization inside Python (designed for Docker)."""
    import os
    import torch

    # Monkeypatch for huggingface_hub API change:
    # huggingface_hub >= 1.0 removed `use_auth_token` in favor of `token`.
    # The Dockerfile pins huggingface_hub==0.30.2 (pre-1.0), so this patch
    # is currently a no-op. It is kept as a safety net: if the pin drifts
    # (e.g. a user runs pip install -U), this patch prevents diarization
    # from breaking. Gate on the actual installed version.
    import huggingface_hub as _hf
    import re as _re
    _ver_match = _re.match(r"(\d+)\.(\d+)", _hf.__version__)
    _hf_version = (int(_ver_match.group(1)), int(_ver_match.group(2))) if _ver_match else (0, 0)
    if _hf_version >= (1, 0):
        _real = _hf.hf_hub_download
        while hasattr(_real, '__wrapped__'):
            _real = _real.__wrapped__

        def _remap_token(*args, **kwargs):
            if 'use_auth_token' in kwargs:
                kwargs['token'] = kwargs.pop('use_auth_token')
            return _real(*args, **kwargs)

        import importlib
        for _mod_name in [
            'pyannote.audio.core.pipeline',
            'pyannote.audio.core.model',
            'pyannote.audio.pipelines.speaker_verification',
        ]:
            try:
                _mod = importlib.import_module(_mod_name)
                if hasattr(_mod, 'hf_hub_download'):
                    _mod.hf_hub_download = _remap_token
            except ImportError:
                pass

    from pyannote.audio import Pipeline

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    pipeline.to(torch.device("cuda"))

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return turns


def _channel_diarize(audio_path: Path, label: str) -> list[dict]:
    """Simple energy-based VAD for a single-channel file, labelling all as one speaker."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    frame_len = int(0.3 * sr)
    energy = waveform.unfold(0, frame_len, frame_len).pow(2).mean(dim=1)
    threshold = energy.quantile(0.15)

    turns = []
    in_speech = False
    start = 0.0
    for i, e in enumerate(energy):
        t = i * 0.3
        if e > threshold and not in_speech:
            start = t
            in_speech = True
        elif e <= threshold and in_speech:
            turns.append({"start": start, "end": t, "speaker": label})
            in_speech = False
    if in_speech:
        turns.append({"start": start, "end": len(waveform) / sr, "speaker": label})

    return turns


def run(
    audio_path: Path,
    output_path: Path,
    *,
    stereo_paths: tuple[Path, Path] | None = None,
    hf_token: str | None = None,
    config: DiarizeConfig | None = None,
    filename_hint: str | None = None,
) -> dict:
    """Run diarization.

    Args:
        audio_path: Main (mono) audio file.
        output_path: Where to write diarize.json.
        stereo_paths: Optional (left, right) paths for per-channel diarization.
        hf_token: HuggingFace token for pyannote.
        config: DiarizeConfig with model and speaker constraints.
        filename_hint: Original filename for heuristic speaker detection.

    Returns:
        Dict with speaker turns.
    """
    if config is None:
        config = DiarizeConfig()

    # Auto-detect speaker count from filename
    if config.min_speakers is None and filename_hint:
        hint = _filename_speaker_hint(filename_hint)
        if hint is not None:
            config.min_speakers = hint
            config.max_speakers = hint

    # Per-channel diarization for stereo split
    if stereo_paths is not None:
        left_path, right_path = stereo_paths
        left_turns = _channel_diarize(left_path, "CH-L")
        right_turns = _channel_diarize(right_path, "CH-R")
        all_turns = sorted(left_turns + right_turns, key=lambda t: t["start"])
        result = {
            "method": "per_channel",
            "speakers": ["CH-L", "CH-R"],
            "turns": all_turns,
        }
        write_json(output_path, result)
        return result

    # Standard pyannote diarization
    if not hf_token:
        result = {
            "method": "none",
            "speakers": [],
            "turns": [],
            "warning": "no HF token, diarization skipped",
        }
        write_json(output_path, result)
        return result

    turns = _run_pyannote(
        audio_path, hf_token,
        model_name=config.model,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
    )
    speakers = sorted(set(t["speaker"] for t in turns))
    result = {
        "method": "pyannote",
        "model": config.model,
        "speakers": speakers,
        "turns": turns,
    }
    write_json(output_path, result)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Speaker diarization stage.")
    ap.add_argument("input", type=Path, help="Preprocessed audio file.")
    ap.add_argument("--output", type=Path, required=True, help="Output diarize.json path.")
    ap.add_argument("--hf-token-file", type=Path, default=None)
    ap.add_argument("--left", type=Path, default=None, help="Left channel for stereo split.")
    ap.add_argument("--right", type=Path, default=None, help="Right channel for stereo split.")
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--speakers", type=int, default=None, help="Pin speaker count.")
    ap.add_argument("--filename-hint", default=None, help="Original filename for heuristics.")
    args = ap.parse_args()

    hf_token = None
    if args.hf_token_file and args.hf_token_file.exists():
        hf_token = args.hf_token_file.read_text().strip()

    stereo = None
    if args.left and args.right:
        stereo = (args.left, args.right)

    cfg = DiarizeConfig(
        model=args.model,
        min_speakers=args.speakers,
        max_speakers=args.speakers,
    )
    result = run(
        args.input, args.output,
        stereo_paths=stereo,
        hf_token=hf_token,
        config=cfg,
        filename_hint=args.filename_hint,
    )
    n_speakers = len(result.get("speakers", []))
    n_turns = len(result.get("turns", []))
    print(f"Diarization: {n_speakers} speakers, {n_turns} turns via {result['method']}")
