"""Render transcript .txt files from ROVER + diarization + (optional) post-correction.

Produces two outputs:
  - audio.txt — verbatim transcript with speaker labels
  - audio.cleaned.txt — post-corrected transcript with speaker labels
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign speaker labels to segments based on diarization turns."""
    if not turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 1.0)
        seg_center = (seg_start + seg_end) / 2

        best_turn = None
        best_overlap = -1.0
        for turn in turns:
            turn_start = turn["start"]
            turn_end = turn["end"]
            # Compute overlap
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        seg["speaker"] = best_turn["speaker"] if best_turn else turns[0]["speaker"]

    return segments


def render_text(segments: list[dict], turns: list[dict]) -> str:
    """Render segments into the standard transcript format with speaker labels."""
    if not segments:
        return ""

    segments = _assign_speakers(segments, turns)

    lines: list[str] = []
    current_speaker = None

    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        text = seg.get("text", "").strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker

        lines.append(f"  {text}")

    return "\n".join(lines) + "\n"


def run(
    rover_path: Path,
    diarize_path: Path,
    cleaned_path: Path | None,
    verbatim_out: Path,
    cleaned_out: Path | None,
) -> None:
    """Render both transcript files."""
    rover = json.loads(rover_path.read_text()) if rover_path.exists() else {"segments": []}
    diarize = json.loads(diarize_path.read_text()) if diarize_path.exists() else {"turns": []}
    cleaned = json.loads(cleaned_path.read_text()) if cleaned_path and cleaned_path.exists() else None

    segments = rover.get("segments", [])
    turns = diarize.get("turns", [])

    # Verbatim output
    verbatim_text = render_text(segments, turns)
    verbatim_out.write_text(verbatim_text, encoding="utf-8")
    try:
        os.chmod(verbatim_out, 0o666)
    except OSError:
        pass

    # Cleaned output (if post-correction ran)
    if cleaned_out and cleaned:
        cleaned_segments = cleaned.get("segments", segments)
        cleaned_text = render_text(cleaned_segments, turns)
        cleaned_out.write_text(cleaned_text, encoding="utf-8")
        try:
            os.chmod(cleaned_out, 0o666)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render transcript text files.")
    ap.add_argument("--rover", type=Path, required=True)
    ap.add_argument("--diarize", type=Path, required=True)
    ap.add_argument("--cleaned", type=Path, default=None)
    ap.add_argument("--verbatim-out", type=Path, required=True)
    ap.add_argument("--cleaned-out", type=Path, default=None)
    args = ap.parse_args()

    run(args.rover, args.diarize, args.cleaned, args.verbatim_out, args.cleaned_out)
    print(f"Wrote {args.verbatim_out}")
    if args.cleaned_out:
        print(f"Wrote {args.cleaned_out}")
