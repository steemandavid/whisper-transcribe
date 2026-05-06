"""Shared artifact paths and JSON schema helpers used by every pipeline stage.

Each run creates a scratch directory at $INPUT_DIR/.whisper-run-<pid>/. All
intermediate artifacts (preprocessed audio, ASR JSON, ROVER JSON, post-correction
JSON) live there until the orchestrator either keeps them (--keep-intermediates)
or deletes them on exit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


SCRATCH_PREFIX = ".whisper-run-"


@dataclass(frozen=True)
class RunPaths:
    input_audio: Path
    input_dir: Path
    input_base: str
    scratch: Path

    @property
    def preprocessed(self) -> Path:
        return self.scratch / "preprocessed.wav"

    @property
    def preprocessed_left(self) -> Path:
        return self.scratch / "preprocessed_L.wav"

    @property
    def preprocessed_right(self) -> Path:
        return self.scratch / "preprocessed_R.wav"

    @property
    def preprocess_meta(self) -> Path:
        return self.scratch / "preprocess.json"

    @property
    def engine_a_json(self) -> Path:
        return self.scratch / "asr_engine_a.json"

    @property
    def engine_b_json(self) -> Path:
        return self.scratch / "asr_engine_b.json"

    @property
    def rover_json(self) -> Path:
        return self.scratch / "rover.json"

    @property
    def diarize_json(self) -> Path:
        return self.scratch / "diarize.json"

    @property
    def merged_json(self) -> Path:
        return self.scratch / "merged.json"

    @property
    def cleaned_json(self) -> Path:
        return self.scratch / "cleaned.json"

    @property
    def verbatim_txt(self) -> Path:
        return self.input_dir / f"{self.input_base}.txt"

    @property
    def cleaned_txt(self) -> Path:
        return self.input_dir / f"{self.input_base}.cleaned.txt"


def resolve_run_paths(input_audio: str | os.PathLike, scratch: str | os.PathLike | None = None) -> RunPaths:
    audio = Path(input_audio).resolve()
    input_dir = audio.parent
    input_base = audio.stem
    if scratch is None:
        scratch = input_dir / f"{SCRATCH_PREFIX}{os.getpid()}"
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        input_audio=audio,
        input_dir=input_dir,
        input_base=input_base,
        scratch=scratch,
    )


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def read_json(path: Path):
    return json.loads(path.read_text())
