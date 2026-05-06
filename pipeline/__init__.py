"""Whisper transcription pipeline.

Stages communicate via JSON artifacts written to a per-run scratch dir.
Each stage is invokable as `python -m pipeline.<stage>`.
"""

__version__ = "2.0.0"
