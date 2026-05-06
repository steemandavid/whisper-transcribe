"""Audio preprocessing: loudness normalisation, channel split, denoise, bandwidth extension.

Produces a canonical 16 kHz mono float32 WAV plus optional stereo-split L/R
files when the input has low-correlated stereo channels (e.g. phone calls).
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PreprocessResult:
    mono_path: Path
    left_path: Path | None
    right_path: Path | None
    sr: int = 16000
    used_loudnorm: bool = False
    used_highpass: bool = False
    used_audiosr: bool = False
    used_deepfilternet: bool = False
    stereo_split: bool = False
    cross_correlation: float | None = None
    notes: list[str] = field(default_factory=list)


def _ffprobe(audio: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-show_streams", "-of", "json", str(audio)],
        capture_output=True, text=True,
    )
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no streams in {audio}")
    return streams[0]


def _cross_correlation(audio: Path, duration_s: float) -> float:
    """Read stereo audio, compute Pearson correlation between channels."""
    import soundfile as sf
    data, sr = sf.read(str(audio), dtype="float32")
    if data.ndim < 2:
        return 1.0
    # Use a 30-s window (or full file if shorter)
    window = min(int(30 * sr), len(data))
    left = data[:window, 0]
    right = data[:window, 1]
    # Pearson correlation
    lf = left - left.mean()
    rf = right - right.mean()
    denom = math.sqrt((lf @ lf) * (rf @ rf))
    if denom < 1e-12:
        return 1.0
    return float((lf @ rf) / denom)


def _loudnorm_two_pass(input_wav: Path, output_wav: Path) -> bool:
    """Apply ffmpeg loudnorm two-pass. Returns True if normalisation was applied."""
    target_i = -16.0
    target_tp = -1.5
    target_lra = 11.0

    # First pass: measure
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # ffmpeg prints the JSON to stderr
    stderr = r.stderr
    # Find the JSON block
    idx = stderr.rfind("{")
    if idx < 0:
        return False
    try:
        meas = json.loads(stderr[idx:])
    except json.JSONDecodeError:
        return False

    input_i = float(meas.get("input_i", target_i))
    if abs(input_i - target_i) < 2.0:
        return False

    # Second pass with measured values
    measured = {
        "measured_I": meas["input_i"],
        "measured_TP": meas["input_tp"],
        "measured_LRA": meas["input_lra"],
        "measured_thresh": meas["input_thresh"],
        "offset": meas["target_offset"],
    }
    af = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":measured_I={measured['measured_I']}"
        f":measured_TP={measured['measured_TP']}"
        f":measured_LRA={measured['measured_LRA']}"
        f":measured_thresh={measured['measured_thresh']}"
        f":offset={measured['offset']}"
        f":linear=true:print_format=summary"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_wav), "-af", af, str(output_wav)],
        capture_output=True, text=True, check=True,
    )
    return True


def _is_narrowband(audio: Path) -> bool:
    """Check if audio energy above 3.4 kHz is below -50 dBFS."""
    import soundfile as sf
    data, sr = sf.read(str(audio), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < 1024:
        return False
    n_fft = 1024
    spec = np.fft.rfft(data[:n_fft])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    magnitudes = np.abs(spec)
    ref = magnitudes.max()
    if ref < 1e-12:
        return False
    db = 20 * np.log10(magnitudes / ref + 1e-12)
    high_mask = freqs > 3400
    if not high_mask.any():
        return False
    mean_high = float(db[high_mask].mean())
    return mean_high < -50


def _apply_audiosr(input_wav: Path, output_wav: Path) -> bool:
    """Run AudioSR bandwidth extension. Returns True if applied."""
    try:
        from audiosr import AudioSR, build_model, super_resolution
        # audiosr API varies; try the simple path
        sr_model = build_model(device="cuda")
        super_resolution(sr_model, str(input_wav), str(output_wav))
        return output_wav.exists()
    except Exception:
        return False


def _apply_deepfilternet(input_wav: Path, output_wav: Path) -> bool:
    """Run DeepFilterNet3 denoising. Returns True if applied."""
    try:
        import df  # deepfilternet
        from df.enhance import enhance, init_df
        model, df_state, _ = init_df()
        import soundfile as sf
        audio, sr = sf.read(str(input_wav), dtype="float32")
        enhanced = enhance(model, df_state, audio)
        sf.write(str(output_wav), enhanced, sr)
        return True
    except Exception:
        return False


def run(input_audio: Path, scratch: Path, *, denoise: bool = False, enhance: bool = False) -> PreprocessResult:
    """Run the preprocessing pipeline.

    Args:
        input_audio: Source audio file path.
        scratch: Scratch directory for intermediate files.
        denoise: Enable DeepFilterNet3 denoising.
        enhance: Enable full enhancement (loudnorm + highpass + bandwidth extension).

    Returns:
        PreprocessResult with paths and metadata.
    """
    result = PreprocessResult(
        mono_path=scratch / "preprocessed.wav",
        left_path=None,
        right_path=None,
    )

    probe = _ffprobe(input_audio)
    channels = int(probe.get("channels", 1))
    source_sr = int(probe.get("sample_rate", 16000))

    # Work in a temp file so we chain ffmpeg operations
    current = input_audio
    tmp_files: list[Path] = []

    def _tmp(suffix: str = ".wav") -> Path:
        p = scratch / f"_tmp_preprocess_{len(tmp_files)}{suffix}"
        tmp_files.append(p)
        return p

    # Convert to WAV upfront if not already WAV (soundfile can't read m4a/mp3/etc.)
    if input_audio.suffix.lower() not in (".wav", ".wave"):
        wav_tmp = _tmp()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_audio), "-ac", str(channels),
             "-c:a", "pcm_s16le", str(wav_tmp)],
            capture_output=True, check=True,
        )
        current = wav_tmp

    # --- Channel split for low-correlated stereo ---
    if channels == 2:
        try:
            corr = _cross_correlation(current, 0)
            result.cross_correlation = round(corr, 3)
        except Exception:
            corr = 1.0
            result.cross_correlation = None

        if corr < 0.6:
            result.stereo_split = True
            result.left_path = scratch / "preprocessed_L.wav"
            result.right_path = scratch / "preprocessed_R.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(current),
                 "-af", "pan=mono|c0=c0",
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                 str(result.left_path)],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(current),
                 "-af", "pan=mono|c0=c1",
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                 str(result.right_path)],
                capture_output=True, check=True,
            )
            # Mono = downmix for the main pipeline
            downmixed = _tmp()
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(current),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(downmixed)],
                capture_output=True, check=True,
            )
            current = downmixed
            result.notes.append(f"stereo split (cross-corr={corr:.3f})")
        else:
            # Downmix to mono
            mono = _tmp()
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(current),
                 "-ac", "1", str(mono)],
                capture_output=True, check=True,
            )
            current = mono
    elif channels > 2:
        mono = _tmp()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(current),
             "-ac", "1", str(mono)],
            capture_output=True, check=True,
        )
        current = mono

    # --- Bandwidth extension (narrowband only) ---
    if enhance and _is_narrowband(current):
        bw_out = _tmp()
        if _apply_audiosr(current, bw_out):
            current = bw_out
            result.used_audiosr = True
            result.notes.append("AudioSR bandwidth extension applied")
        else:
            result.notes.append("AudioSR skipped (unavailable)")

    # --- Loudness normalisation ---
    if enhance:
        loud_out = _tmp()
        if _loudnorm_two_pass(current, loud_out):
            current = loud_out
            result.used_loudnorm = True
            result.notes.append("loudnorm two-pass applied")
        else:
            result.notes.append("loudnorm skipped (already near target or first-pass failed)")

    # --- High-pass 80 Hz (only for SR >= 16 kHz) ---
    if enhance and source_sr >= 16000:
        hp_out = _tmp()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(current),
             "-af", "highpass=f=80", str(hp_out)],
            capture_output=True, check=True,
        )
        current = hp_out
        result.used_highpass = True
        result.notes.append("high-pass 80 Hz applied")

    # --- Denoise ---
    if denoise:
        dn_out = _tmp()
        if _apply_deepfilternet(current, dn_out):
            current = dn_out
            result.used_deepfilternet = True
            result.notes.append("DeepFilterNet3 denoising applied")
        else:
            result.notes.append("DeepFilterNet3 skipped (unavailable)")

    # --- Final resample to 16 kHz mono float32 ---
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(current),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         str(result.mono_path)],
        capture_output=True, check=True,
    )

    # Write metadata
    meta = {
        "input": str(input_audio),
        "channels": channels,
        "source_sample_rate": source_sr,
        "cross_correlation": result.cross_correlation,
        "stereo_split": result.stereo_split,
        "used_loudnorm": result.used_loudnorm,
        "used_highpass": result.used_highpass,
        "used_audiosr": result.used_audiosr,
        "used_deepfilternet": result.used_deepfilternet,
        "notes": result.notes,
    }
    (scratch / "preprocess.json").write_text(json.dumps(meta, indent=2))

    # Cleanup temp files
    for p in tmp_files:
        p.unlink(missing_ok=True)

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Audio preprocessing pipeline stage.")
    ap.add_argument("input", type=Path, help="Input audio file.")
    ap.add_argument("--scratch", type=Path, required=True, help="Scratch directory.")
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--enhance", action="store_true")
    args = ap.parse_args()

    res = run(args.input, args.scratch, denoise=args.denoise, enhance=args.enhance)
    print(json.dumps({
        "mono": str(res.mono_path),
        "left": str(res.left_path) if res.left_path else None,
        "right": str(res.right_path) if res.right_path else None,
        "stereo_split": res.stereo_split,
        "notes": res.notes,
    }, indent=2))
