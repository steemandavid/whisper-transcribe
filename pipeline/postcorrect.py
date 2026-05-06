"""LLM post-correction: fix phonetic mishearings in the ROVER output.

Two backends:
  - OllamaCorrector: local LLM via Ollama HTTP API
  - ZaiCorrector: cloud GLM model via Z.ai OpenAI-compatible API with auto-versioning

Applies only high-confidence corrections (>= 0.8) and rejects segments with
> 35% token churn via a diff guard.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Protocol

from .artifacts import write_json


# --- Diff guard ---

def _token_edit_ratio(original: str, corrected: str) -> float:
    """Token-level edit distance as a fraction of max(len(a), len(b))."""
    a = original.split()
    b = corrected.split()
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return dist / max(la, lb, 1)


# --- GLM auto-version resolver ---

_GLM_CACHE = Path.home() / ".config" / "whisper" / ".glm-resolved"
_GLM_FALLBACK = "glm-5.1"
_GLM_EXCLUDE = {"coder", "embedding", "vision"}
_CACHE_TTL_S = 86400  # 24 hours


def resolve_glm_model(client, *, refresh: bool = False, cache_path: Path = _GLM_CACHE) -> str:
    """Resolve the latest GLM model id from the Z.ai API, with 24h caching."""
    if not refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            resolved_at = cached.get("resolved_at", "")
            if resolved_at:
                age = time.time() - _parse_iso(resolved_at)
                if age < _CACHE_TTL_S:
                    return cached["id"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    try:
        models = client.models.list()
        best: tuple[int, int] = (0, 0)
        best_id = _GLM_FALLBACK
        for m in models:
            mid = m.id if hasattr(m, "id") else str(m)
            if not mid.startswith("glm-"):
                continue
            lower = mid.lower()
            if any(exc in lower for exc in _GLM_EXCLUDE):
                continue
            # Parse glm-X.Y or glm-X
            match = re.match(r"glm-(\d+)(?:\.(\d+))?", mid)
            if not match:
                continue
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            version = (major, minor)
            if version > best:
                best = version
                best_id = mid
        resolved = best_id
    except Exception:
        resolved = _GLM_FALLBACK

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "id": resolved,
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    return resolved


def _parse_iso(s: str) -> float:
    from datetime import datetime, timezone
    s = s.rstrip("Z")
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return dt.timestamp()


# --- Corrector protocol ---

class PostCorrector(Protocol):
    def correct(self, segments: list[dict], system_prompt: str, language: str) -> list[dict]:
        ...


# --- Ollama backend ---

class OllamaCorrector:
    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434"):
        self.model = model or self._auto_select(base_url)
        self.base_url = base_url

    def _auto_select(self, base_url: str) -> str:
        import urllib.request
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            # Prefer instruct variants, penalise coder/qwen3, prefer >=14B
            scored = []
            for name in models:
                s = 1.0
                lower = name.lower()
                if "instruct" in lower:
                    s *= 1.3
                if "coder" in lower:
                    s *= 0.7
                if "qwen3" in lower:
                    s *= 0.5
                # Extract parameter count from tags like :32b, :14b, :7b
                import re as _re
                size_m = _re.search(r":(\d+)b", lower)
                if size_m:
                    params = int(size_m.group(1))
                    if params >= 32:
                        s *= 1.5
                    elif params >= 14:
                        s *= 1.2
                    elif params < 8:
                        s *= 0.6
                scored.append((name, s))
            if scored:
                scored.sort(key=lambda x: -x[1])
                return scored[0][0]
        except Exception:
            pass
        return "qwen2.5:32b-instruct-q4_K_M"

    def correct(self, segments: list[dict], system_prompt: str, language: str) -> list[dict]:
        import urllib.request
        batch_text = _format_batch(segments)
        user_prompt = f"Transcript segments to correct:\n{batch_text}"

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\n\nRespond only in valid JSON. No thinking blocks."},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode()

        def _call():
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())

        data = _retry(_call)
        content = data.get("message", {}).get("content", "")
        return _parse_corrections(content, segments)


# --- Z.ai GLM backend ---

class ZaiCorrector:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI
        self.api_key = api_key or os.environ.get("ZAI_API_KEY", "")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.z.ai/api/paas/v4/",
        )
        self.model = model or resolve_glm_model(self.client)
        self.last_usage: dict | None = None  # accumulated across batches

    def correct(self, segments: list[dict], system_prompt: str, language: str) -> list[dict]:
        batch_text = _format_batch(segments)
        user_prompt = f"Transcript segments to correct:\n{batch_text}"

        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n\nRespond only in valid JSON array. No markdown fences.",
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
            )

        response = _retry(_call)
        # Track token usage for sidecar output
        if hasattr(response, 'usage') and response.usage:
            batch_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            if self.last_usage is None:
                self.last_usage = {"model": self.model, "batches": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}
            self.last_usage["batches"] += 1
            self.last_usage["total_prompt_tokens"] += batch_usage["prompt_tokens"]
            self.last_usage["total_completion_tokens"] += batch_usage["completion_tokens"]
            print(f"Z.ai ({self.model}): {batch_usage['prompt_tokens']} prompt + {batch_usage['completion_tokens']} completion tokens (batch {self.last_usage['batches']})", file=sys.stderr)
        content = response.choices[0].message.content or ""
        return _parse_corrections(content, segments)


# --- Shared helpers ---

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2  # seconds


def _retry(fn, *args, **kwargs):
    """Retry with exponential backoff (3 attempts, 2× base) on transient errors only."""
    import socket
    import urllib.error
    transient = (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError)
    # OpenAI transient errors (imported lazily since openai may not be installed)
    try:
        from openai import (
            RateLimitError as _RateLimit,
            APIConnectionError as _APIConn,
            APITimeoutError as _APITimeout,
            InternalServerError as _Internal,
        )
        transient = transient + (_RateLimit, _APIConn, _APITimeout, _Internal)
    except ImportError:
        pass

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except transient as e:
            if attempt == _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  retry {attempt}/{_MAX_RETRIES} after {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)


_SYSTEM_PROMPT_TEMPLATE = """You are correcting speech-to-text transcription errors.

Language: {language}
Rules:
- Fix ONLY obvious phonetic mishearings (Dutch/Flemish dialect, brand names, proper nouns).
- Words marked with [?] are uncertain — prioritise correcting those.
- Use the glossary below for correct forms.
- Do NOT paraphrase, add, or remove content.
- Do NOT change punctuation outside of fixed words.
- Preserve speaker labels and [?] markers when you are not confident.

Glossary:
{glossary}

Output strict JSON array:
[{{"seg_id": <int>, "original": "<text>", "corrected": "<text>", "changed_words": [{{"idx": <int>, "before": "<word>", "after": "<word>", "confidence": <float>}}]}}]
"""

_BATCH_SIZE = 40
_CONFIDENCE_THRESHOLD = 0.8
_MAX_EDIT_RATIO = 0.35


def _format_batch(segments: list[dict]) -> str:
    """Format segments for the LLM prompt."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg['id']}] {seg['text']}")
    return "\n".join(lines)


def _parse_corrections(raw: str, original_segments: list[dict]) -> list[dict]:
    """Parse LLM response, apply confidence filter and diff guard."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        corrections = json.loads(text)
    except json.JSONDecodeError:
        return original_segments

    if not isinstance(corrections, list):
        return original_segments

    seg_map = {seg["id"]: seg for seg in original_segments}
    results = []

    for corr in corrections:
        seg_id = corr.get("seg_id")
        if seg_id not in seg_map:
            continue
        original = seg_map[seg_id]
        corrected_text = corr.get("corrected", original["text"])

        # Filter changed_words by confidence
        changed = corr.get("changed_words", [])
        high_conf = [w for w in changed if w.get("confidence", 0) >= _CONFIDENCE_THRESHOLD]

        if not high_conf:
            results.append(dict(original))
            continue

        # Diff guard
        edit_ratio = _token_edit_ratio(original["text"], corrected_text)
        if edit_ratio > _MAX_EDIT_RATIO:
            results.append({**original, "text": original["text"] + " [!unverified]"})
            continue

        results.append({
            **original,
            "text": corrected_text,
            "corrections": high_conf,
            "edit_ratio": round(edit_ratio, 3),
        })

    # Fill in any segments the LLM skipped
    for seg in original_segments:
        if not any(r.get("id") == seg["id"] for r in results):
            results.append(dict(seg))

    results.sort(key=lambda s: s.get("id", 0))
    return results


def run(
    rover_path: Path,
    output_path: Path,
    *,
    glossary_text: str = "",
    language: str = "nl",
    backend: str = "ollama",
    ollama_model: str | None = None,
    zai_api_key: str | None = None,
    zai_model: str | None = None,
    refresh_model: bool = False,
) -> dict:
    """Run the post-correction stage.

    Args:
        rover_path: Path to rover.json.
        output_path: Path to write cleaned.json.
        glossary_text: Glossary as text for the system prompt.
        language: Detected language.
        backend: 'ollama' or 'zai'.
        ollama_model: Override Ollama model name.
        zai_api_key: Z.ai API key.
        zai_model: Override Z.ai model name.
        refresh_model: Force GLM model re-resolution.

    Returns:
        Dict with corrected segments.
    """
    from .artifacts import read_json

    rover = read_json(rover_path)
    segments = rover.get("segments", [])
    if not segments:
        write_json(output_path, {"segments": [], "method": "none"})
        return {"segments": [], "method": "none"}

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        language=language,
        glossary=glossary_text or "(none)",
    )

    corrector: PostCorrector
    if backend == "zai":
        corrector = ZaiCorrector(api_key=zai_api_key, model=zai_model)
        if refresh_model and hasattr(corrector, "client"):
            corrector.model = resolve_glm_model(corrector.client, refresh=True)
    else:
        corrector = OllamaCorrector(model=ollama_model)

    # Process in batches
    all_corrected: list[dict] = []
    for i in range(0, len(segments), _BATCH_SIZE):
        batch = segments[i : i + _BATCH_SIZE]
        try:
            corrected = corrector.correct(batch, system_prompt, language)
            all_corrected.extend(corrected)
        except Exception as e:
            print(f"Post-correct batch {i // _BATCH_SIZE}: {type(e).__name__}: {e}", file=sys.stderr)
            all_corrected.extend(batch)

    result = {
        "method": backend,
        "model": getattr(corrector, "model", "unknown"),
        "segments": all_corrected,
    }
    # Write usage sidecar for Z.ai backend
    if backend == "zai" and hasattr(corrector, "last_usage") and corrector.last_usage:
        result["usage"] = corrector.last_usage
        usage_path = output_path.parent / (output_path.stem + "_usage.json")
        write_json(usage_path, corrector.last_usage)
    write_json(output_path, result)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM post-correction stage.")
    ap.add_argument("--rover", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--glossary-file", type=Path, default=None)
    ap.add_argument("--language", default="nl")
    ap.add_argument("--backend", choices=["ollama", "zai"], default="ollama")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--zai-api-key", default=None)
    ap.add_argument("--zai-model", default=None)
    ap.add_argument("--refresh-model", action="store_true")
    args = ap.parse_args()

    glossary_text = ""
    if args.glossary_file and args.glossary_file.exists():
        from .glossary import Glossary
        g = Glossary.load(args.glossary_file)
        glossary_text = g.to_prompt_block()

    result = run(
        args.rover, args.output,
        glossary_text=glossary_text,
        language=args.language,
        backend=args.backend,
        ollama_model=args.ollama_model,
        zai_api_key=args.zai_api_key,
        zai_model=args.zai_model,
        refresh_model=args.refresh_model,
    )
    n_corr = sum(1 for s in result["segments"] if "corrections" in s)
    print(f"Post-correct: {len(result['segments'])} segments, {n_corr} corrected ({result['method']})")
