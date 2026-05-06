"""Build a sanitised Whisper `initial_prompt` from filename, glossary, and LLM output.

The legacy bash pipeline piped raw `ollama` stdout (qwen3 thinking-mode
verbiage and terminal escape codes) directly into Whisper's prompt, which
made transcription quality *worse*. This module is the cure: every string
that touches `initial_prompt` must pass through `sanitise_vocab()`.

Resulting prompt = user context + filename priors + LLM-extracted vocab +
glossary canonical forms, dedup-preserved-order, truncated to ~224 BPE
tokens (Whisper's hard ceiling for `initial_prompt`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .glossary import Glossary

# Whisper's prompt context window is 224 tokens. Use a conservative char proxy
# (~4 chars / token) plus a hard char ceiling so we never silently overflow.
MAX_PROMPT_CHARS = 1500
MAX_PROMPT_TOKENS = 80  # post-sanitise vocab tokens, before joining with context
PROMPT_TOKEN_BUDGET = 224  # whisper-side ceiling, used for the final truncation

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_BARE_CSI_RE = re.compile(r"(?<![A-Za-z])\[[0-9;?]{0,6}[A-HJKSTfhlmnsu]")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_DOTS_RE = re.compile(
    r"^\s*Thinking\.{2,}.*?(?:^\s*\.{2,}\s*done thinking\.?\s*$|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_CHATML_TAG_RE = re.compile(r"<\|im_(?:start|end)\|>(?:\s*\w+)?")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FILENAME_TOKEN_SPLIT_RE = re.compile(r"[\s_\-+.]+")
_PHONE_RE = re.compile(r"^\+?\d[\d\-\s]{4,}$")
_DATESTAMP_RE = re.compile(r"^\d{6,14}$")


@dataclass
class PromptBuildResult:
    prompt: str
    vocab_tokens: list[str]
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def strip_thinking(text: str) -> str:
    """Remove qwen3 / ChatML reasoning residue."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINKING_DOTS_RE.sub("", text)
    text = _CHATML_TAG_RE.sub("", text)
    return text


def strip_ansi(text: str) -> str:
    """Remove proper CSI sequences and bare-bracket terminal artefacts."""
    text = _ANSI_RE.sub("", text)
    text = _BARE_CSI_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    return text


def normalise_text(text: str) -> str:
    """NFC-normalise and squash whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitise_vocab(raw: str) -> tuple[list[str], list[str]]:
    """Turn arbitrary LLM output into a clean vocab token list.

    Returns (accepted_tokens, warnings). Tokens are deduped (case-insensitive,
    preserves first-seen casing). Anything that looks like a sentence is
    rejected — vocab prompts must be terse.
    """
    warnings: list[str] = []
    text = strip_thinking(raw)
    text = strip_ansi(text)
    text = normalise_text(text)
    if not text:
        warnings.append("vocab response was empty after sanitisation")
        return [], warnings
    if len(text) > MAX_PROMPT_CHARS * 4:
        warnings.append(
            f"vocab response was {len(text)} chars; truncating before tokenisation"
        )
        text = text[: MAX_PROMPT_CHARS * 4]

    parts: list[str] = []
    for chunk in re.split(r"[,\n;|]+", text):
        token = chunk.strip(" \t-•*·–—\"'`()[]{}<>")
        if not token:
            continue
        if len(token.split()) > 4:
            warnings.append(f"dropped long phrase: {token[:60]!r}")
            continue
        if len(token) > 60:
            warnings.append(f"dropped overlong token: {token[:60]!r}…")
            continue
        if token.startswith(("http://", "https://", "www.")):
            continue
        if not re.search(r"[A-Za-zÀ-ɏ]", token):
            continue
        parts.append(token)

    seen: set[str] = set()
    accepted: list[str] = []
    for token in parts:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted.append(token)
        if len(accepted) >= MAX_PROMPT_TOKENS:
            break
    return accepted, warnings


def filename_priors(audio_path: str | Path) -> list[str]:
    """Pull plausible proper-noun candidates out of the file basename."""
    stem = Path(audio_path).stem
    raw_tokens = [t for t in _FILENAME_TOKEN_SPLIT_RE.split(stem) if t]
    out: list[str] = []
    for tok in raw_tokens:
        if len(tok) < 3:
            continue
        if _PHONE_RE.match(tok) or _DATESTAMP_RE.match(tok):
            continue
        if not re.search(r"[A-Za-z]", tok):
            continue
        out.append(tok if any(c.isupper() for c in tok) else tok.title())
    seen: set[str] = set()
    deduped: list[str] = []
    for tok in out:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tok)
    return deduped


def _enforce_token_budget(tokens: list[str], budget: int = PROMPT_TOKEN_BUDGET) -> list[str]:
    # Approximate Whisper BPE: 1 token ≈ 4 chars or 1 short word.
    out: list[str] = []
    char_budget = budget * 4
    used = 0
    for tok in tokens:
        cost = len(tok) + 2  # `, `
        if used + cost > char_budget:
            break
        out.append(tok)
        used += cost
    return out


def build_prompt(
    *,
    audio_path: str | Path,
    glossary: Glossary | None = None,
    llm_vocab: str | None = None,
    user_context: str | None = None,
) -> PromptBuildResult:
    """Compose the final initial_prompt string.

    Order: user context → filename priors → LLM vocab → glossary canonicals.
    Earlier inputs survive truncation; the glossary tail is best-effort.
    """
    warnings: list[str] = []
    rejected: list[str] = []

    context_clean = ""
    if user_context:
        context_clean = normalise_text(strip_ansi(strip_thinking(user_context)))
        if len(context_clean) > MAX_PROMPT_CHARS:
            warnings.append("user context truncated to MAX_PROMPT_CHARS")
            context_clean = context_clean[:MAX_PROMPT_CHARS].rstrip()

    priors = filename_priors(audio_path)

    llm_tokens: list[str] = []
    if llm_vocab:
        llm_tokens, vocab_warnings = sanitise_vocab(llm_vocab)
        warnings.extend(vocab_warnings)
        if not llm_tokens and llm_vocab.strip():
            rejected.append("llm_vocab")

    glossary_terms: list[str] = []
    if glossary is not None:
        glossary_terms = glossary.canonical_terms()

    seen: set[str] = set()
    ordered_vocab: list[str] = []
    for tok in [*priors, *llm_tokens, *glossary_terms]:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_vocab.append(tok)

    bounded_vocab = _enforce_token_budget(ordered_vocab)

    pieces: list[str] = []
    if context_clean:
        pieces.append(context_clean)
    if bounded_vocab:
        pieces.append("Vocabulary: " + ", ".join(bounded_vocab) + ".")
    prompt = " ".join(pieces).strip()

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS].rstrip()
        warnings.append("final prompt truncated to MAX_PROMPT_CHARS")

    return PromptBuildResult(
        prompt=prompt,
        vocab_tokens=bounded_vocab,
        rejected=rejected,
        warnings=warnings,
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Build / inspect Whisper initial_prompt.")
    ap.add_argument("audio", help="Audio file path (used for filename priors).")
    ap.add_argument("--context", help="Free-text user context.")
    ap.add_argument("--vocab-file", help="Path to raw LLM vocab response (will be sanitised).")
    ap.add_argument("--glossary", help="Override glossary path.")
    args = ap.parse_args()

    raw_vocab = None
    if args.vocab_file:
        raw_vocab = Path(args.vocab_file).read_text(encoding="utf-8", errors="replace")

    g = Glossary.resolve(args.glossary)
    res = build_prompt(
        audio_path=args.audio,
        glossary=g,
        llm_vocab=raw_vocab,
        user_context=args.context,
    )
    json.dump(
        {
            "prompt": res.prompt,
            "prompt_chars": len(res.prompt),
            "vocab_tokens": res.vocab_tokens,
            "rejected": res.rejected,
            "warnings": res.warnings,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
