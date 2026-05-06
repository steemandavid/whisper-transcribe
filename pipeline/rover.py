"""ROVER reconciliation: confidence-weighted majority vote between Engine A and B.

Aligns words from two ASR engines by timestamp proximity, then picks the best
candidate per slot using confidence scores weighted by glossary membership.
Emits [?] markers on low-margin decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_json
from .glossary import Glossary


@dataclass
class RoverConfig:
    anchor_window_s: float = 0.150
    margin_threshold: float = 0.15


def _levenshtein_ratio(a: str, b: str) -> float:
    """Character-level similarity ratio (0..1)."""
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if abs(la - lb) > max(la, lb) * 0.5:
        return 0.0
    # Quick check for equality
    if a == b:
        return 1.0
    # Standard Levenshtein
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _flatten_words(engine_data: dict) -> list[dict]:
    """Flatten segments to per-word list with timing and confidence."""
    words = []
    for seg in engine_data.get("segments", []):
        for w in seg.get("words", []):
            if not w.get("word", "").strip():
                continue
            words.append({
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
                "center": (w.get("start", 0.0) + w.get("end", 0.0)) / 2,
                "word": w["word"].strip(),
                "prob": w.get("prob", 0.0),
                "logprob": seg.get("avg_logprob", 0.0),
            })
    return words


def _build_anchor_clusters(
    words_a: list[dict],
    words_b: list[dict],
    window: float,
) -> list[list[dict]]:
    """Greedy left-to-right alignment: pair words whose centres are within window."""
    clusters: list[list[dict]] = []
    used_b: set[int] = set()

    for wa in words_a:
        best_idx = -1
        best_dist = window + 1
        best_ratio = 0.0
        for j, wb in enumerate(words_b):
            if j in used_b:
                continue
            dist = abs(wa["center"] - wb["center"])
            if dist > window:
                continue
            ratio = _levenshtein_ratio(wa["word"].lower(), wb["word"].lower())
            if dist < best_dist or (dist == best_dist and ratio > best_ratio):
                best_dist = dist
                best_idx = j
                best_ratio = ratio

        cluster = [{"engine": "A", **wa}]
        if best_idx >= 0 and best_ratio >= 0.6:
            cluster.append({"engine": "B", **words_b[best_idx]})
            used_b.add(best_idx)
        clusters.append(cluster)

    # Add unmatched B words as singletons
    for j, wb in enumerate(words_b):
        if j not in used_b:
            clusters.append([{"engine": "B", **wb}])

    return clusters


def _score_word(w: dict, glossary: Glossary | None) -> tuple[float, float, bool, str]:
    """Score a word candidate. Returns (score, logprob, is_glossary, engine).

    Tie-break cascade per spec:
      1. Higher base score (prob or normalised logprob + glossary bonus)
      2. Higher mean log-prob
      3. Presence in glossary
      4. Engine A as default
    """
    prob = w.get("prob", 0.0)
    if prob <= 0:
        prob = min(1.0, max(0.0, 0.5 + w.get("logprob", 0.0) / 2.0))
    word_text = w["word"].rstrip("[?]").strip()
    is_glossary = False
    if glossary and glossary.is_canonical(word_text):
        # FSD §2 stage 2: "Multiply by glossary weight."
        # The cascade's step 3 (glossary membership) is correctly redundant here:
        # the multiplicative score already encodes glossary preference, so
        # the tiebreak only fires on the rare case of identical multiplicative
        # scores — which is intentional.
        weight = 1.0
        for entry in glossary.entries:
            if entry.right.lower() == word_text.lower():
                weight = entry.weight / 4.0
                break
        prob *= (1.0 + weight)
        is_glossary = True
    logprob = w.get("logprob", 0.0)
    engine = w.get("engine", "A")
    return (prob, logprob, is_glossary, engine)


def _reconcile_cluster(
    cluster: list[dict],
    glossary: Glossary | None,
    margin_threshold: float,
) -> dict:
    """Pick the best word from an anchor cluster using the spec's tie-break cascade."""
    if len(cluster) == 1:
        return {
            "word": cluster[0]["word"],
            "start": cluster[0]["start"],
            "end": cluster[0]["end"],
            "source": cluster[0].get("engine", "A"),
            "margin": 1.0,
        }

    # Sort by: score desc, logprob desc, glossary=True first, engine=A first
    scored = [(w, _score_word(w, glossary)) for w in cluster]
    scored.sort(key=lambda x: (
        -x[1][0],       # higher score first
        -x[1][1],       # higher logprob second
        not x[1][2],    # glossary members third
        x[1][3] != "A", # Engine A default last
    ))

    winner, (top_score, _, _, _) = scored[0]
    source = winner.get("engine", "A")

    if len(scored) > 1:
        runner_up_score = scored[1][1][0]
        margin = (top_score - runner_up_score) / max(top_score, 1e-6)
        word = winner["word"]
        if margin < margin_threshold:
            word = word + "[?]"
        return {
            "word": word,
            "start": winner["start"],
            "end": winner["end"],
            "source": source if runner_up_score == 0 else "both",
            "margin": round(margin, 3),
        }

    return {
        "word": winner["word"],
        "start": winner["start"],
        "end": winner["end"],
        "source": source,
        "margin": 1.0,
    }


def _rebuild_segments(
    engine_a_data: dict,
    reconciled_words: list[dict],
) -> list[dict]:
    """Re-emit segments using Engine A timing as spine, swapping in winner words.
    Unmatched B words (past the end of Engine A's word list) are merged into
    the temporally-nearest Engine A segment to avoid one-word fragment segments."""
    segments = []
    word_ptr = 0
    for seg in engine_a_data.get("segments", []):
        seg_words = []
        seg_text_parts = []
        for w in seg.get("words", []):
            if word_ptr < len(reconciled_words):
                rw = reconciled_words[word_ptr]
                seg_words.append(rw)
                seg_text_parts.append(rw["word"])
                word_ptr += 1
        text = " ".join(seg_text_parts)
        segments.append({
            "id": len(segments),
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "lang": seg.get("lang", "nl"),
            "words": seg_words,
        })

    # Merge unmatched B words that fall past Engine A's word list into
    # the temporally-nearest segment rather than creating tiny one-word fragments.
    remaining = reconciled_words[word_ptr:]
    if not segments and remaining:
        # Engine A produced zero segments — emit B words as a single segment
        segments.append({
            "id": 0,
            "start": remaining[0]["start"],
            "end": remaining[-1]["end"],
            "text": " ".join(w["word"] for w in remaining),
            "lang": "nl",
            "words": list(remaining),
        })
        return segments
    for w in remaining:
        w_center = (w["start"] + w["end"]) / 2
        best_idx = 0
        best_dist = float("inf")
        for i, seg in enumerate(segments):
            dist = abs(w_center - (seg["start"] + seg["end"]) / 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        segments[best_idx]["words"].append(w)
        segments[best_idx]["text"] += " " + w["word"]
        if w["start"] < segments[best_idx]["start"]:
            segments[best_idx]["start"] = w["start"]
        if w["end"] > segments[best_idx]["end"]:
            segments[best_idx]["end"] = w["end"]

    return segments


def reconcile(
    engine_a_path: Path,
    engine_b_path: Path,
    glossary: Glossary | None,
    output_path: Path,
    cfg: RoverConfig | None = None,
) -> dict:
    """Run ROVER reconciliation between two ASR engine outputs."""
    if cfg is None:
        cfg = RoverConfig()

    from .artifacts import read_json

    engine_a = read_json(engine_a_path)
    engine_b = read_json(engine_b_path)

    words_a = _flatten_words(engine_a)
    words_b = _flatten_words(engine_b)

    clusters = _build_anchor_clusters(words_a, words_b, cfg.anchor_window_s)

    reconciled = [
        _reconcile_cluster(c, glossary, cfg.margin_threshold)
        for c in clusters
    ]

    segments = _rebuild_segments(engine_a, reconciled)

    result = {
        "engine": "rover",
        "source_a": str(engine_a_path),
        "source_b": str(engine_b_path),
        "anchor_window_s": cfg.anchor_window_s,
        "margin_threshold": cfg.margin_threshold,
        "total_words": len(reconciled),
        "uncertain_words": sum(1 for w in reconciled if "[?]" in w["word"]),
        "segments": segments,
    }

    write_json(output_path, result)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ROVER reconciliation stage.")
    ap.add_argument("--engine-a", type=Path, required=True)
    ap.add_argument("--engine-b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--glossary", type=Path, default=None)
    ap.add_argument("--anchor-window", type=float, default=0.150)
    ap.add_argument("--margin-threshold", type=float, default=0.15)
    args = ap.parse_args()

    g = Glossary.resolve(args.glossary) if args.glossary else None
    cfg = RoverConfig(
        anchor_window_s=args.anchor_window,
        margin_threshold=args.margin_threshold,
    )
    result = reconcile(args.engine_a, args.engine_b, g, args.output, cfg)
    print(f"ROVER: {result['total_words']} words, {result['uncertain_words']} uncertain")
