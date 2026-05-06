"""Tests for pipeline.rover — Levenshtein, clustering, scoring, reconciliation."""

import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.rover import (
    RoverConfig,
    _build_anchor_clusters,
    _flatten_words,
    _levenshtein_ratio,
    _rebuild_segments,
    _reconcile_cluster,
    _score_word,
)
from pipeline.glossary import Glossary, GlossaryEntry


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(_levenshtein_ratio("hello", "hello"), 1.0)

    def test_empty(self):
        self.assertEqual(_levenshtein_ratio("", "hello"), 0.0)
        self.assertEqual(_levenshtein_ratio("hello", ""), 0.0)

    def test_similar(self):
        r = _levenshtein_ratio("Anixter", "Annexter")
        self.assertGreater(r, 0.6)

    def test_different(self):
        r = _levenshtein_ratio("hello", "world")
        self.assertLess(r, 0.4)

    def test_case_sensitive(self):
        # Function is case-sensitive; same word different case gives 0.0
        r = _levenshtein_ratio("fluke", "FLUKE")
        self.assertEqual(r, 0.0)
        # Identical case gives 1.0
        self.assertEqual(_levenshtein_ratio("fluke", "fluke"), 1.0)


class TestFlattenWords(unittest.TestCase):
    def test_basic(self):
        data = {
            "segments": [
                {
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "hello", "prob": 0.9},
                        {"start": 0.5, "end": 1.0, "word": "world", "prob": 0.8},
                    ],
                    "avg_logprob": -0.3,
                }
            ]
        }
        words = _flatten_words(data)
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["word"], "hello")
        self.assertAlmostEqual(words[0]["logprob"], -0.3)
        self.assertAlmostEqual(words[0]["center"], 0.25)

    def test_strips_empty(self):
        data = {
            "segments": [
                {
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "  ", "prob": 0.9},
                        {"start": 0.5, "end": 1.0, "word": "real", "prob": 0.8},
                    ],
                    "avg_logprob": -0.2,
                }
            ]
        }
        words = _flatten_words(data)
        self.assertEqual(len(words), 1)


class TestBuildAnchorClusters(unittest.TestCase):
    def _word(self, word, start, end, prob=0.9, engine="A"):
        return {"word": word, "start": start, "end": end, "center": (start+end)/2,
                "prob": prob, "logprob": -0.3, "engine": engine}

    def test_perfect_match(self):
        a = [self._word("hello", 0.0, 0.5)]
        b = [self._word("hello", 0.0, 0.5)]
        clusters = _build_anchor_clusters(a, b, 0.150)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_no_match(self):
        a = [self._word("hello", 0.0, 0.5)]
        b = [self._word("world", 5.0, 5.5)]
        clusters = _build_anchor_clusters(a, b, 0.150)
        self.assertEqual(len(clusters), 2)  # A singleton + B singleton
        self.assertEqual(len(clusters[0]), 1)
        self.assertEqual(len(clusters[1]), 1)

    def test_threshold_06(self):
        a = [self._word("Fluke", 0.0, 0.5)]
        b = [self._word("broke", 0.0, 0.5)]
        clusters = _build_anchor_clusters(a, b, 0.150)
        # "Fluke" vs "broke" similarity < 0.6 → no pairing
        self.assertEqual(len(clusters), 2)


class TestScoreWord(unittest.TestCase):
    def test_basic_score(self):
        w = {"prob": 0.8, "logprob": -0.3, "word": "hello", "engine": "A"}
        score, logprob, is_glossary, engine = _score_word(w, None)
        self.assertAlmostEqual(score, 0.8)
        self.assertFalse(is_glossary)
        self.assertEqual(engine, "A")

    def test_glossary_multiplicative(self):
        glossary = Glossary(entries=[GlossaryEntry("vloek", "Fluke", "brand")])
        w = {"prob": 0.8, "logprob": -0.3, "word": "Fluke", "engine": "A"}
        score, _, is_glossary, _ = _score_word(w, glossary)
        self.assertTrue(is_glossary)
        # brand weight = 4, weight/4 = 1.0, so prob *= (1 + 1.0) = 0.8 * 2.0 = 1.6
        self.assertAlmostEqual(score, 1.6)

    def test_zero_prob_fallback(self):
        w = {"prob": 0.0, "logprob": -0.5, "word": "test", "engine": "A"}
        score, _, _, _ = _score_word(w, None)
        self.assertGreater(score, 0)


class TestReconcileCluster(unittest.TestCase):
    def test_single_word(self):
        cluster = [{"word": "hello", "start": 0.0, "end": 0.5, "prob": 0.9,
                     "logprob": -0.2, "engine": "A"}]
        result = _reconcile_cluster(cluster, None, 0.15)
        self.assertEqual(result["word"], "hello")
        self.assertEqual(result["margin"], 1.0)

    def test_higher_prob_wins(self):
        cluster = [
            {"word": "hello", "start": 0.0, "end": 0.5, "prob": 0.7,
             "logprob": -0.3, "engine": "A"},
            {"word": "hallo", "start": 0.0, "end": 0.5, "prob": 0.9,
             "logprob": -0.1, "engine": "B"},
        ]
        result = _reconcile_cluster(cluster, None, 0.15)
        self.assertEqual(result["word"], "hallo")

    def test_uncertain_marker(self):
        cluster = [
            {"word": "hello", "start": 0.0, "end": 0.5, "prob": 0.90,
             "logprob": -0.1, "engine": "A"},
            {"word": "hallo", "start": 0.0, "end": 0.5, "prob": 0.89,
             "logprob": -0.1, "engine": "B"},
        ]
        result = _reconcile_cluster(cluster, None, 0.15)
        self.assertIn("[?]", result["word"])


class TestRebuildSegments(unittest.TestCase):
    def _engine_a(self, words):
        return {
            "segments": [
                {
                    "id": 0, "start": 0.0, "end": 2.0, "lang": "nl",
                    "words": words,
                }
            ]
        }

    def test_basic_rebuild(self):
        rw = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        a_data = self._engine_a([{"word": w["word"]} for w in rw])
        segments = _rebuild_segments(a_data, rw)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hello world")

    def test_empty_engine_a_with_remaining(self):
        """Engine A produces zero segments but Engine B has words."""
        a_data = {"segments": []}
        rw = [
            {"word": "orphan", "start": 1.0, "end": 1.5},
            {"word": "word", "start": 2.0, "end": 2.5},
        ]
        segments = _rebuild_segments(a_data, rw)
        self.assertEqual(len(segments), 1)
        self.assertIn("orphan", segments[0]["text"])
        self.assertIn("word", segments[0]["text"])

    def test_unmatched_b_merged_into_nearest(self):
        """Unmatched B words merge into temporally nearest A segment."""
        a_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "lang": "nl",
                 "words": [{"word": "hello"}]},
                {"id": 1, "start": 5.0, "end": 8.0, "lang": "nl",
                 "words": [{"word": "world"}]},
            ]
        }
        # One matched word + one unmatched B word at t=6
        rw = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 5.0, "end": 5.5},
            {"word": "extra", "start": 6.0, "end": 6.5},  # unmatched
        ]
        segments = _rebuild_segments(a_data, rw)
        # "extra" at t=6 should merge into segment 1 (t=5-8), not segment 0 (t=0-3)
        self.assertEqual(len(segments), 2)
        self.assertIn("extra", segments[1]["text"])
        self.assertNotIn("extra", segments[0]["text"])


if __name__ == "__main__":
    unittest.main()
