"""Tests for pipeline.postcorrect — diff guard, parsing, retry logic."""

import unittest
from unittest.mock import patch

from pipeline.postcorrect import (
    _token_edit_ratio,
    _parse_corrections,
    _MAX_EDIT_RATIO,
)


class TestTokenEditRatio(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(_token_edit_ratio("hello world", "hello world"), 0.0)

    def test_completely_different(self):
        r = _token_edit_ratio("aaa bbb ccc", "xxx yyy zzz")
        self.assertEqual(r, 1.0)

    def test_one_word_changed(self):
        r = _token_edit_ratio("hello world test", "hello earth test")
        self.assertAlmostEqual(r, 1/3, places=2)

    def test_empty(self):
        self.assertEqual(_token_edit_ratio("", ""), 0.0)
        self.assertEqual(_token_edit_ratio("hello", ""), 1.0)

    def test_additions(self):
        r = _token_edit_ratio("hello", "hello world")
        self.assertAlmostEqual(r, 0.5)


class TestParseCorrections(unittest.TestCase):
    def _seg(self, seg_id, text):
        return {"id": seg_id, "text": text}

    def test_valid_correction(self):
        raw = '''[{"seg_id": 0, "original": "vloek tester device kabel aansluiting", "corrected": "Fluke tester device kabel aansluiting",
                   "changed_words": [{"idx": 0, "before": "vloek", "after": "Fluke", "confidence": 0.95}]}]'''
        segs = [self._seg(0, "vloek tester device kabel aansluiting")]
        result = _parse_corrections(raw, segs)
        self.assertEqual(result[0]["text"], "Fluke tester device kabel aansluiting")
        self.assertIn("corrections", result[0])

    def test_low_confidence_rejected(self):
        raw = '''[{"seg_id": 0, "original": "vloek tester", "corrected": "Fluke tester",
                   "changed_words": [{"idx": 0, "before": "vloek", "after": "Fluke", "confidence": 0.5}]}]'''
        segs = [self._seg(0, "vloek tester")]
        result = _parse_corrections(raw, segs)
        # Low confidence → original kept
        self.assertEqual(result[0]["text"], "vloek tester")

    def test_diff_guard(self):
        # Too many changes
        seg_text = " ".join(f"word{i}" for i in range(20))
        corr_text = " ".join(f"changed{i}" for i in range(20))
        raw = f'''[{{"seg_id": 0, "original": "{seg_text}", "corrected": "{corr_text}",
                   "changed_words": [{{"idx": 0, "before": "word0", "after": "changed0", "confidence": 0.95}}]}}]'''
        segs = [self._seg(0, seg_text)]
        result = _parse_corrections(raw, segs)
        self.assertIn("[!unverified]", result[0]["text"])

    def test_invalid_json(self):
        segs = [self._seg(0, "hello")]
        result = _parse_corrections("not json", segs)
        self.assertEqual(result[0]["text"], "hello")

    def test_markdown_fences_stripped(self):
        raw = '```json\n[{"seg_id": 0, "original": "x dit is een lange zin", "corrected": "y dit is een lange zin", "changed_words": [{"idx": 0, "before": "x", "after": "y", "confidence": 0.9}]}]\n```'
        segs = [self._seg(0, "x dit is een lange zin")]
        result = _parse_corrections(raw, segs)
        self.assertEqual(result[0]["text"], "y dit is een lange zin")

    def test_skipped_segments_filled(self):
        raw = '[]'  # LLM returned empty array
        segs = [self._seg(0, "hello"), self._seg(1, "world")]
        result = _parse_corrections(raw, segs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "hello")
        self.assertEqual(result[1]["text"], "world")

    def test_results_sorted_by_id(self):
        raw = '''[
            {"seg_id": 2, "original": "c token extra words hier", "corrected": "c2 token extra words hier", "changed_words": [{"idx": 0, "before": "c", "after": "c2", "confidence": 0.9}]},
            {"seg_id": 0, "original": "a token extra words hier", "corrected": "a2 token extra words hier", "changed_words": [{"idx": 0, "before": "a", "after": "a2", "confidence": 0.9}]}
        ]'''
        segs = [self._seg(0, "a token extra words hier"), self._seg(1, "b"), self._seg(2, "c token extra words hier")]
        result = _parse_corrections(raw, segs)
        self.assertEqual(result[0]["id"], 0)
        self.assertEqual(result[1]["id"], 1)
        self.assertEqual(result[2]["id"], 2)


class TestGlmVersionRegex(unittest.TestCase):
    """Test the GLM model version parsing from postcorrect.py."""

    def test_standard_version(self):
        import re
        match = re.match(r"glm-(\d+)(?:\.(\d+))?", "glm-5.1")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 5)
        self.assertEqual(int(match.group(2)), 1)

    def test_major_only(self):
        import re
        match = re.match(r"glm-(\d+)(?:\.(\d+))?", "glm-6")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 6)
        self.assertIsNone(match.group(2))


if __name__ == "__main__":
    unittest.main()
