"""Tests for pipeline.render — speaker assignment and text rendering."""

import unittest

from pipeline.render import _assign_speakers, render_text, run


class TestAssignSpeakers(unittest.TestCase):
    def _seg(self, start, end, text="hello"):
        return {"start": start, "end": end, "text": text}

    def test_no_turns(self):
        segs = [self._seg(0, 1)]
        result = _assign_speakers(segs, [])
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")

    def test_single_turn(self):
        segs = [self._seg(0, 1), self._seg(2, 3)]
        turns = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        result = _assign_speakers(segs, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_00")

    def test_two_speakers(self):
        segs = [self._seg(0, 2), self._seg(3, 5)]
        turns = [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        result = _assign_speakers(segs, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")

    def test_overlap_selection(self):
        segs = [self._seg(0, 3)]
        turns = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        result = _assign_speakers(segs, turns)
        # Center is 1.5 — both overlaps are equal, picks last best
        self.assertIn(result[0]["speaker"], ["SPEAKER_00", "SPEAKER_01"])


class TestRenderText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(render_text([], []), "")

    def test_single_speaker(self):
        segs = [{"text": "Hello world", "start": 0, "end": 2}]
        turns = [{"start": 0, "end": 5, "speaker": "SPEAKER_00"}]
        text = render_text(segs, turns)
        self.assertIn("[SPEAKER_00]", text)
        self.assertIn("  Hello world", text)

    def test_speaker_change(self):
        segs = [
            {"text": "Hi", "start": 0, "end": 1},
            {"text": "Hey", "start": 1, "end": 2},
        ]
        turns = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00"},
            {"start": 1, "end": 2, "speaker": "SPEAKER_01"},
        ]
        text = render_text(segs, turns)
        self.assertIn("[SPEAKER_00]", text)
        self.assertIn("[SPEAKER_01]", text)

    def test_empty_text_skipped(self):
        segs = [{"text": "", "start": 0, "end": 1}]
        text = render_text(segs, [])
        self.assertEqual(text.strip(), "")

    def test_uncertain_markers_preserved(self):
        segs = [{"text": "Fluke[?] test", "start": 0, "end": 2}]
        turns = [{"start": 0, "end": 5, "speaker": "SPEAKER_00"}]
        text = render_text(segs, turns)
        self.assertIn("[?]", text)

    def test_unverified_tag_preserved(self):
        segs = [{"text": "test [!unverified]", "start": 0, "end": 2}]
        turns = [{"start": 0, "end": 5, "speaker": "SPEAKER_00"}]
        text = render_text(segs, turns)
        self.assertIn("[!unverified]", text)


if __name__ == "__main__":
    unittest.main()
