"""Tests for pipeline.glossary — loading, parsing, weighting, prompt generation."""

import sys
import textwrap
import unittest
from pathlib import Path

from pipeline.glossary import (
    DEFAULT_SECTION,
    SECTION_WEIGHTS,
    Glossary,
    GlossaryEntry,
    _SECTION_RE,
    _SEP_RE,
)


class TestGlossaryEntry(unittest.TestCase):
    def test_weight_by_section(self):
        self.assertEqual(GlossaryEntry("x", "y", "brand").weight, 4)
        self.assertEqual(GlossaryEntry("x", "y", "person").weight, 3)
        self.assertEqual(GlossaryEntry("x", "y", "place").weight, 2)
        self.assertEqual(GlossaryEntry("x", "y", "term").weight, 1)

    def test_unknown_section_weight(self):
        self.assertEqual(GlossaryEntry("x", "y", "custom").weight, 1)


class TestParsing(unittest.TestCase):
    def _load(self, text: str) -> Glossary:
        p = Path("/tmp/test_glossary_load.txt")
        p.write_text(textwrap.dedent(text), encoding="utf-8")
        return Glossary.load(p)

    def test_basic_entries(self):
        g = self._load("""
        vloek -> Fluke
        annexter → Anixter
        """)
        self.assertEqual(len(g.entries), 2)
        self.assertEqual(g.entries[0].wrong, "vloek")
        self.assertEqual(g.entries[0].right, "Fluke")
        self.assertEqual(g.entries[1].right, "Anixter")

    def test_sections(self):
        g = self._load("""
        [brand]
        vloek -> Fluke

        [person]
        bjorn -> Bjorn
        """)
        self.assertEqual(g.entries[0].section, "brand")
        self.assertEqual(g.entries[1].section, "person")

    def test_comments_and_blanks(self):
        g = self._load("""
        # This is a comment
        vloek -> Fluke

        # Another comment
        annexter -> Anixter
        """)
        self.assertEqual(len(g.entries), 2)

    def test_malformed_lines_skipped(self):
        g = self._load("""
        vloek -> Fluke
        this has no arrow
        """)
        self.assertEqual(len(g.entries), 1)

    def test_empty_wrong_or_right(self):
        g = self._load("""
        -> right
        wrong ->
        """)
        self.assertEqual(len(g.entries), 0)

    def test_default_section_is_term(self):
        g = self._load("vloek -> Fluke\n")
        self.assertEqual(g.entries[0].section, DEFAULT_SECTION)


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.g = Glossary(entries=[
            GlossaryEntry("vloek", "Fluke", "brand"),
            GlossaryEntry("fluk", "Fluke", "brand"),
        ])

    def test_exact_match(self):
        self.assertEqual(self.g.lookup("vloek"), "Fluke")

    def test_case_insensitive(self):
        self.assertEqual(self.g.lookup("VLOEK"), "Fluke")

    def test_no_match(self):
        self.assertIsNone(self.g.lookup("unknown"))

    def test_is_canonical(self):
        self.assertTrue(self.g.is_canonical("Fluke"))
        self.assertTrue(self.g.is_canonical("fluke"))
        self.assertFalse(self.g.is_canonical("vloek"))


class TestCanonicalTerms(unittest.TestCase):
    def test_dedup_and_order_by_weight(self):
        g = Glossary(entries=[
            GlossaryEntry("bjorn", "Bjorn", "person"),
            GlossaryEntry("vloek", "Fluke", "brand"),
            GlossaryEntry("fluk", "Fluke", "brand"),
        ])
        terms = g.canonical_terms()
        # Fluke (brand, weight 4) should come before Bjorn (person, weight 3)
        self.assertEqual(terms, ["Fluke", "Bjorn"])
        # No duplicates
        self.assertEqual(len(terms), len(set(t.lower() for t in terms)))


class TestPromptBlock(unittest.TestCase):
    def test_empty_glossary(self):
        g = Glossary()
        self.assertEqual(g.to_prompt_block(), "")

    def test_section_order_by_weight(self):
        g = Glossary(entries=[
            GlossaryEntry("bjorn", "Bjorn", "person"),
            GlossaryEntry("vloek", "Fluke", "brand"),
        ])
        block = g.to_prompt_block()
        # Brand section should appear before person section
        brand_idx = block.index("[brand]")
        person_idx = block.index("[person]")
        self.assertLess(brand_idx, person_idx)

    def test_format(self):
        g = Glossary(entries=[
            GlossaryEntry("vloek", "Fluke", "brand"),
        ])
        block = g.to_prompt_block()
        self.assertIn("[brand]", block)
        self.assertIn("vloek -> Fluke", block)


class TestSeedDefault(unittest.TestCase):
    def test_seed_creates_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "glossary.txt"
            from pipeline.glossary import seed_default
            result = seed_default(p)
            self.assertTrue(result)
            self.assertTrue(p.exists())
            # Second call should not overwrite
            result2 = seed_default(p)
            self.assertFalse(result2)


if __name__ == "__main__":
    unittest.main()
