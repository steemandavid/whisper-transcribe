"""Smoke tests for prompt_builder — proves the poison-prompt scenario is handled."""

import unittest
from pathlib import Path

from pipeline.glossary import Glossary
from pipeline.prompt_builder import (
    MAX_PROMPT_CHARS,
    MAX_PROMPT_TOKENS,
    PromptBuildResult,
    build_prompt,
    filename_priors,
    sanitise_vocab,
    strip_ansi,
    strip_thinking,
)

POISON_PATH = Path("/storage/fileshare/.whisper_prompt_v1_2965923.txt")

POISON_TEXT = (
    "\x1b[K\x1b[2D<think!\x1b[0m>\n"
    "Let me analyze this transcript step by step.\n"
    "The user wants keywords extracted.\n"
    "done thinking.\n"
    "\x1b[?25h\x1b[?25l\x1b[K\x1b[2D\n"
    "Here are the keywords: Fluke, Anixter, CommScope, ConnectWise, wiremap, "
    "RustOleum, Bjorn, Microsoft, Teams, GigaSpeed XL, Berendrechtstraat, Mespelare, "
    "vloek, annexter, comscope, applyermap\n"
)


class TestStripThinking(unittest.TestCase):
    def test_think_tag_block(self):
        raw = "before\x3cthink\x3einternal analysis here\x3c/think\x3eafter"
        cleaned = strip_thinking(raw)
        self.assertNotIn("internal analysis", cleaned)
        self.assertIn("before", cleaned)
        self.assertIn("after", cleaned)

    def test_done_thinking_line(self):
        raw = "Thinking...\nsome verbiage\n...done thinking.\nactual output"
        cleaned = strip_thinking(raw)
        self.assertNotIn("done thinking", cleaned)
        self.assertIn("actual output", cleaned)


class TestStripAnsi(unittest.TestCase):
    def test_csi_sequences(self):
        raw = "hello\x1b[K\x1b[2Dworld"
        cleaned = strip_ansi(raw)
        self.assertEqual(cleaned, "helloworld")

    def test_bare_bracket_artefacts(self):
        # Real poison uses \x1b[K and \x1b[2D (proper CSI sequences)
        raw = "foo\x1b[Kbar\x1b[2Dbaz"
        cleaned = strip_ansi(raw)
        self.assertNotIn("\x1b[K", cleaned)
        self.assertNotIn("\x1b[2D", cleaned)
        self.assertEqual(cleaned, "foobarbaz")


class TestSanitiseVocabSynthetic(unittest.TestCase):
    def test_clean_vocab_passes_through(self):
        tokens, warns = sanitise_vocab("Fluke, Anixter, CommScope, ConnectWise")
        self.assertIn("Fluke", tokens)
        self.assertIn("Anixter", tokens)
        self.assertEqual(len(warns), 0)

    def test_ansi_stripped(self):
        tokens, _ = sanitise_vocab("Fluke\x1b[K\x1b[2D, Anixter")
        self.assertTrue(len(tokens) >= 1)
        for t in tokens:
            self.assertNotIn("\x1b", t)
            self.assertNotIn("[K", t)
            self.assertNotIn("[2D", t)

    def test_thinking_block_stripped(self):
        raw = "\x1b[K\x1b[2DThinking...\nanalysis here\n...done thinking.\nFluke, Anixter"
        tokens, _ = sanitise_vocab(raw)
        for t in tokens:
            self.assertNotIn("done thinking", t)

    def test_empty_input(self):
        tokens, warns = sanitise_vocab("")
        self.assertEqual(tokens, [])
        self.assertTrue(len(warns) > 0)

    def test_long_phrases_dropped(self):
        raw = "This is a very long sentence that should definitely be rejected, Fluke"
        tokens, warns = sanitise_vocab(raw)
        self.assertIn("Fluke", tokens)
        self.assertTrue(any("long phrase" in w for w in warns))

    def test_max_token_cap(self):
        parts = [f"word{i}" for i in range(200)]
        tokens, _ = sanitise_vocab(", ".join(parts))
        self.assertLessEqual(len(tokens), MAX_PROMPT_TOKENS)


class TestSanitiseVocabPoisonFile(unittest.TestCase):
    """Test against the real poison prompt file if it exists."""

    @unittest.skipUnless(POISON_PATH.exists(), "poison file not available")
    def test_real_poison_file(self):
        raw = POISON_PATH.read_text(encoding="utf-8", errors="replace")
        tokens, warns = sanitise_vocab(raw)
        # Must produce some tokens (don't reject everything)
        self.assertTrue(len(tokens) > 0, "sanitise_vocab should extract something useful")
        # No escape artefacts
        for t in tokens:
            self.assertNotIn("\x1b", t, f"token contains ESC: {t!r}")
            self.assertNotIn("[K", t, f"token contains bare [K: {t!r}")
            self.assertNotIn("[2D", t, f"token contains bare [2D: {t!r}")
        # No thinking residue
        for t in tokens:
            self.assertNotIn("\x1b", t)
            self.assertNotIn("done thinking", t.lower())


class TestFilenamePriors(unittest.TestCase):
    def test_brent_microsoft(self):
        priors = filename_priors("brent microsoft - CallRecord_20260424-164320_+32472073503.m4a")
        names = [p.lower() for p in priors]
        self.assertIn("brent", names)
        self.assertIn("microsoft", names)

    def test_phone_numbers_dropped(self):
        priors = filename_priors("call_+32472073503.m4a")
        self.assertNotIn("+32472073503", priors)

    def test_datestamps_dropped(self):
        priors = filename_priors("recording_20260424.m4a")
        self.assertNotIn("20260424", priors)

    def test_short_tokens_dropped(self):
        priors = filename_priors("a-b-c.m4a")
        self.assertEqual(priors, [])


class TestBuildPrompt(unittest.TestCase):
    def _glossary(self) -> Glossary:
        p = Path.home() / ".config" / "whisper" / "glossary.txt"
        if p.exists():
            return Glossary.load(p)
        return Glossary()

    def test_basic_build(self):
        res = build_prompt(audio_path="test.m4a", glossary=self._glossary())
        self.assertTrue(len(res.prompt) > 0)
        self.assertLessEqual(len(res.prompt), MAX_PROMPT_CHARS)

    def test_poison_input_fallback(self):
        g = self._glossary()
        res = build_prompt(
            audio_path="brent_microsoft_test.m4a",
            glossary=g,
            llm_vocab=POISON_TEXT,
        )
        # Prompt must be within bounds
        self.assertLessEqual(len(res.prompt), MAX_PROMPT_CHARS)
        # Filename priors should fire
        prompt_lower = res.prompt.lower()
        self.assertIn("brent", prompt_lower)
        self.assertIn("microsoft", prompt_lower)
        # Glossary canonical terms should be present (if glossary loaded)
        if g.entries:
            self.assertIn("Fluke", res.prompt)
        # Warnings should be non-empty (truncation or long-phrase drops)
        self.assertTrue(len(res.warnings) > 0)

    @unittest.skipUnless(POISON_PATH.exists(), "poison file not available")
    def test_real_poison_build(self):
        raw = POISON_PATH.read_text(encoding="utf-8", errors="replace")
        res = build_prompt(
            audio_path="brent_microsoft - CallRecord_20260424.m4a",
            glossary=self._glossary(),
            llm_vocab=raw,
        )
        self.assertLessEqual(len(res.prompt), MAX_PROMPT_CHARS)
        self.assertIn("Brent", res.prompt)
        self.assertIn("Microsoft", res.prompt)
        # At least one glossary canonical term
        glossary_hit = any(t in res.prompt for t in ("Fluke", "Anixter", "CommScope", "ConnectWise"))
        self.assertTrue(glossary_hit, "expected at least one glossary canonical term in prompt")

    def test_user_context_prepended(self):
        res = build_prompt(
            audio_path="test.m4a",
            user_context="Dit is een medisch consult.",
        )
        self.assertTrue(res.prompt.startswith("Dit is een medisch consult."))

    def test_empty_glossary_ok(self):
        res = build_prompt(audio_path="test.m4a", glossary=Glossary())
        self.assertTrue(len(res.prompt) >= 0)


if __name__ == "__main__":
    unittest.main()
