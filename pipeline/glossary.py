"""Glossary loader for brand names, people, places, and domain terms.

Format (plain text, one entry per line):

    [brand]
    vloek -> Fluke
    annexter → Anixter
    comscope → CommScope

    [person]
    bjorn → Bjorn

Comments start with `#`. Either `->` or `→` separates `wrong` from `right`.
Sections drive a weight used by ROVER tie-breaking and prompt ordering: brands
are strongest because they collide with common words most often.
"""

from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path

SECTION_WEIGHTS = {"brand": 4, "person": 3, "place": 2, "term": 1}
DEFAULT_SECTION = "term"
USER_GLOSSARY = Path.home() / ".config" / "whisper" / "glossary.txt"
CONTAINER_GLOSSARY = Path("/run/glossary.txt")

_SECTION_RE = re.compile(r"^\[([a-z]+)\]\s*$")
_SEP_RE = re.compile(r"\s*(?:->|→)\s*")


@dataclass(frozen=True)
class GlossaryEntry:
    wrong: str
    right: str
    section: str

    @property
    def weight(self) -> int:
        return SECTION_WEIGHTS.get(self.section, 1)


@dataclass
class Glossary:
    entries: list[GlossaryEntry] = field(default_factory=list)
    source: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        entries: list[GlossaryEntry] = []
        section = DEFAULT_SECTION
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _SECTION_RE.match(line)
            if m:
                section = m.group(1).lower()
                continue
            parts = _SEP_RE.split(line, maxsplit=1)
            if len(parts) != 2:
                print(f"glossary: skipped malformed line: {raw!r}", file=sys.stderr)
                continue
            wrong, right = parts[0].strip(), parts[1].strip()
            if not wrong or not right:
                continue
            entries.append(GlossaryEntry(wrong=wrong, right=right, section=section))
        return cls(entries=entries, source=path)

    @classmethod
    def resolve(cls, override: str | os.PathLike | None = None) -> "Glossary":
        candidates: list[Path] = []
        if override:
            candidates.append(Path(override))
        if CONTAINER_GLOSSARY.exists():
            candidates.append(CONTAINER_GLOSSARY)
        candidates.append(USER_GLOSSARY)
        for path in candidates:
            if path and path.exists():
                return cls.load(path)
        return cls()

    def canonical_terms(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for entry in sorted(self.entries, key=lambda e: -e.weight):
            key = entry.right.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(entry.right)
        return ordered

    def lookup(self, term: str) -> str | None:
        needle = term.lower()
        for entry in self.entries:
            if entry.wrong.lower() == needle:
                return entry.right
        return None

    def is_canonical(self, term: str) -> bool:
        needle = term.lower()
        return any(entry.right.lower() == needle for entry in self.entries)

    def to_prompt_block(self) -> str:
        if not self.entries:
            return ""
        lines: list[str] = []
        by_section: dict[str, list[GlossaryEntry]] = {}
        for entry in self.entries:
            by_section.setdefault(entry.section, []).append(entry)
        for section in sorted(by_section, key=lambda s: -SECTION_WEIGHTS.get(s, 1)):
            lines.append(f"[{section}]")
            for entry in by_section[section]:
                lines.append(f"{entry.wrong} -> {entry.right}")
        return "\n".join(lines)


def seed_default(path: Path = USER_GLOSSARY) -> bool:
    """Write a starter glossary if none exists. Returns True if a file was written."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_GLOSSARY, encoding="utf-8")
    return True


_DEFAULT_GLOSSARY = """# Whisper transcription glossary.
# Format: wrong -> right     (or:  wrong → right)
# Sections weight the entry: brand > person > place > term.

[brand]
vloek -> Fluke
fluk -> Fluke
annexter -> Anixter
anexter -> Anixter
comscope -> CommScope
com scope -> CommScope
connect-wise -> ConnectWise
connect wise -> ConnectWise
applyermap -> wiremap
apply-er-map -> wiremap
rust oleum -> RustOleum
rustolium -> RustOleum
giga speed xl -> GigaSpeed XL
gigaspeed -> GigaSpeed
microsoft teams -> Microsoft Teams
ms teams -> Microsoft Teams
dnai -> de AI
dinapse -> Dynapse

[person]
bjorn -> Bjorn
brent -> Brent
kevin -> Kevin
david -> David

[place]
berendrechtstraat -> Berendrechtstraat
mespelare -> Mespelare

# [term]
# Add domain-specific terms here (weight 1, lowest priority).
# example -> Example
"""


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Glossary inspector / seeder.")
    ap.add_argument("--seed", action="store_true", help="Write the default glossary if none exists.")
    ap.add_argument("--glossary", help="Path to a glossary file (otherwise resolved).")
    ap.add_argument("--dump", action="store_true", help="Print loaded entries as JSON.")
    args = ap.parse_args()

    if args.seed:
        wrote = seed_default()
        print(f"{'wrote' if wrote else 'kept'} {USER_GLOSSARY}")

    g = Glossary.resolve(args.glossary)
    print(f"loaded {len(g.entries)} entries from {g.source}")
    if args.dump:
        print(json.dumps([e.__dict__ for e in g.entries], indent=2, ensure_ascii=False))
