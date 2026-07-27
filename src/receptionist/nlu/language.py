"""Language detection for inbound calls.

Script detection by Unicode range, which is exact for Indic scripts, plus a
romanised-keyword fallback for callers who code-switch into Latin script.

Detection is deliberately conservative. Answering a Malayalam caller in
English is a bad experience; answering an English caller in Malayalam is a
failed call. When the signal is weak the agent asks rather than guesses -
`UNCERTAIN` is a valid outcome and the call flow handles it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "ta", "kn", "ml", "hi", "uncertain"]

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "ta": "Tamil", "kn": "Kannada",
    "ml": "Malayalam", "hi": "Hindi", "uncertain": "Undetermined",
}

# Unicode blocks. Exact - a Tamil codepoint is Tamil, with no ambiguity.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "ta": (0x0B80, 0x0BFF),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "hi": (0x0900, 0x097F),
}

# Romanised cues, for callers typing/speaking transliterated Indic.
ROMANISED_MARKERS: dict[str, tuple[str, ...]] = {
    "ta": ("vanakkam", "enakku", "vendum", "podhu", "naan", "illai", "seri"),
    "kn": ("namaskara", "beku", "nanage", "illa", "sari", "yavaga"),
    "ml": ("namaskaram", "enikku", "venam", "illa", "sheri", "eppol"),
    "hi": ("namaste", "mujhe", "chahiye", "kab", "nahi", "theek"),
}

MIN_SCRIPT_RATIO = 0.08
MIN_SCRIPT_CHARS = 4
MIN_ROMANISED_HITS = 2


@dataclass(frozen=True)
class LanguageDetection:
    language: Language
    confidence: float
    method: str
    evidence: str

    @property
    def is_confident(self) -> bool:
        return self.language != "uncertain" and self.confidence >= 0.6


def detect_language(text: str) -> LanguageDetection:
    if not text or not text.strip():
        return LanguageDetection("uncertain", 0.0, "empty", "no speech recognised")

    counts = {code: 0 for code in SCRIPT_RANGES}
    latin = 0
    total = 0
    for ch in text:
        cp = ord(ch)
        # Check the Indic blocks BEFORE isalpha(). Combining vowel signs and
        # viramas are Unicode category Mn/Mc, not Lo - so isalpha() is False
        # for them. Filtering on isalpha() first silently discarded roughly a
        # fifth of the Devanagari/Dravidian characters in a typical utterance
        # and pushed every code-switched call below the detection threshold.
        matched = False
        for code, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[code] += 1
                total += 1
                matched = True
                break
        if matched:
            continue
        if not ch.isalpha():
            continue
        total += 1
        if cp < 128:
            latin += 1

    if total == 0:
        return LanguageDetection("uncertain", 0.0, "no_letters", "digits/punctuation only")

    best = max(counts, key=lambda k: counts[k])
    ratio = counts[best] / total
    if ratio >= MIN_SCRIPT_RATIO and counts[best] >= MIN_SCRIPT_CHARS:
        # Script evidence is strong evidence; confidence scales with purity.
        return LanguageDetection(
            best, min(0.99, 0.6 + ratio * 0.4), "script",
            f"{counts[best]}/{total} characters in {LANGUAGE_NAMES[best]} block",
        )

    lowered = text.lower()
    hits = {
        code: sum(1 for m in markers if re.search(rf"\b{m}\b", lowered))
        for code, markers in ROMANISED_MARKERS.items()
    }
    best_rom = max(hits, key=lambda k: hits[k])
    if hits[best_rom] >= MIN_ROMANISED_HITS:
        return LanguageDetection(
            best_rom, min(0.85, 0.5 + 0.12 * hits[best_rom]), "romanised",
            f"{hits[best_rom]} {LANGUAGE_NAMES[best_rom]} markers in Latin script",
        )

    if latin / total > 0.9:
        return LanguageDetection("en", 0.75, "latin_default",
                                 f"{latin}/{total} Latin characters, no Indic markers")

    return LanguageDetection("uncertain", 0.3, "mixed",
                             "mixed script with no dominant language")
