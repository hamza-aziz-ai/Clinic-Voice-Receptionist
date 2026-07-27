"""Piper text-to-speech. Free, offline, and missing two of our languages.

THE GAP, STATED FIRST

Piper has voices for English, Hindi and Malayalam. It has **none for Tamil or
Kannada**. On a project whose entire subject is these five languages, that is
not a footnote - two of the five cannot be spoken at all.

The system refuses to speak those rather than substituting an English voice.
An English model fed Tamil script does not produce accented Tamil, it
produces noise, and a caller hearing noise where their language should be is
worse served than one who gets text. ``synthesize`` returns None and says
which language it could not voice.

Closing that gap means AI4Bharat's IndicF5 or Indic Parler-TTS, which cover
all four Indic languages here and need torch. That is a real install, not a
config change, and it is the next piece of work if the Tamil and Kannada
demos have to be audible.
"""
from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_VOICE_DIR = Path("models/piper")

# One voice per language we can actually speak. Absence is the point: a
# missing key means the system stays silent in that language rather than
# guessing with the wrong phoneme set.
VOICES: dict[str, str] = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-pratham-medium",
    "ml": "ml_IN-meera-medium",
    # "ta": no Piper voice exists
    # "kn": no Piper voice exists
}

UNVOICED_LANGUAGES = ("ta", "kn")


class PiperSpeaker:
    """Text to WAV bytes. Voices are loaded once and reused."""

    name = "piper"

    def __init__(self, voice_dir: Path | str = DEFAULT_VOICE_DIR) -> None:
        self.voice_dir = Path(voice_dir)
        self._voices: dict[str, Any] = {}

    def can_speak(self, language: str) -> bool:
        return language in VOICES and self._voice_path(language).exists()

    def _voice_path(self, language: str) -> Path:
        return self.voice_dir / f"{VOICES[language]}.onnx"

    def _voice(self, language: str) -> Any:
        if language not in self._voices:
            from piper import PiperVoice

            self._voices[language] = PiperVoice.load(str(self._voice_path(language)))
        return self._voices[language]

    def synthesize(self, text: str, language: str = "en") -> bytes | None:
        """WAV bytes, or None with a logged reason.

        None is a normal outcome, not an error path. The caller falls back to
        showing the text, which is what a language with no voice gets.
        """
        if not text.strip():
            return None
        if language in UNVOICED_LANGUAGES:
            log.warning(
                "no Piper voice for %r; the agent stays silent rather than "
                "reading %s text with an English voice", language, language,
            )
            return None
        if not self.can_speak(language):
            log.warning("no voice file for %r in %s", language, self.voice_dir)
            return None

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                self._voice(language).synthesize_wav(text, wav)
            return buffer.getvalue()
        except Exception as exc:
            # A failed reply is a degraded call, not a lost booking - the
            # appointment is already made by the time anything is spoken.
            log.warning("TTS failed: %s: %s", type(exc).__name__, exc)
            return None
