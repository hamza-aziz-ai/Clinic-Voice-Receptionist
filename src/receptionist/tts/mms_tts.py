"""Meta MMS-TTS, for the languages Piper cannot speak.

WHY THIS AND NOT AI4BHARAT

IndicF5 and Indic Parler-TTS are the better Indic models and both were tried
first. Two things stopped them:

* Every AI4Bharat model repository is **gated** on Hugging Face - `config.json`
  returns 401 without an accepted licence and a token. A build that needs a
  human to click through a licence page is not one this repository can set up
  for you.
* `f5-tts` pins `numpy<=1.26.4`. Python 3.14 needs numpy 2.x. Installing it
  downgraded numpy and broke `torch` outright - not a version preference, a
  hard incompatibility.

MMS covers 1,100+ languages, is ungated, is plain `transformers` VITS, and
works with numpy 2. It is the model that runs.

THE LICENCE MATTERS HERE

MMS-TTS is **CC-BY-NC 4.0 — non-commercial**. Fine for a portfolio piece and
for evaluation; *not* fine for a clinic that charges patients. A real
deployment needs either an AI4Bharat licence accepted deliberately, or a
commercially-licensed voice. That is a legal constraint, not a technical one,
and it does not go away by ignoring it.
"""
from __future__ import annotations

import io
import logging
import wave
from typing import Any

log = logging.getLogger(__name__)

# Our language codes to MMS's ISO 639-3.
MMS_LANGUAGES: dict[str, str] = {
    "ta": "tam",
    "kn": "kan",
    "ml": "mal",
    "hi": "hin",
    "en": "eng",
}

LICENCE = "CC-BY-NC-4.0 (non-commercial)"


class MMSSpeaker:
    """VITS per language, loaded on first use and kept.

    Each model is roughly 140 MB of VRAM, so holding the two or three a clinic
    actually uses is cheap; loading all five would not be.
    """

    name = "mms-tts"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._models: dict[str, Any] = {}

    def can_speak(self, language: str) -> bool:
        return language in MMS_LANGUAGES

    def _model(self, language: str) -> Any:
        if language not in self._models:
            import torch
            from transformers import AutoTokenizer, VitsModel

            name = f"facebook/mms-tts-{MMS_LANGUAGES[language]}"
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = VitsModel.from_pretrained(name)
            try:
                model = model.to(self.device)
            except Exception as exc:
                log.warning("MMS on %s unavailable (%s); using CPU",
                            self.device, type(exc).__name__)
                self.device = "cpu"
                model = model.to("cpu")
            self._models[language] = (tokenizer, model.eval(), torch)
        return self._models[language]

    def synthesize(self, text: str, language: str = "en") -> bytes | None:
        if not text.strip() or not self.can_speak(language):
            return None
        try:
            tokenizer, model, torch = self._model(language)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                waveform = model(**inputs).waveform[0].cpu().numpy()
            return _to_wav(waveform, model.config.sampling_rate)
        except Exception as exc:
            log.warning("MMS TTS failed for %r: %s: %s", language,
                        type(exc).__name__, exc)
            return None


def _to_wav(waveform: Any, sample_rate: int) -> bytes:
    """Float array to 16-bit PCM WAV, without pulling in soundfile.

    The browser plays a WAV blob directly, so the conversion belongs here
    rather than leaving the caller to deal with float32 arrays.
    """
    import numpy as np

    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class CompositeSpeaker:
    """Piper where it has a voice, MMS for the rest.

    Not a preference for one vendor: Piper runs on CPU at RTF 0.05 and sounds
    better in the three languages it covers, while MMS covers the two it does
    not. Trying Piper first keeps the common path fast and off the GPU, which
    matters when Whisper is already holding VRAM.
    """

    name = "piper+mms"

    def __init__(self, piper: Any, mms: Any) -> None:
        self.piper = piper
        self.mms = mms

    def can_speak(self, language: str) -> bool:
        return self.piper.can_speak(language) or self.mms.can_speak(language)

    def synthesize(self, text: str, language: str = "en") -> bytes | None:
        if self.piper.can_speak(language):
            audio = self.piper.synthesize(text, language)
            if audio:
                return audio
        return self.mms.synthesize(text, language)
