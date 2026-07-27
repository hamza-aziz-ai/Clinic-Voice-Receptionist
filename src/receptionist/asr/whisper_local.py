"""faster-whisper, running locally on the GPU.

WHY WHISPER, GIVEN base.py ARGUES AGAINST IT

``base.py`` makes the case that a CTC model is the right source of truth for
this system, because Whisper fabricates fluent text at high confidence on
silence and noise. That argument still stands. This exists anyway, for a
reason worth stating plainly: IndicConformer needs NeMo, NeMo needs torch,
and none of that installs on the Python here. Whisper via CTranslate2 needs
neither. It is the model that runs today.

So its confidence is treated as the weaker evidence it is, and two specific
mitigations are applied to its known failure mode:

**VAD filtering is on.** Silero runs ahead of the decoder and drops
non-speech, which removes most of the silence that Whisper hallucinates
over. This is the single highest-value setting on the whole integration.

**no_speech_prob discounts the whole segment.** When Whisper reports it was
probably not hearing speech, every word from that segment is scaled down.
The failure mode is *confident fluent invention*, so the only useful defence
is to distrust confidence that comes from a segment the model itself doubts
was speech at all.

Even so: a word probability from a sequence-to-sequence decoder is a
statement about the model's own output, not about the acoustics. It is
better than the no-metadata default the Bolna path falls back on, and worse
than a CTC posterior. Do not read the numbers as calibrated.
"""
from __future__ import annotations

import glob
import logging
import os
import site
import sys
from typing import Any

from .base import AudioRef, Transcriber, Transcript, Word

log = logging.getLogger(__name__)

DEFAULT_MODEL = "small"
_cuda_ready = False


def ensure_cuda_libraries() -> None:
    """Put the pip-installed CUDA DLLs where Windows can find them.

    CTranslate2 links cuBLAS and cuDNN at call time, not import time, so a
    missing DLL surfaces as a RuntimeError in the middle of the first
    transcription rather than when the model loads - "Library cublas64_12.dll
    is not found", after a clean load and a plausible-looking start.

    The nvidia-* wheels ship the DLLs inside site-packages rather than on
    PATH, so they have to be registered explicitly.
    """
    global _cuda_ready
    if _cuda_ready or not sys.platform.startswith("win"):
        _cuda_ready = True
        return

    # Located from the nvidia package itself, not site.getsitepackages():
    # inside a virtualenv that returns the *base* interpreter's directories,
    # so the glob matched nothing and the fix silently did nothing at all.
    roots: list[str] = []
    try:
        import nvidia

        roots = list(getattr(nvidia, "__path__", []))
    except ImportError:
        roots = [os.path.join(p, "nvidia") for p in site.getsitepackages()]

    for root in roots:
        for directory in glob.glob(os.path.join(root, "*", "bin")):
            try:
                os.add_dll_directory(directory)
            except (OSError, AttributeError):
                pass
            # add_dll_directory covers the loader; PATH covers libraries that
            # resolve dependencies by name at runtime, which cuDNN does.
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    _cuda_ready = True


class WhisperTranscriber(Transcriber):
    """Local Whisper. Falls back to CPU rather than failing."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        ensure_cuda_libraries()
        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:
            # A machine without a usable GPU still has to answer the phone.
            # int8 on CPU is several times slower and materially worse for
            # conversation, which is a degradation worth logging loudly.
            log.warning(
                "GPU unavailable (%s: %s); falling back to CPU int8, expect "
                "seconds per turn rather than fractions", type(exc).__name__, exc,
            )
            self.device, self.compute_type = "cpu", "int8"
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
        return self._model

    def transcribe(
        self, audio: AudioRef, language_hint: str | None = None
    ) -> Transcript | None:
        refusal = self.guard(audio)
        if refusal:
            log.warning("refusing to transcribe: %s", refusal)
            return None

        try:
            model = self._load()
            segments, info = model.transcribe(
                audio.uri,
                language=language_hint,
                word_timestamps=True,
                # Silero ahead of the decoder. Whisper's characteristic
                # failure is inventing fluent sentences over silence, and
                # never showing it the silence is the cheapest defence there
                # is.
                vad_filter=True,
                beam_size=1,
            )
            words: list[Word] = []
            for segment in segments:
                # The model's own estimate that this segment was not speech.
                # Hallucinations arrive with high word probabilities, so the
                # segment-level doubt is the only independent signal available.
                speech_factor = 1.0 - float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
                for word in (segment.words or []):
                    text = word.word.strip()
                    if not text:
                        continue
                    words.append(Word(
                        text=text,
                        confidence=max(0.0, min(1.0, float(word.probability) * speech_factor)),
                        start_s=float(word.start),
                        end_s=float(word.end),
                    ))
        except Exception as exc:
            log.warning("ASR failed: %s: %s", type(exc).__name__, exc)
            return None

        if not words:
            return None

        notes = [f"{self.model_size} on {self.device}"]
        if audio.is_narrowband:
            notes.append("narrowband 8 kHz audio; Whisper is trained at 16 kHz")
        return Transcript(
            words=words,
            language=(getattr(info, "language", None) or language_hint or "en"),
            model=f"faster-whisper-{self.model_size}",
            notes=notes,
        )
