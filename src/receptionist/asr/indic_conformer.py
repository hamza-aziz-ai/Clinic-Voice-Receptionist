"""IndicConformer adapter. Not exercised by the test suite.

AI4Bharat's IndicConformer is a hybrid CTC-RNNT conformer covering all 22
scheduled Indian languages, MIT licensed, fine-tunable in NeMo. It is the
intended production implementation for three reasons, in order of importance
to this system:

1. **The CTC branch produces frame-level posteriors.** That is a real
   confidence signal, computed from acoustics, not a decoder's opinion of its
   own output. It is the input ``_asr_confidence`` was designed around and has
   never actually received.

2. **It fails loudly.** A CTC model given noise emits blanks and low-posterior
   fragments. Whisper given noise emits a fluent sentence at high confidence.
   The first is caught by the gate; the second walks straight through it.

3. It covers en, ta, kn, ml and hi in one model.

Whisper is not excluded - IndicWhisper has the better WER on most Indic
benchmarks - but it belongs behind ``nlu.crosscheck`` as a second opinion
that can only lower confidence, never as the transcript the booking is
derived from. Two recognisers with genuinely different architectures disagree
in informative ways; that is a far better cross-check than the LLM one, whose
errors were correlated with the rule extractor by construction.

THE NARROWBAND GAP

Every published number for these models is 16 kHz wideband. A phone call is
8 kHz, mu-law companded, with packet loss. Fine-tuning has to include that
degradation - downsample, codec round-trip, drop packets - or the deployed
error rate will be materially worse than the benchmark that justified the
choice, and worse specifically on digits, which is the slot that matters
most and the one nothing downstream can check.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import AudioRef, Transcriber, Transcript, Word

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"

# NeMo language codes for the five this clinic answers in.
LANGUAGE_CODES: dict[str, str] = {
    "en": "en", "hi": "hi", "ta": "ta", "kn": "kn", "ml": "ml",
}


class IndicConformerTranscriber(Transcriber):
    """Thin wrapper. Deliberately holds no logic the mock does not share."""

    name = "indic-conformer"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        """Loaded on first use, not at import.

        nemo_toolkit pulls in a deep dependency tree and the checkpoint is
        hundreds of megabytes. Importing at module scope would make the whole
        package - and the offline test suite - depend on both.
        """
        if self._model is None:
            import nemo.collections.asr as nemo_asr

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()
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
            # CTC decoding, not RNNT: the CTC branch is what exposes per-frame
            # posteriors. RNNT would give a slightly better transcript and no
            # usable confidence, which is the wrong trade for this system.
            hypotheses = model.transcribe(
                [audio.uri],
                batch_size=1,
                return_hypotheses=True,
                logprobs=False,
            )
        except Exception as exc:
            # No transcript means the ingest path keeps its existing
            # behaviour: no word confidences, so more slots fall below
            # threshold and more calls land in the callback queue. Degraded,
            # never wrong.
            log.warning("ASR unavailable: %s: %s", type(exc).__name__, exc)
            return None

        return _to_transcript(
            hypotheses, model_name=self.model_name,
            language=language_hint or "en", narrowband=audio.is_narrowband,
        )


def _to_transcript(
    hypotheses: Any, model_name: str, language: str, narrowband: bool
) -> Transcript | None:
    """Map NeMo hypotheses onto this system's Word list.

    Isolated so the NeMo-shaped part of the integration is one function. Their
    hypothesis object has changed shape across releases, and everything that
    knows about it should be replaceable in one edit.
    """
    hypothesis = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
    if hypothesis is None:
        return None

    words: list[Word] = []
    timestamps = getattr(hypothesis, "timestamp", None) or {}
    for entry in (timestamps.get("word") or []):
        text = entry.get("word") or entry.get("char") or ""
        if not text:
            continue
        words.append(Word(
            text=text,
            # NeMo exposes per-word confidence when confidence estimation is
            # enabled on the decoding config. Absent it, refuse to invent a
            # number - a fabricated confidence is worse than none, because the
            # gate would treat it as evidence.
            confidence=float(entry.get("confidence", 0.0)),
            start_s=float(entry.get("start", 0.0)),
            end_s=float(entry.get("end", 0.0)),
        ))

    if not words:
        return None

    notes = []
    if narrowband:
        notes.append("narrowband 8 kHz audio; model trained at 16 kHz")
    return Transcript(words=words, language=language, model=model_name, notes=notes)
