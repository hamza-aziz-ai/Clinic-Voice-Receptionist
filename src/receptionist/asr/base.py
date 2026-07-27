"""Speech recognition boundary: audio in, words with confidence out.

WHY THIS INTERFACE EXISTS AT ALL

The slot layer was built to average ASR word confidence over the span a value
came from. Bolna does not send that - its webhook carries a plain transcript
string - so on the live path every slot falls back to the no-metadata
default. The confidence model's strongest input has never actually been
connected to anything.

Bolna does send ``recording_url``. Re-transcribing that after the call gives
real per-word confidence without replacing the telephony, and latency does
not matter because the call has already ended.

WHY THE MODEL CHOICE IS NOT ABOUT WORD ERROR RATE

Whisper has the best multilingual WER available, and it is the wrong thing to
put behind this interface as the source of truth. It is sequence-to-sequence,
so its "confidence" is a decoder log-probability, and its documented failure
on silence or noise is to emit fluent, coherent, fabricated text at *high*
confidence - high avg_logprob, low no_speech_prob, straight past its own
internal filter.

That is precisely the failure this repository exists to prevent, relocated
one layer upstream where the read-back gate cannot see it. A CTC model fails
toward blanks and garbage with low posteriors: worse on paper, catchable in
practice. For a system whose entire claim is "it knows when it is unsure", a
loud failure mode beats a lower error rate.

So the intended implementation is a hybrid CTC-RNNT conformer (AI4Bharat's
IndicConformer covers all five languages here), with Whisper available as a
*second* opinion through the existing cross-check rather than as the value
that gets booked.

WHY CHANNEL SELECTION IS MANDATORY

A call recording contains both parties. Transcribing the mix and handing the
result to the extractor would let the agent's own speech supply confidence
for the caller's slots - and the agent says the name and the number aloud
during read-backs, clearly, at high confidence. The result would be a
confidence score for "Priya Menon" derived from the agent pronouncing it,
which is circular and would silently defeat the gate on the exact fields it
protects.

Telephony recordings are normally two-channel with the parties separated. If
the caller's channel cannot be identified, this layer returns nothing rather
than guessing.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Word:
    """One recognised word and how sure the recogniser was of it."""

    text: str
    confidence: float
    start_s: float = 0.0
    end_s: float = 0.0


@dataclass(frozen=True)
class AudioRef:
    """Where the audio is and how to read the caller out of it."""

    uri: str
    sample_rate_hz: int = 8000
    channels: int = 2
    # Which channel carries the caller. None means the recording is mixed and
    # the two parties cannot be separated - see the module docstring for why
    # that is refused rather than approximated.
    caller_channel: int | None = 0

    @property
    def is_separable(self) -> bool:
        return self.channels >= 2 and self.caller_channel is not None

    @property
    def is_narrowband(self) -> bool:
        """Telephony is 8 kHz. Every published Indic ASR benchmark is 16 kHz.

        Flagged rather than silently resampled, because the gap between those
        two numbers is most of the difference between a benchmark result and
        what a clinic will actually experience.
        """
        return self.sample_rate_hz < 16000


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def word_confidences(self) -> dict[str, float]:
        """The mapping ``extract_slots`` consumes.

        A word said more than once keeps its *lowest* confidence. The caller
        repeating a digit because the line was bad is evidence against that
        digit, not for it, and taking the best occurrence would let a single
        clear repetition paper over a value that was mostly unintelligible.
        """
        out: dict[str, float] = {}
        for word in self.words:
            key = word.text.lower().strip().strip(".,?!").strip()
            if not key:
                continue
            out[key] = min(out.get(key, 1.0), word.confidence)
        return out


class Transcriber(abc.ABC):
    """Audio to words. Returns None on anything it cannot do safely."""

    name: str

    @abc.abstractmethod
    def transcribe(
        self, audio: AudioRef, language_hint: str | None = None
    ) -> Transcript | None: ...

    def guard(self, audio: AudioRef) -> str | None:
        """Reasons to refuse before any model runs. None means proceed."""
        if not audio.uri:
            return "no recording available"
        if not audio.is_separable:
            return (
                "recording is not channel-separated; caller speech cannot be "
                "distinguished from the agent's own read-backs"
            )
        return None


class MockTranscriber(Transcriber):
    """Replays scripted results. What the suite and the demo run.

    Keyed by recording URI so a test can describe an entire call - including
    which words came out badly - without needing audio, a model, or a GPU.
    """

    name = "mock-asr"

    def __init__(self, scripted: dict[str, Transcript] | None = None) -> None:
        self.scripted = scripted or {}
        self.calls: list[AudioRef] = []

    def transcribe(
        self, audio: AudioRef, language_hint: str | None = None
    ) -> Transcript | None:
        self.calls.append(audio)
        refusal = self.guard(audio)
        if refusal:
            return None
        result = self.scripted.get(audio.uri)
        if result is None:
            return None
        if audio.is_narrowband and "narrowband" not in " ".join(result.notes):
            result.notes.append("narrowband 8 kHz audio; models are trained at 16 kHz")
        return result


def transcript_from_words(
    pairs: list[tuple[str, float]], language: str = "en", model: str = "mock-asr"
) -> Transcript:
    """Build a Transcript from (word, confidence) pairs. Test convenience."""
    return Transcript(
        words=[Word(text=t, confidence=c) for t, c in pairs],
        language=language,
        model=model,
    )
