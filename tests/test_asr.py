"""The ASR boundary. Nothing here loads a model or touches audio."""
import pytest

from receptionist.asr.base import (
    AudioRef,
    MockTranscriber,
    Transcript,
    Word,
    transcript_from_words,
)
from receptionist.messaging.base import MockWhatsApp
from receptionist.scheduling.calendar import Calendar, ClinicHours
from receptionist.telephony.bolna import parse_execution
from receptionist.telephony.ingest import ingest_execution
from receptionist.workflow.call import CallHandler

from datetime import datetime

NOW = datetime(2026, 7, 27, 10, 0)
RECORDING = "https://api.bolna.ai/recordings/call/abc123"

BOOKED_CALL = (
    "assistant: Thank you for calling.\n"
    "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm "
    "my number is 0501234567\n"
    "assistant: Let me confirm your number. Is that correct?\n"
    "user: yes correct\n"
)


def payload(**over) -> dict:
    body = {
        "id": "exec-asr-1", "agent_id": "a", "status": "completed",
        "transcript": BOOKED_CALL,
        "created_at": "2026-07-27T06:00:00Z",
        "telephony_data": {"from_number": "+971501234567",
                           "recording_url": RECORDING},
    }
    body.update(over)
    return body


def handler():
    return CallHandler(Calendar(hours=ClinicHours(), chairs=2), MockWhatsApp())


# ---------------------------------------------------------------- confidences
def test_repeated_words_keep_their_lowest_confidence():
    """A caller repeating a digit because the line was bad is evidence
    against it. Taking the best occurrence would let one clear repetition
    paper over a value that was mostly unintelligible."""
    t = transcript_from_words([("five", 0.94), ("five", 0.31), ("five", 0.88)])
    assert t.word_confidences()["five"] == pytest.approx(0.31)


def test_confidences_are_case_and_punctuation_insensitive():
    """extract_slots looks words up lowercased and stripped."""
    t = transcript_from_words([("Priya", 0.8), ("Menon,", 0.7)])
    assert set(t.word_confidences()) == {"priya", "menon"}


def test_empty_words_are_ignored():
    t = Transcript(words=[Word("", 0.9), Word("  ", 0.9), Word("ok", 0.5)])
    assert t.word_confidences() == {"ok": 0.5}


# ---------------------------------------------------------------- refusals
def test_a_mixed_recording_is_refused_not_approximated():
    """The agent says the name and number aloud during read-backs. Scoring
    the caller's slots off the agent's own pronunciation is circular and
    defeats the gate on exactly the fields it protects."""
    mono = AudioRef(uri=RECORDING, channels=1, caller_channel=None)
    assert not mono.is_separable
    transcriber = MockTranscriber({RECORDING: transcript_from_words([("hi", 0.9)])})
    assert transcriber.transcribe(mono) is None


def test_a_missing_recording_is_refused():
    transcriber = MockTranscriber()
    assert transcriber.transcribe(AudioRef(uri="")) is None


def test_telephony_audio_is_flagged_as_narrowband():
    """Every published Indic ASR benchmark is 16 kHz; a phone call is 8 kHz,
    and that gap is most of the difference between the benchmark and reality."""
    assert AudioRef(uri=RECORDING, sample_rate_hz=8000).is_narrowband
    assert not AudioRef(uri=RECORDING, sample_rate_hz=16000).is_narrowband

    transcriber = MockTranscriber({RECORDING: transcript_from_words([("hi", 0.9)])})
    result = transcriber.transcribe(AudioRef(uri=RECORDING, sample_rate_hz=8000))
    assert any("narrowband" in n for n in result.notes)


# ---------------------------------------------------------------- ingest
def test_without_a_transcriber_nothing_changes():
    """The ASR path is additive. Absent it, the system behaves exactly as it
    did when Bolna's confidence-free transcript was all there was."""
    a = ingest_execution(parse_execution(payload()), handler(), now=NOW)
    b = ingest_execution(parse_execution(payload()), handler(), now=NOW,
                         transcriber=MockTranscriber())
    assert a.outcome == b.outcome


def test_confident_audio_feeds_the_gate():
    words = [(w, 0.97) for w in
             "my name is priya menon i need a cleaning tomorrow at 3 pm "
             "my number is 0501234567 yes correct".split()]
    transcriber = MockTranscriber({RECORDING: transcript_from_words(words)})
    h = handler()
    result = ingest_execution(parse_execution(payload()), h, now=NOW,
                              transcriber=transcriber)
    assert result.outcome == "booked", result.reason
    assert transcriber.calls[0].uri == RECORDING


# A call where the agent never read anything back, so nothing is confirmed
# aloud and the outcome rests entirely on how well the audio was heard. With
# Bolna's confidence-free transcript this could only ever land in the callback
# queue; connected to real ASR confidence it can also book.
UNCONFIRMED_CALL = (
    "assistant: Thank you for calling.\n"
    "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm "
    "my number is 0501234567\n"
)

_SPOKEN = ("my name is priya menon i need a cleaning tomorrow at 3 pm "
           "my number is 0501234567").split()


def test_clear_audio_can_book_without_a_readback():
    """0.97 across the utterance clears the phone threshold on its own."""
    transcriber = MockTranscriber({
        RECORDING: transcript_from_words([(w, 0.97) for w in _SPOKEN])})
    result = ingest_execution(
        parse_execution(payload(transcript=UNCONFIRMED_CALL)), handler(),
        now=NOW, transcriber=transcriber)
    assert result.outcome == "booked", result.reason


def test_poor_audio_on_the_digits_blocks_the_same_booking():
    """Same transcript, same words, only the acoustic confidence differs -
    and the appointment does not get made. This is the signal the confidence
    model was designed around and has never actually received."""
    words = [(w, 0.97) for w in _SPOKEN if w != "0501234567"]
    words.append(("0501234567", 0.22))
    transcriber = MockTranscriber({RECORDING: transcript_from_words(words)})
    h = handler()
    result = ingest_execution(
        parse_execution(payload(transcript=UNCONFIRMED_CALL)), h,
        now=NOW, transcriber=transcriber)
    assert result.outcome == "needs_callback"
    assert "phone" in result.unresolved
    assert h.calendar.active() == []


def test_a_failing_transcriber_degrades_rather_than_breaks():
    class Broken(MockTranscriber):
        def transcribe(self, audio, language_hint=None):
            raise RuntimeError("model not downloaded")

    result = ingest_execution(parse_execution(payload()), handler(), now=NOW,
                              transcriber=Broken())
    assert result.outcome in ("booked", "needs_callback")


def test_an_execution_with_no_recording_skips_asr():
    body = payload()
    body["telephony_data"].pop("recording_url")
    transcriber = MockTranscriber({RECORDING: transcript_from_words([("hi", 0.9)])})
    ingest_execution(parse_execution(body), handler(), now=NOW,
                     transcriber=transcriber)
    assert transcriber.calls == [], "no recording means nothing to transcribe"


def test_narrowband_warning_reaches_the_session():
    words = [(w, 0.97) for w in "my name is priya menon".split()]
    transcriber = MockTranscriber({RECORDING: transcript_from_words(words)})
    result = ingest_execution(parse_execution(payload()), handler(), now=NOW,
                              transcriber=transcriber)
    assert any("narrowband" in n for n in result.session.transcript_notes)
