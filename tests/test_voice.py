"""The local voice loop. No model is loaded here.

The adapters are exercised through injected fakes so the suite stays offline
and fast; the real Whisper and Piper runs are measured by hand and recorded
in the README. What is tested here is the wiring and, more importantly, the
refusals.
"""
import base64

import pytest

from receptionist.asr.base import AudioRef, transcript_from_words
from receptionist.tts.piper_tts import UNVOICED_LANGUAGES, VOICES, PiperSpeaker


# ---------------------------------------------------------------- TTS gaps
def test_tamil_and_kannada_have_no_voice_and_are_not_faked():
    """Piper has no model for either. An English voice fed Tamil script does
    not produce accented Tamil, it produces noise - and a caller hearing noise
    where their language should be is worse served than one who gets text."""
    speaker = PiperSpeaker()
    for language in ("ta", "kn"):
        assert language not in VOICES
        assert speaker.synthesize("test", language) is None


def test_the_gap_is_declared_not_incidental():
    assert set(UNVOICED_LANGUAGES) == {"ta", "kn"}
    assert set(VOICES) == {"en", "hi", "ml"}


def test_empty_text_is_not_synthesised():
    assert PiperSpeaker().synthesize("   ", "en") is None


def test_a_missing_voice_file_returns_none_rather_than_raising():
    """A silent reply is a degraded call. A raised exception during the reply
    would be a lost one, after the appointment is already booked."""
    speaker = PiperSpeaker(voice_dir="models/does-not-exist")
    assert not speaker.can_speak("en")
    assert speaker.synthesize("hello", "en") is None


# ---------------------------------------------------------------- mic audio
def test_a_browser_mic_capture_is_accepted_despite_being_mono():
    """The mixed-recording refusal exists because the agent's read-backs would
    supply confidence for the caller's own slots. A microphone stream contains
    only the caller - the agent plays through the speaker - so mono is safe
    here for exactly the reason it is not safe on a call recording."""
    mic = AudioRef(uri="turn.webm", channels=1, caller_channel=None,
                   single_speaker=True)
    assert mic.is_separable


def test_a_mixed_recording_is_still_refused():
    mixed = AudioRef(uri="call.wav", channels=1, caller_channel=None)
    assert not mixed.is_separable


# ---------------------------------------------------------------- endpoint
class FakeTranscriber:
    def __init__(self, transcript=None):
        self.transcript = transcript
        self.seen = []

    def transcribe(self, audio, language_hint=None):
        self.seen.append(audio)
        return self.transcript


class FakeSpeaker:
    def __init__(self, audio=b"RIFFfake"):
        self.audio = audio
        self.spoken = []

    def synthesize(self, text, language="en"):
        self.spoken.append((text, language))
        return self.audio


@pytest.fixture
def voice_app(monkeypatch):
    from receptionist.api import main
    from receptionist.config import settings

    monkeypatch.setattr(settings, "voice_enabled", True)
    words = [(w, 0.95) for w in
             "my name is priya menon i need a cleaning tomorrow at 3 pm".split()]
    transcriber = FakeTranscriber(transcript_from_words(words))
    speaker = FakeSpeaker()
    monkeypatch.setattr(main, "_voice", lambda: (transcriber, speaker))
    return main, transcriber, speaker


def test_voice_is_off_unless_enabled(monkeypatch):
    from fastapi.testclient import TestClient
    from receptionist.api.main import app
    from receptionist.config import settings

    monkeypatch.setattr(settings, "voice_enabled", False)
    client = TestClient(app)
    call = client.post("/calls", json={}).json()
    r = client.post(f"/calls/{call['call_id']}/audio",
                    files={"audio": ("t.webm", b"x", "audio/webm")})
    assert r.status_code == 503


def test_a_spoken_turn_reaches_the_booking_gate(voice_app):
    from fastapi.testclient import TestClient

    main, transcriber, speaker = voice_app
    client = TestClient(main.app)
    call = client.post("/calls", json={}).json()
    body = client.post(f"/calls/{call['call_id']}/audio",
                       files={"audio": ("t.webm", b"x", "audio/webm")}).json()

    assert "priya menon" in body["heard"].lower()
    assert body["reply"]
    assert base64.b64decode(body["audio"]) == b"RIFFfake"
    # The words the recogniser heard fed the confidence model, not a default.
    assert any(s["name"] == "procedure" and s["value"] == "cleaning"
               for s in body["slots"])


def test_the_mic_stream_is_marked_single_speaker(voice_app):
    from fastapi.testclient import TestClient

    main, transcriber, _ = voice_app
    client = TestClient(main.app)
    call = client.post("/calls", json={}).json()
    client.post(f"/calls/{call['call_id']}/audio",
                files={"audio": ("t.webm", b"x", "audio/webm")})
    assert transcriber.seen[0].single_speaker is True


def test_unintelligible_audio_says_so_rather_than_extracting_nothing(voice_app, monkeypatch):
    """Feeding empty text to the extractor would report that no details were
    captured, which reads as the caller having said nothing useful rather than
    the microphone having failed."""
    from fastapi.testclient import TestClient
    from receptionist.api import main

    monkeypatch.setattr(main, "_voice", lambda: (FakeTranscriber(None), FakeSpeaker()))
    client = TestClient(main.app)
    call = client.post("/calls", json={}).json()
    body = client.post(f"/calls/{call['call_id']}/audio",
                       files={"audio": ("t.webm", b"x", "audio/webm")}).json()
    assert body["heard"] == ""
    assert "didn't catch" in body["reply"]
    assert body["audio"] is None


def test_an_unknown_call_is_404(voice_app):
    from fastapi.testclient import TestClient

    main, _, _ = voice_app
    client = TestClient(main.app)
    r = client.post("/calls/nope/audio",
                    files={"audio": ("t.webm", b"x", "audio/webm")})
    assert r.status_code == 404
