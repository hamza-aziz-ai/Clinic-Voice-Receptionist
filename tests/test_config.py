""".env loading.

Exists because `VOICE_ENABLED=1 uvicorn ...` is bash syntax, and on the
Windows machine this is developed on PowerShell reads it as a command name
and fails outright.
"""
import os

import pytest

from receptionist.config import load_dotenv


def write(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_values_are_read_into_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CLINIC_TEST_A", raising=False)
    applied = load_dotenv(write(tmp_path, "CLINIC_TEST_A=hello\n"))
    assert applied == {"CLINIC_TEST_A": "hello"}
    assert os.environ["CLINIC_TEST_A"] == "hello"


def test_a_real_environment_variable_always_wins(monkeypatch, tmp_path):
    """The file is a default for local development, not an override of what
    an operator deliberately exported - the reverse would let a stale .env
    quietly beat production config."""
    monkeypatch.setenv("CLINIC_TEST_B", "from-environment")
    load_dotenv(write(tmp_path, "CLINIC_TEST_B=from-file\n"))
    assert os.environ["CLINIC_TEST_B"] == "from-environment"


def test_comments_blanks_and_export_prefixes(tmp_path, monkeypatch):
    for key in ("CLINIC_TEST_C", "CLINIC_TEST_D"):
        monkeypatch.delenv(key, raising=False)
    applied = load_dotenv(write(tmp_path, """
# a comment
   # an indented comment

CLINIC_TEST_C=one
export CLINIC_TEST_D=two
"""))
    assert applied == {"CLINIC_TEST_C": "one", "CLINIC_TEST_D": "two"}


def test_quotes_are_stripped_and_spaces_survive(tmp_path, monkeypatch):
    monkeypatch.delenv("CLINIC_TEST_E", raising=False)
    monkeypatch.delenv("CLINIC_TEST_F", raising=False)
    load_dotenv(write(tmp_path, 'CLINIC_TEST_E="Al Noor Dental"\n'
                                "CLINIC_TEST_F='single quoted'\n"))
    assert os.environ["CLINIC_TEST_E"] == "Al Noor Dental"
    assert os.environ["CLINIC_TEST_F"] == "single quoted"


def test_a_value_containing_equals_is_not_truncated(tmp_path, monkeypatch):
    """Base64 secrets and URLs with query strings both contain '='."""
    monkeypatch.delenv("CLINIC_TEST_G", raising=False)
    load_dotenv(write(tmp_path, "CLINIC_TEST_G=abc==def?x=1\n"))
    assert os.environ["CLINIC_TEST_G"] == "abc==def?x=1"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_a_malformed_line_is_skipped_rather_than_crashing(tmp_path, monkeypatch):
    """A typo in .env must not stop the clinic answering the phone."""
    monkeypatch.delenv("CLINIC_TEST_H", raising=False)
    applied = load_dotenv(write(tmp_path, "this line has no equals sign\n"
                                          "CLINIC_TEST_H=fine\n"))
    assert applied == {"CLINIC_TEST_H": "fine"}


def test_the_example_documents_every_setting_the_code_reads():
    """A variable the code honours but the example never mentions is one
    nobody will discover."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / "src" / "receptionist" / "config.py").read_text(encoding="utf-8")
    example = (root / ".env.example").read_text(encoding="utf-8")

    referenced = set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"', source))
    referenced |= set(re.findall(r'_csv\(\s*"([A-Z0-9_]+)"', source))
    missing = sorted(name for name in referenced if name not in example)
    assert not missing, f"undocumented in .env.example: {missing}"


def test_the_example_holds_no_real_secret():
    """.env.example is committed; .env is not."""
    from pathlib import Path

    example = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if any(word in key for word in ("SECRET", "TOKEN", "KEY")):
            assert not value.strip(), f"{key} has a value in a committed file"


# ------------------------------------------------- configuration is external
# Settings reads os.environ in its default_factory, so a fresh Settings()
# picks up monkeypatched values. Reloading the config module instead would
# mint a new `settings` object while api.main still holds the old one, which
# quietly breaks every other test that patches it.
def test_clinic_hours_come_from_the_environment(monkeypatch):
    """Opening hours are the most clinic-specific thing here and were
    hardcoded in the scheduler, so a second clinic meant editing code."""
    from receptionist.config import Settings

    monkeypatch.setenv("CLINIC_OPEN_TIME", "08:30")
    monkeypatch.setenv("CLINIC_CLOSE_TIME", "17:00")
    monkeypatch.setenv("CLINIC_CLOSED_WEEKDAYS", "5,6")
    monkeypatch.setenv("CLINIC_CHAIRS", "7")
    s = Settings()

    assert (s.clinic_open_time.hour, s.clinic_open_time.minute) == (8, 30)
    assert s.clinic_close_time.hour == 17
    assert s.clinic_closed_weekdays == (5, 6)
    assert s.clinic_chairs == 7


def test_llm_and_voice_tuning_come_from_the_environment(monkeypatch):
    from receptionist.config import Settings

    monkeypatch.setenv("LLM_NUM_CTX", "8192")
    monkeypatch.setenv("LLM_KEEP_ALIVE", "5m")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("TTS_DEVICE", "cpu")
    s = Settings()

    assert s.llm_num_ctx == 8192
    assert s.llm_keep_alive == "5m"
    assert s.whisper_compute_type == "int8"
    assert s.tts_device == "cpu"


def test_a_malformed_value_falls_back_instead_of_crashing(monkeypatch):
    """A typo in .env must not stop the clinic answering the phone."""
    from receptionist.config import Settings

    monkeypatch.setenv("CLINIC_CHAIRS", "lots")
    monkeypatch.setenv("CLINIC_OPEN_TIME", "not-a-time")
    monkeypatch.setenv("CLINIC_CLOSED_WEEKDAYS", "banana")
    monkeypatch.setenv("LLM_NUM_CTX", "huge")
    s = Settings()

    assert s.clinic_chairs == 2
    assert s.clinic_open_time.hour == 9
    assert s.clinic_closed_weekdays == (4,)
    assert s.llm_num_ctx == 2048


def test_confidence_thresholds_are_tunable(monkeypatch):
    """The safety policy of the system, previously editable only in source."""
    from receptionist.nlu.slots import _threshold

    monkeypatch.setenv("THRESHOLD_PHONE", "0.99")
    assert _threshold("THRESHOLD_PHONE", 0.92) == 0.99
    monkeypatch.setenv("THRESHOLD_PHONE", "not-a-number")
    assert _threshold("THRESHOLD_PHONE", 0.92) == 0.92


def test_a_slot_without_a_threshold_still_raises():
    """Invariant: a missing key is a deliberate KeyError, not a default that
    silently books. There is no way to add a slot from the environment."""
    from receptionist.nlu.slots import CONFIRMATION_THRESHOLDS, Slot

    assert set(CONFIRMATION_THRESHOLDS) == {
        "phone", "appointment_time", "patient_name", "procedure"}
    rogue = Slot("patient_name", value="x", confidence=0.5)
    rogue.name = "invented_slot"
    with pytest.raises(KeyError):
        _ = rogue.needs_confirmation


def test_the_calendar_is_built_from_settings():
    """Wired, not merely available: the app must actually use the values."""
    from receptionist.api.main import calendar
    from receptionist.config import settings

    assert calendar.chairs == settings.clinic_chairs
    assert calendar.hours.open_time == settings.clinic_open_time
    assert calendar.hours.closed_weekdays == settings.clinic_closed_weekdays
