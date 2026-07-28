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
