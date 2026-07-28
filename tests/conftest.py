"""Pin a hermetic environment before anything imports the app.

``config`` loads .env at import time, which is what makes `uvicorn` work
without a shell prefix - and it also meant the test suite inherited whatever
the developer had switched on locally. With LLM_CROSSCHECK_ENABLED=1 in .env
the suite went from 3 seconds to 42 and started making real network calls to
a language model, which is not a test suite, it is a weather report.

A real environment variable beats .env, so setting these here disables the
optional integrations regardless of local configuration. conftest is imported
before the test modules, so this runs before `receptionist.config` exists.

Anything a test needs switched on, it switches on itself with monkeypatch.
"""
import os

# Empty string rather than deletion: the loader skips keys already present in
# os.environ, so an empty value is what actually blocks the file's version.
for name in (
    "LLM_CROSSCHECK_ENABLED",   # network call per turn
    "ASR_ENABLED",              # would load a NeMo checkpoint
    "VOICE_ENABLED",            # would load Whisper and Piper
    "LLM_EXTRACTION_ENABLED",   # network call per turn
    "MESSAGING_PROVIDER",       # tests pick their own connector
    "TELEGRAM_BOT_TOKEN",
    "AISENSY_API_KEY",
    "BOLNA_WEBHOOK_SECRET",
    "BOLNA_EXTRA_SOURCE_IPS",
):
    os.environ.setdefault(name, "")
    os.environ[name] = ""
