"""Deployment settings, read from the environment.

Everything here has a safe default *except* the secrets, which have no
default at all. A secret that defaults to an empty string and is then treated
as "checking disabled" is how a webhook endpoint ends up unauthenticated in
production while passing every test. The checks that use these values fail
closed when they are unset - see ``telephony.bolna.verify_source``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .telephony.bolna import BOLNA_WEBHOOK_SOURCE_IPS

# Repository root: src/receptionist/config.py -> up three.
ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read .env into the process environment. Returns what it set.

    Exists because ``VOICE_ENABLED=1 uvicorn ...`` is bash syntax and this
    project is developed on Windows, where PowerShell reads it as a command
    name and fails with "is not recognized as the name of a cmdlet". Telling
    people to remember `$env:VOICE_ENABLED = "1"` before every run is how a
    demo ends up silently running without voice.

    A real environment variable always wins. The file is a default for local
    development, not an override of what an operator deliberately exported -
    the reverse would make a stale .env quietly beat production config.
    """
    path = path or ROOT / ".env"
    applied: dict[str, str] = {}
    if not path.is_file():
        return applied

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `export FOO=bar` is what people paste in from shell notes.
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        # Strip one matched pair of quotes, so a value with spaces survives.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
        applied[key] = value
    return applied


# Loaded before Settings is constructed: the dataclass defaults read
# os.environ at instantiation, so anything applied later would be ignored.
load_dotenv()


def _csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass
class Settings:
    # Shared secret Bolna sends back to us in a header. Bolna signs nothing,
    # so this is the only thing distinguishing a real delivery from anyone
    # who has guessed the URL.
    bolna_webhook_secret: str = field(
        default_factory=lambda: os.environ.get("BOLNA_WEBHOOK_SECRET", "")
    )
    # Extra source addresses, for a tunnel during development or a proxy in
    # front of the app. Additive to the documented Bolna egress address so a
    # local override cannot silently remove the real one.
    bolna_extra_source_ips: tuple[str, ...] = field(
        default_factory=lambda: _csv("BOLNA_EXTRA_SOURCE_IPS")
    )

    # "aisensy", "telegram", or "" to auto-detect from whichever credential
    # is present. Explicit beats inferred once there is more than one channel:
    # a stale AISENSY_API_KEY in the environment silently deciding where
    # patient messages go is not a thing anyone should have to debug.
    messaging_provider: str = field(
        default_factory=lambda: os.environ.get("MESSAGING_PROVIDER", "").strip().lower()
    )

    # Free of charge, unlike WhatsApp. Note Telegram addresses a chat_id, not
    # a phone number - see messaging/telegram.py.
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )

    aisensy_api_key: str = field(
        default_factory=lambda: os.environ.get("AISENSY_API_KEY", "")
    )
    aisensy_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AISENSY_BASE_URL", "https://backend.aisensy.com"
        )
    )

    # Clinic wall-clock offset from UTC. Bolna timestamps arrive UTC and
    # "tomorrow at 3pm" is meant in the clinic's local time; +4 is UAE.
    clinic_utc_offset_hours: float = field(
        default_factory=lambda: float(os.environ.get("CLINIC_UTC_OFFSET_HOURS", "4"))
    )
    # LLM second-opinion extractor. Off by default: it is an enhancement, and
    # a deployment that has not thought about sending transcripts to a remote
    # model should not start doing it because a package was installed.
    llm_crosscheck_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "LLM_CROSSCHECK_ENABLED", ""
        ).strip().lower() in ("1", "true", "yes")
    )
    # LLM as the primary understanding layer, with the rule extractor as the
    # fallback when it is unreachable.
    llm_extraction_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "LLM_EXTRACTION_ENABLED", ""
        ).strip().lower() in ("1", "true", "yes")
    )
    llm_extraction_model: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_EXTRACTION_MODEL", "gpt-oss:120b-cloud"
        )
    )
    # Transcripts carry patient names, numbers and complaints. Sending them
    # to a model on someone else's hardware is allowed, but has to be chosen:
    # build_model raises rather than letting it happen by default. Note that
    # a "-cloud" tag is remote even though it is served via localhost.
    llm_allow_remote: bool = field(
        default_factory=lambda: os.environ.get(
            "LLM_ALLOW_REMOTE", ""
        ).strip().lower() in ("1", "true", "yes")
    )

    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://localhost:11434")
    )

    # Re-transcribe the Bolna recording for per-word confidence. Off by
    # default: it needs a model checkpoint on disk, and a deployment without
    # one must keep working rather than fail to answer the phone.
    asr_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "ASR_ENABLED", ""
        ).strip().lower() in ("1", "true", "yes")
    )
    # Browser push-to-talk: local Whisper in, local Piper out. Off by default
    # because it needs model files on disk and a GPU to be conversational.
    voice_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "VOICE_ENABLED", ""
        ).strip().lower() in ("1", "true", "yes")
    )
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("WHISPER_MODEL", "small")
    )
    whisper_device: str = field(
        default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cuda")
    )
    piper_voice_dir: str = field(
        default_factory=lambda: os.environ.get("PIPER_VOICE_DIR", "models/piper")
    )

    asr_model: str = field(
        default_factory=lambda: os.environ.get(
            "ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual"
        )
    )

    clinic_name: str = field(
        default_factory=lambda: os.environ.get("CLINIC_NAME", "Al Noor Dental")
    )
    review_link: str = field(
        default_factory=lambda: os.environ.get(
            "REVIEW_LINK", "https://g.page/r/alnoor-dental/review"
        )
    )

    @property
    def bolna_allowed_ips(self) -> tuple[str, ...]:
        return BOLNA_WEBHOOK_SOURCE_IPS + self.bolna_extra_source_ips


settings = Settings()
