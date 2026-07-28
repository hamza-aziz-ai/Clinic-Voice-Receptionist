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
from datetime import time
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


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _int(name: str, default: int) -> int:
    """Fall back rather than crash on a typo.

    A malformed number in .env should not stop the clinic answering the
    phone, and the default is always a working value.
    """
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _time(name: str, default: str) -> time:
    """HH:MM from the environment."""
    raw = (os.environ.get(name, "").strip() or default)
    try:
        hour, _, minute = raw.partition(":")
        return time(int(hour), int(minute or 0))
    except ValueError:
        hour, _, minute = default.partition(":")
        return time(int(hour), int(minute or 0))


def _weekdays(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Monday=0 ... Sunday=6, comma separated."""
    raw = _csv(name)
    if not raw:
        return default
    try:
        return tuple(int(v) for v in raw if 0 <= int(v) <= 6)
    except ValueError:
        return default


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
        default_factory=lambda: _flag("LLM_CROSSCHECK_ENABLED")
    )
    # LLM as the primary understanding layer, with the rule extractor as the
    # fallback when it is unreachable.
    llm_extraction_enabled: bool = field(
        default_factory=lambda: _flag("LLM_EXTRACTION_ENABLED")
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
        default_factory=lambda: _flag("LLM_ALLOW_REMOTE")
    )

    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://localhost:11434")
    )
    # Context window. The single biggest latency factor for a local model:
    # Ollama defaults qwen3 to 131072, which inflates a 2.5 GB model to 23 GB
    # and offloads most of it to the CPU. A system prompt and one utterance
    # fit in 2048 with room to spare.
    llm_num_ctx: int = field(default_factory=lambda: _int("LLM_NUM_CTX", 2048))
    # The cap stops a rambling model holding the turn hostage - but it has to
    # pay for the model's *hidden reasoning* as well as the answer, and that is
    # not optional on a reasoning model.
    #
    # THE FAILURE THIS DOCUMENTS. This was 256, sized against the four-field
    # object the model actually returns. gpt-oss ignores `think: false` and
    # reasons anyway; the chain of thought spent the whole 256-token budget, so
    # generation stopped before a single token of the answer. The result was an
    # empty completion - no content, no tool call, no error - which
    # `extract_slots_llm` reads as "model answered unusably" and silently falls
    # back to the rules. Extraction looked like a 9-in-11 provider outage and
    # was a one-line budget bug here. Measured over the corpus: 256 -> 0/4,
    # 1024 -> 4/4, 2048 -> 4/4. -1 (uncapped) is rejected by Ollama Cloud
    # outright, so the timeout, not an unbounded generation, is the guard.
    llm_num_predict: int = field(
        default_factory=lambda: _int("LLM_NUM_PREDICT", 1024)
    )
    # Keep the model resident between turns; reloading it per utterance would
    # dominate the latency budget.
    llm_keep_alive: str = field(
        default_factory=lambda: os.environ.get("LLM_KEEP_ALIVE", "30m")
    )
    # Bounds the worst case of a turn, so it is a conversational number rather
    # than a network one. The rules are a competent fallback and produce an
    # answer instantly; waiting 30 s for a model that may never reply is
    # strictly worse for the caller than falling back sooner.
    #
    # 20 s, not the 10 s this was. A full reasoning pass takes ~5.3 s and the
    # cloud endpoint swings, but the old number was worse than tight: it was
    # calibrated while every call was truncating at 256 tokens and returning
    # empty in under 3 s, so it was fitted to a broken response rather than a
    # working one. See `llm_num_predict`.
    llm_timeout_s: float = field(
        default_factory=lambda: _float("LLM_TIMEOUT_S", 20.0)
    )

    # Re-transcribe the Bolna recording for per-word confidence. Off by
    # default: it needs a model checkpoint on disk, and a deployment without
    # one must keep working rather than fail to answer the phone.
    asr_enabled: bool = field(
        default_factory=lambda: _flag("ASR_ENABLED")
    )
    # Browser push-to-talk: local Whisper in, local Piper out. Off by default
    # because it needs model files on disk and a GPU to be conversational.
    voice_enabled: bool = field(
        default_factory=lambda: _flag("VOICE_ENABLED")
    )
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("WHISPER_MODEL", "small")
    )
    whisper_device: str = field(
        default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cuda")
    )
    # float16 on a GPU, int8 on CPU. Mismatched to the device it either
    # refuses to load or runs far slower than it should.
    whisper_compute_type: str = field(
        default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    )
    piper_voice_dir: str = field(
        default_factory=lambda: os.environ.get("PIPER_VOICE_DIR", "models/piper")
    )
    # MMS covers Tamil and Kannada, which Piper has no voice for. Separate
    # from the Whisper device so the two can be split across CPU and GPU when
    # VRAM is tight.
    tts_device: str = field(
        default_factory=lambda: os.environ.get("TTS_DEVICE", "cuda")
    )

    asr_model: str = field(
        default_factory=lambda: os.environ.get(
            "ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual"
        )
    )

    # ---------------------------------------------------------------- hours
    # Opening hours are the most clinic-specific thing in the system and were
    # hardcoded in ClinicHours, so a second clinic meant editing the
    # scheduler. Weekdays are Monday=0; the default closure is Friday, the
    # UAE weekend anchor.
    clinic_open_time: time = field(
        default_factory=lambda: _time("CLINIC_OPEN_TIME", "09:00")
    )
    clinic_close_time: time = field(
        default_factory=lambda: _time("CLINIC_CLOSE_TIME", "20:00")
    )
    clinic_closed_weekdays: tuple[int, ...] = field(
        default_factory=lambda: _weekdays("CLINIC_CLOSED_WEEKDAYS", (4,))
    )
    clinic_lunch_start: time = field(
        default_factory=lambda: _time("CLINIC_LUNCH_START", "13:00")
    )
    clinic_lunch_end: time = field(
        default_factory=lambda: _time("CLINIC_LUNCH_END", "14:00")
    )
    clinic_slot_granularity_min: int = field(
        default_factory=lambda: _int("CLINIC_SLOT_GRANULARITY_MIN", 15)
    )
    # How many patients can be treated at once. Drives the booking conflict
    # check, so it is the difference between a full calendar and a
    # double-booked chair.
    clinic_chairs: int = field(default_factory=lambda: _int("CLINIC_CHAIRS", 2))

    # ------------------------------------------------------------ thresholds
    # The safety policy of the whole system, and previously only editable in
    # source. Below these a value is read back to the caller instead of
    # booked. Phone is highest because one wrong digit is still a valid
    # number, so nothing downstream can catch it.
    #
    # There is deliberately no way to add a slot here from the environment: a
    # slot with no threshold must raise, not inherit a default that silently
    # books. See nlu/slots.CONFIRMATION_THRESHOLDS.
    threshold_phone: float = field(
        default_factory=lambda: _float("THRESHOLD_PHONE", 0.92)
    )
    threshold_appointment_time: float = field(
        default_factory=lambda: _float("THRESHOLD_APPOINTMENT_TIME", 0.85)
    )
    threshold_patient_name: float = field(
        default_factory=lambda: _float("THRESHOLD_PATIENT_NAME", 0.75)
    )
    threshold_procedure: float = field(
        default_factory=lambda: _float("THRESHOLD_PROCEDURE", 0.70)
    )

    # ---------------------------------------------------------- conversation
    # How many times one slot may be asked before handing to a human, and how
    # many consecutive turns may add nothing. Progress, not elapsed turns, is
    # what decides whether a call is going anywhere.
    max_asks_per_slot: int = field(
        default_factory=lambda: _int("MAX_ASKS_PER_SLOT", 3)
    )
    max_turns_without_progress: int = field(
        default_factory=lambda: _int("MAX_TURNS_WITHOUT_PROGRESS", 4)
    )
    max_readback_failures: int = field(
        default_factory=lambda: _int("MAX_READBACK_FAILURES", 2)
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
