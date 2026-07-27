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

from .telephony.bolna import BOLNA_WEBHOOK_SOURCE_IPS


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

    aisensy_api_key: str = field(
        default_factory=lambda: os.environ.get("AISENSY_API_KEY", "")
    )
    aisensy_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AISENSY_BASE_URL", "https://backend.aisensy.com"
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
