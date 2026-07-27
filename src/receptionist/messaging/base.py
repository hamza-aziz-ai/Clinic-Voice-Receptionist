"""WhatsApp messaging contracts (AiSensy-shaped).

Messages are template-based because WhatsApp Business requires pre-approved
templates for business-initiated conversations. That constraint is modelled
here rather than discovered in production: a free-text confirmation would be
rejected by the API, and finding that out after the call has ended means the
patient never hears from you.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

TemplateName = Literal["appointment_confirmation", "appointment_reminder", "review_request"]


@dataclass
class OutboundMessage:
    template: TemplateName
    to: str
    language: str
    parameters: dict[str, str]
    send_after: datetime | None = None
    booking_id: str | None = None
    # "expired" is distinct from "cancelled": the booking was fine, the send
    # window simply passed. A reminder that arrives after the appointment is
    # worse than no reminder, so it is dropped rather than sent late, and it
    # is recorded separately because a rising expired count means the
    # dispatcher is not running often enough.
    status: Literal["queued", "sent", "failed", "cancelled", "expired"] = "queued"
    result: dict[str, Any] | None = None

    def render(self) -> str:
        return f"[{self.template} → {self.to} ({self.language})] {self.parameters}"


TEMPLATES: dict[str, dict[str, str]] = {
    "appointment_confirmation": {
        "en": "Hello {name}, your {procedure} at {clinic} is confirmed for {when}. Reply CANCEL to cancel.",
        "ta": "வணக்கம் {name}, {clinic}-இல் உங்கள் {procedure} {when}-க்கு உறுதி செய்யப்பட்டது.",
        "kn": "ನಮಸ್ಕಾರ {name}, {clinic} ನಲ್ಲಿ ನಿಮ್ಮ {procedure} {when} ಕ್ಕೆ ದೃಢಪಡಿಸಲಾಗಿದೆ.",
        "ml": "നമസ്കാരം {name}, {clinic}-ൽ നിങ്ങളുടെ {procedure} {when}-ന് സ്ഥിരീകരിച്ചു.",
        "hi": "नमस्ते {name}, {clinic} में आपका {procedure} {when} के लिए निश्चित है।",
    },
    "appointment_reminder": {
        "en": "Reminder: {name}, your {procedure} at {clinic} is tomorrow at {when}.",
        "ta": "நினைவூட்டல்: {name}, {clinic}-இல் உங்கள் {procedure} நாளை {when}.",
        "kn": "ಜ್ಞಾಪನೆ: {name}, {clinic} ನಲ್ಲಿ ನಿಮ್ಮ {procedure} ನಾಳೆ {when}.",
        "ml": "ഓർമ്മപ്പെടുത്തൽ: {name}, {clinic}-ൽ നിങ്ങളുടെ {procedure} നാളെ {when}.",
        "hi": "अनुस्मारक: {name}, {clinic} में आपका {procedure} कल {when} बजे है।",
    },
    "review_request": {
        "en": "Thank you for visiting {clinic}, {name}. Would you leave us a Google review? {link}",
        "ta": "{clinic}-க்கு வருகை தந்ததற்கு நன்றி, {name}. Google விமர்சனம் அளிக்கவும்: {link}",
        "kn": "{clinic} ಗೆ ಭೇಟಿ ನೀಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು, {name}. Google ವಿಮರ್ಶೆ ನೀಡಿ: {link}",
        "ml": "{clinic} സന്ദർശിച്ചതിന് നന്ദി, {name}. Google റിവ്യൂ നൽകാമോ? {link}",
        "hi": "{clinic} पधारने के लिए धन्यवाद, {name}। कृपया Google समीक्षा दें: {link}",
    },
}


def render_template(template: TemplateName, language: str, params: dict[str, str]) -> str:
    """Render, falling back to English if the language is unavailable.

    The fallback is explicit and logged rather than silent. A patient
    receiving English when they called in Malayalam is a degraded experience
    that someone should be able to see and fix.
    """
    body = TEMPLATES[template].get(language) or TEMPLATES[template]["en"]
    return body.format(**params)


class MessagingConnector(abc.ABC):
    name: str

    @abc.abstractmethod
    def send(self, message: OutboundMessage) -> dict[str, Any]: ...


class MockWhatsApp(MessagingConnector):
    """Records instead of sending. What the console and tests inspect."""

    name = "aisensy-mock"

    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    def send(self, message: OutboundMessage) -> dict[str, Any]:
        body = render_template(message.template, message.language, message.parameters)
        self.sent.append(message)
        used_fallback = message.language not in TEMPLATES[message.template]
        return {
            "ok": True, "provider": self.name,
            "message_id": f"mock-{len(self.sent)}",
            "body": body,
            "language_fallback": used_fallback,
        }
