"""Telegram connector. Free of charge, and not a drop-in for WhatsApp.

WHY THIS EXISTS

Meta bills per template message from the first send, plus a BSP fee on top,
so AiSensy cannot be made free - only cheaper. Telegram's bot API has no
per-message cost. The ``MessagingConnector`` interface was written so the
channel could be swapped without touching the call flow, and this is that
claim being cashed rather than asserted.

THE DIFFERENCE THAT IS NOT COSMETIC

WhatsApp addresses a **phone number**. Telegram addresses a **chat_id**, and
a chat_id only exists once the patient has messaged the bot first. There is
no way to message an arbitrary number: it is not a rate limit or a
permission, it is the shape of the product.

So a clinic switching to Telegram has to get every patient to press Start
before it can confirm anything, and a patient who has not done that cannot
be reached at all. That is modelled here as an explicit, non-retryable
failure with a reason a receptionist can act on, rather than a send that
quietly goes nowhere. Silently dropping it would be the messaging-layer
version of the silent error this whole project is built around.

WHAT GETS SIMPLER

No pre-approved templates and no positional parameter array, so the entire
class of transposition bug the AiSensy connector guards against cannot occur
here - the body is rendered locally from the same per-language templates and
sent as text. What is lost with it is Meta's approval step, which is the only
thing that ever verified the wording before a patient saw it.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import MessagingConnector, OutboundMessage, render_template

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramError(RuntimeError):
    """Refused before the request left the process."""


class ChatDirectory:
    """Phone number to Telegram chat_id.

    Populated when a patient messages the bot - in production from the
    ``/start`` update, where Telegram supplies the chat_id and the patient can
    share their number via a contact button. Kept behind a class so the
    storage can become a real table without the connector noticing.
    """

    def __init__(self, links: dict[str, int] | None = None) -> None:
        self._links: dict[str, int] = dict(links or {})

    def link(self, phone: str, chat_id: int) -> None:
        self._links[_normalise(phone)] = chat_id

    def chat_id_for(self, phone: str) -> int | None:
        return self._links.get(_normalise(phone))

    def __len__(self) -> int:
        return len(self._links)


def _normalise(phone: str) -> str:
    """Compare numbers by digits alone.

    The calendar stores E.164 and a patient may have shared their number in
    any format. Matching on the raw string would fail on a leading plus and
    look, from the outside, exactly like a patient who never linked at all.
    """
    return "".join(ch for ch in str(phone or "") if ch.isdigit())


def build_message(message: OutboundMessage, directory: ChatDirectory) -> dict[str, Any]:
    """The sendMessage body, or raise with a reason worth reading."""
    chat_id = directory.chat_id_for(message.to)
    if chat_id is None:
        raise TelegramError(
            f"{message.to} has not started a chat with the clinic's Telegram bot; "
            "Telegram cannot message a phone number that has not opted in"
        )

    return {
        "chat_id": chat_id,
        # Rendered from the same per-language templates the WhatsApp path
        # uses, so the wording does not fork by channel.
        "text": render_template(message.template, message.language, message.parameters),
        # No parse_mode on purpose. Markdown would treat an underscore or
        # asterisk in a patient's name as formatting, and either mangle the
        # name or fail the send outright on unbalanced markup.
        "disable_notification": message.template == "review_request",
    }


class TelegramConnector(MessagingConnector):
    """Live connector. Not exercised by the test suite - see MockTelegram."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        directory: ChatDirectory | None = None,
        base_url: str = TELEGRAM_API,
        timeout_s: float = 10.0,
    ) -> None:
        if not bot_token:
            raise TelegramError("no Telegram bot token configured")
        self.bot_token = bot_token
        self.directory = directory or ChatDirectory()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send(self, message: OutboundMessage) -> dict[str, Any]:
        import httpx2

        body = build_message(message, self.directory)
        url = f"{self.base_url}/bot{self.bot_token}/sendMessage"
        try:
            response = httpx2.post(url, json=body, timeout=self.timeout_s)
        except httpx2.HTTPError as exc:
            return {"ok": False, "provider": self.name, "retryable": True,
                    "error": f"{type(exc).__name__}: {exc}"}

        payload = _safe_json(response)
        ok = response.status_code < 400 and bool(payload.get("ok"))
        return {
            "ok": ok,
            "provider": self.name,
            "status_code": response.status_code,
            # 429 carries retry_after and is worth retrying. 400 and 403 mean
            # the chat is gone or the bot was blocked, and retrying that is a
            # loop against a decision the patient has made.
            "retryable": response.status_code == 429 or response.status_code >= 500,
            "message_id": (payload.get("result") or {}).get("message_id"),
            "response": payload,
        }


class MockTelegram(MessagingConnector):
    """Records what would be sent, having built the real request body.

    Building the real body is the point: it is what exercises the chat_id
    lookup, so a patient who never linked fails here and in the demo rather
    than only against a live bot.
    """

    name = "telegram-mock"

    def __init__(self, directory: ChatDirectory | None = None) -> None:
        self.directory = directory or ChatDirectory()
        self.sent: list[OutboundMessage] = []
        self.requests: list[dict[str, Any]] = []

    def send(self, message: OutboundMessage) -> dict[str, Any]:
        body = build_message(message, self.directory)
        self.sent.append(message)
        self.requests.append(body)
        return {
            "ok": True,
            "provider": self.name,
            "chat_id": body["chat_id"],
            "text": body["text"],
            "message_id": len(self.sent),
        }


def _safe_json(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"raw": data}
    except Exception:
        return {"raw": response.text[:500]}
