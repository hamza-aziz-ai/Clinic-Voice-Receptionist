"""Sending the messages that were scheduled for later.

The confirmation goes out while the call is still fresh. The reminder and the
review request do not - they are queued with a ``send_after`` and something
has to come back for them, which is the part that is easy to leave as a TODO
and then discover missing when a clinic asks why nobody was reminded.

Three rules, all of them about not sending:

**A cancelled appointment cancels its messages.** The booking status is read
at dispatch time, not at queue time. A patient who cancelled on Monday and
gets "your appointment is tomorrow" on Thursday has been told something
false by a system they cannot argue with.

**A missed window expires rather than sends late.** "Your appointment is
tomorrow at 3pm", delivered the day after the appointment, is worse than
silence: it is confusing, it looks broken, and it invites the patient to turn
up on a day they are not booked. Expired is counted separately from cancelled
because a rising expired count means the dispatcher is not running often
enough, which is an operational fault rather than a patient decision.

**A transport failure stays queued; a rejection does not.** Retrying a 500 is
free. Retrying a rejected template is an infinite loop against an API that
will never accept it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from ..scheduling.calendar import Calendar
from .base import MessagingConnector, OutboundMessage

# How long after its scheduled time a message may still go out. Measured
# from the appointment, not from send_after, because what makes a reminder
# worthless is the appointment having happened.
REMINDER_VALID_UNTIL_APPOINTMENT = True
REVIEW_REQUEST_WINDOW = timedelta(days=7)


@dataclass
class DispatchReport:
    sent: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    retrying: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sent": len(self.sent), "cancelled": len(self.cancelled),
            "expired": len(self.expired), "failed": len(self.failed),
            "retrying": len(self.retrying),
            "detail": {
                "sent": self.sent, "cancelled": self.cancelled,
                "expired": self.expired, "failed": self.failed,
                "retrying": self.retrying,
            },
        }


def is_due(message: OutboundMessage, now: datetime) -> bool:
    return (
        message.status == "queued"
        and message.send_after is not None
        and message.send_after <= now
    )


def is_expired(message: OutboundMessage, calendar: Calendar, now: datetime) -> bool:
    """Has the moment this message was useful passed?"""
    booking = calendar.get(message.booking_id) if message.booking_id else None
    if booking is None:
        return False

    if message.template == "appointment_reminder" and REMINDER_VALID_UNTIL_APPOINTMENT:
        return now >= booking.start
    if message.template == "review_request":
        return now > booking.start + REVIEW_REQUEST_WINDOW
    return False


def dispatch_due(
    messages: Iterable[OutboundMessage],
    connector: MessagingConnector,
    calendar: Calendar,
    now: datetime | None = None,
) -> DispatchReport:
    """Send everything due, and record why anything else was not sent."""
    now = now or datetime.now()
    report = DispatchReport()

    for message in messages:
        if not is_due(message, now):
            continue

        label = f"{message.template}:{message.booking_id or '-'}"

        booking = calendar.get(message.booking_id) if message.booking_id else None
        if booking is not None and booking.status == "cancelled":
            message.status = "cancelled"
            message.result = {"skipped": "booking cancelled"}
            report.cancelled.append(label)
            continue

        if is_expired(message, calendar, now):
            message.status = "expired"
            message.result = {"skipped": "send window passed"}
            report.expired.append(label)
            continue

        try:
            result = connector.send(message)
        except Exception as exc:
            # A refusal built by the connector itself - an undeclared campaign,
            # a missing parameter. Never retryable: the payload is wrong and
            # will be equally wrong next time.
            message.status = "failed"
            message.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            report.failed.append(label)
            continue

        message.result = result
        if result.get("ok"):
            message.status = "sent"
            report.sent.append(label)
        elif result.get("retryable"):
            message.status = "queued"          # left for the next pass
            report.retrying.append(label)
        else:
            message.status = "failed"
            report.failed.append(label)

    return report
