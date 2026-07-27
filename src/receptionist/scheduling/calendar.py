"""Transactional appointment calendar.

Two calls can arrive at the same second and ask for the same 3pm slot. A
voice agent that checks availability and then books is not safe - between the
two operations another call can take the slot, and both callers are told yes.

So booking is a single atomic reserve-or-fail under a lock, with idempotency
keys so a webhook retry cannot create a duplicate. This is unglamorous and it
is the difference between a demo and something a clinic can answer the phone
with.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Literal

from ..nlu.normalize import spoken_time
from ..nlu.slots import PROCEDURE_DURATION_MIN

BookingStatus = Literal["confirmed", "cancelled", "no_show", "completed"]


@dataclass(frozen=True)
class ClinicHours:
    """Opening hours per weekday. Friday is the UAE weekend anchor."""

    open_time: time = time(9, 0)
    close_time: time = time(20, 0)
    closed_weekdays: tuple[int, ...] = (4,)          # Friday
    lunch_start: time = time(13, 0)
    lunch_end: time = time(14, 0)
    slot_granularity_min: int = 15

    def is_open(self, when: datetime, duration_min: int) -> tuple[bool, str]:
        if when.weekday() in self.closed_weekdays:
            return False, f"clinic is closed on {when:%A}"
        end = (when + timedelta(minutes=duration_min)).time()
        # These strings are spoken to the caller verbatim by the call flow, so
        # they carry the half of the day. "closing at 20:00" is unambiguous on
        # paper and unusable through a TTS voice.
        if when.time() < self.open_time:
            return False, f"the clinic opens at {spoken_time(self.open_time)}"
        if end > self.close_time:
            return False, (
                f"that would run past closing at {spoken_time(self.close_time)}"
            )
        if when.time() < self.lunch_end and end > self.lunch_start:
            return False, (
                f"that overlaps the break between {spoken_time(self.lunch_start)} "
                f"and {spoken_time(self.lunch_end)}"
            )
        if when.minute % self.slot_granularity_min:
            return False, f"appointments start on {self.slot_granularity_min}-minute boundaries"
        return True, "available"


@dataclass
class Booking:
    booking_id: str
    patient_name: str
    phone: str
    procedure: str
    start: datetime
    duration_min: int
    language: str
    status: BookingStatus = "confirmed"
    idempotency_key: str | None = None

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_min)

    def overlaps(self, other_start: datetime, other_duration: int) -> bool:
        other_end = other_start + timedelta(minutes=other_duration)
        return self.start < other_end and other_start < self.end


@dataclass
class BookingResult:
    ok: bool
    booking: Booking | None
    reason: str
    alternatives: list[datetime] = field(default_factory=list)
    # True when the idempotency key matched an existing booking. Callers need
    # this to distinguish "reserved just now" from "already reserved": the
    # booking is correct either way, but re-running the side effects that
    # follow a booking - the confirmation WhatsApp above all - would message
    # the patient twice for one appointment.
    replayed: bool = False


class Calendar:
    def __init__(self, hours: ClinicHours | None = None, chairs: int = 1) -> None:
        self.hours = hours or ClinicHours()
        self.chairs = chairs
        self._bookings: dict[str, Booking] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def active(self) -> list[Booking]:
        return [b for b in self._bookings.values() if b.status == "confirmed"]

    def get(self, booking_id: str) -> Booking | None:
        return self._bookings.get(booking_id)

    def _concurrent_at(self, start: datetime, duration: int) -> int:
        return sum(1 for b in self.active() if b.overlaps(start, duration))

    def suggest(self, around: datetime, duration_min: int, count: int = 3) -> list[datetime]:
        """Nearest bookable starts, searched outward from the requested time."""
        found: list[datetime] = []
        step = timedelta(minutes=self.hours.slot_granularity_min)
        for i in range(1, 200):
            for candidate in (around + step * i, around - step * i):
                if candidate < around.replace(hour=0, minute=0):
                    continue
                open_ok, _ = self.hours.is_open(candidate, duration_min)
                if open_ok and self._concurrent_at(candidate, duration_min) < self.chairs:
                    if candidate not in found:
                        found.append(candidate)
                if len(found) >= count:
                    return sorted(found)
        return sorted(found)

    # ------------------------------------------------------------------
    def book(
        self,
        patient_name: str,
        phone: str,
        procedure: str,
        start: datetime,
        language: str = "en",
        idempotency_key: str | None = None,
        duration_min: int | None = None,
    ) -> BookingResult:
        """Reserve atomically, or fail with alternatives. Never partially applies."""
        duration = duration_min or PROCEDURE_DURATION_MIN.get(procedure, 30)

        with self._lock:
            # Retry safety: the same key always returns the same booking.
            if idempotency_key and idempotency_key in self._by_key:
                existing = self._bookings[self._by_key[idempotency_key]]
                return BookingResult(
                    True, existing, "idempotent replay of an existing booking",
                    replayed=True,
                )

            open_ok, why = self.hours.is_open(start, duration)
            if not open_ok:
                return BookingResult(False, None, why, self.suggest(start, duration))

            if self._concurrent_at(start, duration) >= self.chairs:
                return BookingResult(
                    False, None,
                    # Also spoken verbatim. "all 2 chair(s) busy at 15:00" is
                    # a log line: a caller does not know or care how many
                    # chairs the clinic has, and "chair(s)" has no
                    # pronunciation.
                    f"we're fully booked at {spoken_time(start.time())}",
                    self.suggest(start, duration),
                )

            booking = Booking(
                booking_id=uuid.uuid4().hex[:10],
                patient_name=patient_name, phone=phone, procedure=procedure,
                start=start, duration_min=duration, language=language,
                idempotency_key=idempotency_key,
            )
            self._bookings[booking.booking_id] = booking
            if idempotency_key:
                self._by_key[idempotency_key] = booking.booking_id
            return BookingResult(True, booking, "booked")

    def cancel(self, booking_id: str) -> bool:
        with self._lock:
            b = self._bookings.get(booking_id)
            if not b or b.status != "confirmed":
                return False
            b.status = "cancelled"
            return True
