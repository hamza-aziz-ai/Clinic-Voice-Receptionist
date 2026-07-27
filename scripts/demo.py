#!/usr/bin/env python3
"""End-to-end demo. No API keys, no network, no telephony account."""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from receptionist.evaluation.corpus import CASES, REFERENCE  # noqa: E402
from receptionist.evaluation.harness import evaluate, render_report  # noqa: E402
from receptionist.messaging.base import MockWhatsApp, render_template  # noqa: E402
from receptionist.nlu.language import LANGUAGE_NAMES, detect_language  # noqa: E402
from receptionist.scheduling.calendar import Calendar  # noqa: E402
from receptionist.workflow.call import CallHandler  # noqa: E402

RULE = "─" * 84
NOW = datetime(2026, 7, 27, 9, 0)


def banner(t): print(f"\n{RULE}\n  {t}\n{RULE}")


def run_call(handler, turns, wc=None, number="+971501112222"):
    s = handler.start(number)
    print(f"  agent : {s.transcript[-1].text}")
    for i, text in enumerate(turns):
        print(f"  caller: {text}")
        print(f"  agent : {handler.handle_utterance(s, text, NOW, wc if i == 0 else None)}")
    return s


def main() -> int:
    banner("1 · A CALL WHERE ASR IS CONFIDENT — books straight through")
    cal, wa = Calendar(chairs=2), MockWhatsApp()
    h = CallHandler(cal, wa)
    s = run_call(h, [
        "my name is Priya Menon I need a cleaning tomorrow at 3 pm "
        "my number is nine seven one five zero one two three four five six seven",
        "yes",
    ], wc={w: 0.97 for w in "priya menon cleaning tomorrow nine seven one five zero two three four six".split()})
    print(f"\n  booking {s.booking_id} · {len(s.messages)} WhatsApp messages queued")

    banner("2 · THE SAME CALL WITH WEAK ASR — every uncertain field is read back")
    cal2, wa2 = Calendar(chairs=2), MockWhatsApp()
    h2 = CallHandler(cal2, wa2)
    s2 = run_call(h2, [
        "my name is Priya Menon I need a cleaning tomorrow at 3 pm "
        "my number is nine seven one five zero one two three four five six seven",
        "yes", "yes", "yes",
    ], wc={"priya": 0.41, "menon": 0.38, "cleaning": 0.55, "nine": 0.62, "seven": 0.58})
    print("\n  slot confidence:")
    for slot in s2.slots.all_slots():
        state = "confirmed aloud" if slot.confirmed else ("read-back" if slot.needs_confirmation else "accepted")
        print(f"    {slot.name:17s} {str(slot.value)[:24]:26s} {slot.confidence:.2f}  {state}")

    banner("3 · MULTILINGUAL — language detected, WhatsApp sent in the same language")
    for utt in [
        "നമസ്കാരം, my name is Anjali Nair, എനിക്ക് filling വേണം tomorrow at 4 pm, "
        "number zero five zero one two three four five six seven",
        "வணக்கம், my name is Karthik Raman, எனக்கு cleaning வேண்டும் tomorrow at 5 pm, "
        "number zero five five one two three four five six seven",
    ]:
        d = detect_language(utt)
        cal3, wa3 = Calendar(chairs=2), MockWhatsApp()
        h3 = CallHandler(cal3, wa3)
        s3 = h3.start()
        for t in (utt, "yes", "yes", "yes", "yes"):
            h3.handle_utterance(s3, t, NOW)
        print(f"\n  detected {LANGUAGE_NAMES[d.language]} ({d.confidence:.2f}, {d.method})")
        if s3.messages:
            print(f"  WhatsApp: {render_template(s3.messages[0].template, s3.messages[0].language, s3.messages[0].parameters)}")

    banner("4 · CONCURRENCY — 20 simultaneous calls for one chair")
    from concurrent.futures import ThreadPoolExecutor
    cal4 = Calendar(chairs=1)
    slot = datetime(2026, 7, 28, 15, 0)
    with ThreadPoolExecutor(max_workers=20) as ex:
        res = list(ex.map(lambda i: cal4.book(f"P{i}", "+971501234567", "cleaning", slot), range(20)))
    print(f"  {sum(r.ok for r in res)} booked, {sum(not r.ok for r in res)} rejected with alternatives")
    print(f"  calendar holds {len(cal4.active())} appointment — no double-booking")
    a = cal4.book("R", "+919876543210", "checkup", datetime(2026, 7, 28, 17, 0), idempotency_key="k")
    b = cal4.book("R", "+919876543210", "checkup", datetime(2026, 7, 28, 17, 0), idempotency_key="k")
    print(f"  webhook retry returns the same booking: {a.booking.booking_id == b.booking.booking_id}")

    banner("5 · ACCURACY IN LANGUAGES I DO NOT SPEAK")
    print("  Every test utterance is paired with the values it must produce, so")
    print("  correctness is checkable without understanding a word.\n")
    print("  SILENT = wrong AND not flagged for read-back — the only kind a patient ever receives.\n")
    for sev, label in [(0.0, "clean transcript"), (0.15, "15% ASR error"),
                       (0.30, "30% ASR error"), (0.50, "50% ASR error")]:
        print("  " + render_report(evaluate(CASES, REFERENCE, sev, seed=11), label))
    print("\n  Degradation lowers confidence, which triggers read-backs, which catch")
    print("  the errors. The safety mechanism scales with the risk.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
