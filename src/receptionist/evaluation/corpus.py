"""Ground-truth test corpus.

Each case pairs an utterance with the structured values it must produce.
Indic-script cases carry romanised phone digits and English procedure terms
deliberately: that is how these calls actually sound. Callers code-switch,
and a corpus of pure single-language sentences would test a population that
does not exist.
"""
from __future__ import annotations

from datetime import datetime

from .harness import TestCase

REFERENCE = datetime(2026, 7, 27, 9, 0)     # Monday

CASES: list[TestCase] = [
    TestCase(
        "en-01", "en",
        "Hello my name is Priya Menon I need a cleaning tomorrow at 3 pm "
        "my number is nine seven one five zero one two three four five six seven",
        {"patient_name": "Priya Menon", "procedure": "cleaning",
         "appointment_time": datetime(2026, 7, 28, 15, 0), "phone": "+971501234567"},
    ),
    TestCase(
        "en-02", "en",
        "This is Ahmed Al Rashid, I have tooth pain, can I come in on Wednesday morning? "
        "Call me on zero five five one two three four five six seven",
        {"patient_name": "Ahmed Al Rashid", "procedure": "checkup",
         "appointment_time": datetime(2026, 7, 29, 10, 0), "phone": "+971551234567"},
    ),
    TestCase(
        "en-03", "en",
        "I'm Sarah Thomas and I need a root canal, is 15/08 at 11am free? "
        "My mobile is nine one nine eight seven six five four three two one zero",
        {"patient_name": "Sarah Thomas", "procedure": "root_canal",
         "appointment_time": datetime(2026, 8, 15, 11, 0), "phone": "+919876543210"},
        note="date-then-time ordering - the day-of-month must not be read as the hour",
    ),
    TestCase(
        "ta-01", "ta",
        "வணக்கம், my name is Karthik Raman, எனக்கு cleaning வேண்டும் tomorrow morning, "
        "number nine seven one five zero one two three four five six seven",
        {"patient_name": "Karthik Raman", "procedure": "cleaning",
         "phone": "+971501234567"},
        note="Tamil script with code-switched English clinical terms",
    ),
    TestCase(
        "ml-01", "ml",
        "നമസ്കാരം, my name is Anjali Nair, എനിക്ക് filling വേണം tomorrow at 4 pm, "
        "number zero five zero one two three four five six seven",
        {"patient_name": "Anjali Nair", "procedure": "filling",
         "appointment_time": datetime(2026, 7, 28, 16, 0), "phone": "+971501234567"},
        note="Malayalam script, romanised digits",
    ),
    TestCase(
        "kn-01", "kn",
        "ನಮಸ್ಕಾರ, this is Suresh Gowda, ನನಗೆ extraction ಬೇಕು on Thursday at 5 pm, "
        "number nine one nine eight seven six five four three two one zero",
        {"patient_name": "Suresh Gowda", "procedure": "extraction",
         "appointment_time": datetime(2026, 7, 30, 17, 0), "phone": "+919876543210"},
        note="Kannada script",
    ),
    TestCase(
        "ml-02", "ml",
        "namaskaram enikku oru appointment venam, my name is Deepak Menon, "
        "whitening, tomorrow at 6 pm, number zero five five one two three four five six seven",
        {"patient_name": "Deepak Menon", "procedure": "whitening",
         "appointment_time": datetime(2026, 7, 28, 18, 0), "phone": "+971551234567"},
        note="fully romanised Malayalam - script detection cannot help here",
    ),
    TestCase(
        "hi-01", "hi",
        "नमस्ते, my name is Rohit Sharma, मुझे braces चाहिए, Saturday at 2 pm, "
        "number nine one nine eight seven six five four three two one zero",
        {"patient_name": "Rohit Sharma", "procedure": "braces",
         "appointment_time": datetime(2026, 8, 1, 14, 0), "phone": "+919876543210"},
    ),
    TestCase(
        "en-04", "en",
        "My name is Fatima Al Blooshi and I want whitening on Sunday evening, "
        "reach me on zero five six one two three four five six seven",
        {"patient_name": "Fatima Al Blooshi", "procedure": "whitening",
         "phone": "+971561234567"},
        note="vague time of day - must be flagged, not silently defaulted",
    ),
    TestCase(
        "en-05", "en",
        "Hi this is Mohammed, I need a checkup, my number is double nine "
        "eight seven six five four three two one and can I come tomorrow at 10",
        {"patient_name": "Mohammed", "procedure": "checkup",
         "appointment_time": datetime(2026, 7, 28, 10, 0)},
        note="'double nine' multiplier expansion",
    ),
]
