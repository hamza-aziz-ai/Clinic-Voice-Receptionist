"""Measure what the LLM cross-check actually buys, against the real model.

Not part of the test suite: it makes a network call per corpus case per
severity, which at the measured ~5s each is minutes of wall time and needs
Ollama running. The suite uses an injected fake instead.

    python scripts/eval_crosscheck.py

The comparison is rules-only versus rules-plus-cross-check on the same
corpus, same seed, same degraded transcripts. Two numbers matter:

  rescued        wrong values that would have gone through unflagged, and
                 are now flagged. The only figure that justifies the feature.
  false alarms   disagreements about values the rules got right. Each one is
                 a question asked of a caller who did not need asking.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from receptionist.evaluation.corpus import CASES, REFERENCE      # noqa: E402
from receptionist.evaluation.harness import evaluate             # noqa: E402
from receptionist.nlu.llm_extractor import (                     # noqa: E402
    DEFAULT_MODEL,
    build_chat_model,
    extract_llm_slots,
)

SEVERITIES = (0.0, 0.15, 0.30, 0.50)
SEED = 11


def main() -> int:
    print(f"corpus: {len(CASES)} utterances · model: {DEFAULT_MODEL}\n")

    try:
        model = build_chat_model()
    except Exception as exc:
        print(f"cannot reach Ollama: {type(exc).__name__}: {exc}")
        return 1

    calls = 0

    def crosscheck(text, reference_time, name, phone):
        nonlocal calls
        calls += 1
        return extract_llm_slots(text, reference_time, name, phone, chat_model=model)

    header = (
        f"{'ASR err':>7}  {'':<12} {'slots':>7} {'SILENT':>7} "
        f"{'flagged':>8} {'rescued':>8} {'false alarm':>12}"
    )
    print(header)
    print("-" * len(header))

    started = time.time()
    for severity in SEVERITIES:
        base = evaluate(CASES, REFERENCE, severity, seed=SEED)
        with_llm = evaluate(CASES, REFERENCE, severity, seed=SEED, crosscheck=crosscheck)

        for label, r in (("rules only", base), ("+ cross-check", with_llm)):
            counts = r.by_outcome()
            flagged = counts["correct_flagged"] + counts["wrong_caught"]
            rescued = r.rescued if label != "rules only" else 0
            alarms = r.disagreed_on_correct if label != "rules only" else 0
            print(
                f"{severity:>6.0%}  {label:<12} {r.slot_accuracy:>7.1%} "
                f"{counts['wrong_silent']:>7} {flagged:>8} {rescued:>8} "
                f"{alarms:>12}"
            )
        if with_llm.crosschecks_unavailable:
            print(f"{'':>7}  cross-check unavailable on "
                  f"{with_llm.crosschecks_unavailable}/{len(CASES)} cases")
        print()

    elapsed = time.time() - started
    print(f"{calls} model calls in {elapsed:.0f}s "
          f"({elapsed / calls:.1f}s each)" if calls else "no model calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
