"""Score the LLM extractor against the rules, on the same corpus.

    python scripts/eval_llm.py

Not part of the test suite: one model call per utterance per severity, so it
needs the model reachable and takes minutes. The suite injects fakes.

Until this existed the harness only scored the rule extractor, which meant
the accuracy table in the README described the *fallback* rather than the
path that actually runs. This closes that gap.

The number to read is SILENT - a wrong value that was not flagged for
read-back is the only kind that reaches a patient. Slot accuracy going up
while silent errors go up too would be a worse system, not a better one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from receptionist.config import settings                        # noqa: E402
from receptionist.evaluation.corpus import CASES, REFERENCE     # noqa: E402
from receptionist.evaluation.harness import evaluate            # noqa: E402
from receptionist.nlu.llm_slots import (                        # noqa: E402
    build_model,
    extract_slots_llm,
)

SEVERITIES = (0.0, 0.15, 0.30, 0.50)
SEED = 11

# Attempts per utterance before conceding the point to the fallback.
RETRIES = 3
BACKOFF_S = 4.0


def main() -> int:
    print(f"corpus: {len(CASES)} utterances · model: {settings.llm_extraction_model}\n")

    try:
        model = build_model(
            settings.llm_extraction_model,
            settings.llm_base_url,
            timeout_s=settings.llm_timeout_s,
            allow_remote=settings.llm_allow_remote,
            num_ctx=settings.llm_num_ctx,
            num_predict=settings.llm_num_predict,
            keep_alive=settings.llm_keep_alive,
        )
    except Exception as exc:
        print(f"cannot build the model: {type(exc).__name__}: {exc}")
        return 1

    calls = 0
    retries = 0

    def extractor(text, reference_time, word_confidences):
        """Retry, unlike production.

        The live path falls back to the rules the moment the model fails,
        which is right for a caller: the rules answer instantly and nobody is
        left listening to silence. Here the goal is the opposite - to measure
        the model - so a provider hiccup should cost seconds, not a data
        point silently attributed to the fallback.

        Ollama's cloud returns Internal Server Error in bursts while single
        calls succeed, which reads as rate limiting rather than an outage, so
        the backoff is linear and the pacing is deliberate.
        """
        nonlocal calls, retries
        for attempt in range(RETRIES):
            calls += 1
            slots = extract_slots_llm(text, reference_time, model, word_confidences)
            if slots is not None:
                return slots
            retries += 1
            time.sleep(BACKOFF_S * (attempt + 1))
        return None

    header = (
        f"{'ASR err':>7}  {'extractor':<12} {'slots':>7} {'lang':>7} "
        f"{'SILENT':>7} {'caught':>7} {'flagged':>8}  {'fell back':>9}"
    )
    print(header)
    print("-" * len(header))

    started = time.time()
    for severity in SEVERITIES:
        rules = evaluate(CASES, REFERENCE, severity, seed=SEED)
        llm = evaluate(CASES, REFERENCE, severity, seed=SEED, extractor=extractor)

        for label, report in (("rules", rules), ("llm", llm)):
            counts = report.by_outcome()
            flagged = counts["correct_flagged"] + counts["wrong_caught"]
            fell_back = (
                f"{report.extractor_unavailable}/{len(CASES)}"
                if label == "llm" else "-"
            )
            print(
                f"{severity:>6.0%}  {label:<12} {report.slot_accuracy:>7.1%} "
                f"{report.language_accuracy:>7.1%} {counts['wrong_silent']:>7} "
                f"{counts['wrong_caught']:>7} {flagged:>8}  {fell_back:>9}"
            )
        print()

    elapsed = time.time() - started
    print(f"{calls} model calls in {elapsed:.0f}s"
          + (f" ({elapsed / calls:.1f}s each)" if calls else "")
          + (f", {retries} retried after a provider error" if retries else ""))
    print("\nSILENT is the number that matters: wrong AND not flagged for "
          "read-back.\nHigher slot accuracy with more silent errors is a worse "
          "system, not a better one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
