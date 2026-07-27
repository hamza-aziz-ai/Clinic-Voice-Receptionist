"""Download the speech models the voice loop needs.

    python scripts/setup_voice.py

Piper voices land in models/piper (about 190 MB for three languages).
Whisper is fetched by faster-whisper on first use into its own cache.

Neither is committed: they are large, and they are reproducible from here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "models" / "piper"

# One per language Piper can actually speak. Tamil and Kannada have no Piper
# voice at all - see src/receptionist/tts/piper_tts.py.
VOICES = ["en_US-lessac-medium", "hi_IN-pratham-medium", "ml_IN-meera-medium"]


def main() -> int:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(VOICES)} Piper voices into {VOICE_DIR}")
    subprocess.check_call(
        [sys.executable, "-m", "piper.download_voices", *VOICES,
         "--data-dir", str(VOICE_DIR)]
    )

    print("\nwarming Whisper (downloads the checkpoint on first run)")
    from faster_whisper import WhisperModel

    sys.path.insert(0, str(ROOT / "src"))
    from receptionist.asr.whisper_local import ensure_cuda_libraries

    ensure_cuda_libraries()
    try:
        WhisperModel("small", device="cuda", compute_type="float16")
        print("  GPU ready")
    except Exception as exc:
        print(f"  no usable GPU ({type(exc).__name__}); CPU will be several "
              f"times slower: {exc}")
    print("\nrun with: VOICE_ENABLED=1 uvicorn receptionist.api.main:app --app-dir src")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
