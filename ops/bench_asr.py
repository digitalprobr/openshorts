"""ASR benchmark: real-time factor (RTF) per backend/config on a sample file.

Usage:
    python ops/bench_asr.py path/to/sample.mp4 [--repeat N]

Verifies the speed work in docs/transcription_gemini_fix.md Part 1. Target:
RTF <= 0.1 on the prod GPU host (a 10 min video transcribes in under 60 s).
Runs each enabled configuration over the same media and prints one line per
config. Nothing is written; model downloads warm the HF cache on first run
(set HF_TOKEN to avoid unauthenticated rate limits).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transcribe_backends as tb  # noqa: E402


def bench(label, backend_env, whisper_env=None):
    """Transcribe the sample once under the given env override; print RTF."""
    saved = {k: os.environ.get(k) for k in
             ("TRANSCRIBE_BACKEND", "WHISPER_MODEL", "WHISPER_DEVICE",
              "WHISPER_COMPUTE", "WHISPER_BATCH_SIZE")}
    os.environ["TRANSCRIBE_BACKEND"] = backend_env
    for k, v in (whisper_env or {}).items():
        os.environ[k] = v
    try:
        started = time.time()
        transcript = tb.transcribe_media(SAMPLE)
        segments = transcript.get("segments") or []
        audio = float(segments[-1]["end"]) if segments else 0.0
        elapsed = time.time() - started
        rtf = (elapsed / audio) if audio else float("nan")
        print(f"{label:<38} {elapsed:8.1f}s wall  RTF {rtf:5.2f}  "
              f"lang={transcript.get('language')} segments={len(segments)}")
    except Exception as e:
        print(f"{label:<38} FAILED: {type(e).__name__}: {e}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", help="sample audio/video file")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per config (first run includes warmup)")
    args = parser.parse_args()
    if not os.path.exists(args.media):
        sys.exit(f"no such file: {args.media}")
    SAMPLE = args.media

    print(f"bench: {SAMPLE} ({tb._media_duration_seconds(SAMPLE):.0f}s), "
          f"{args.repeat} run(s) per config\n")
    for i in range(args.repeat):
        tag = f"[warmup {i}]" if i == 0 and args.repeat > 1 else ""
        bench(f"whisper cpu sequential {tag}".strip(), "whisper",
              {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE": "int8",
               "WHISPER_BATCH_SIZE": "1"})
        bench(f"whisper cpu batched {tag}".strip(), "whisper",
              {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE": "int8",
               "WHISPER_BATCH_SIZE": os.environ.get("WHISPER_BATCH_SIZE", "8")})
        bench(f"whisper cuda batched {tag}".strip(), "whisper",
              {"WHISPER_DEVICE": "cuda", "WHISPER_COMPUTE": "float16",
               "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "large-v3-turbo")})
        bench(f"parakeet (onnx-asr) {tag}".strip(), "parakeet")
