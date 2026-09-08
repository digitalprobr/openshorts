"""Transcription backends: NVIDIA Parakeet (onnx-asr) with faster-whisper fallback.

Every caller goes through transcribe_media(), which returns the transcript
contract the whole pipeline depends on:

    {
      "text": str,          # full punctuated transcript
      "language": str,      # whisper-style short code ("es", "en", ...)
      "segments": [
        {"start": float, "end": float, "text": str,
         "words": [{"word": str, "start": float, "end": float}, ...]},
      ],
    }

Invariants the consumers rely on (clip cutting, karaoke subtitles, Remotion):
  - word["word"] carries a LEADING SPACE on true word starts; continuation
    fragments are merged into their base word (merge_continuation_words).
  - all numerics are native Python floats (json.dump of the transcript).
  - words sorted by start, segments chronological, absolute file timestamps.

TRANSCRIBE_BACKEND env: "whisper" (default) | "parakeet".
The parakeet path falls back to whisper automatically when the model errors,
produces no usable words, or the detected language is outside its 25
supported European languages (e.g. Japanese/Chinese/Arabic uploads).
GPU whisper in turn falls back to CPU whisper on CUDA errors (VRAM is shared
with other models on the host, so loads can OOM under load).
"""
import os
import subprocess
import tempfile
import threading
import time

from subtitles import (
    get_whisper_config,
    WHISPER_TRANSCRIBE_PARAMS,
    merge_continuation_words,
)

PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# The 25 European languages parakeet-tdt-0.6b-v3 supports (ISO 639-1).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Serializes GPU transcription across concurrent jobs so N jobs can't stack
# N model contexts / decode batches in VRAM. CPU whisper stays ungated
# (CTranslate2 models are thread-safe and that matches the old behavior).
_ASR_GATE = threading.Semaphore(int(os.environ.get("ASR_GPU_CONCURRENCY", "1")))


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


class _TranscribeProgress:
    """Emits '🎙️ Transcribing… NN% (Xs)' lines at 1% steps.

    These are the only transcription lines cloud users see (log_view keeps
    them), so they must stay free of technical detail. 1% steps (instead of
    the coarser 25% used before) make it obvious from the log whether the
    decode is moving or stuck — gaps between consecutive timestamps are
    small enough to judge progress in near-real-time.
    """

    def __init__(self, total_seconds):
        self.total = max(float(total_seconds or 0), 0.0)
        self.started = time.time()
        self.next_pct = 1

    def update(self, position_seconds):
        if self.total <= 0:
            return
        pct = min(int(position_seconds / self.total * 100), 100)
        while pct >= self.next_pct and self.next_pct <= 100:
            elapsed = int(time.time() - self.started)
            print(f"🎙️ Transcribing… {self.next_pct}% ({elapsed}s)", flush=True)
            self.next_pct += 1

# --- whisper singleton ------------------------------------------------------

_whisper_model = None
_whisper_key = None
_whisper_pipeline = None  # BatchedInferencePipeline wrapper (lazy)
_whisper_lock = threading.Lock()
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
_whisper_force_cpu = False


def _get_whisper_model():
    """Process-wide WhisperModel singleton, rebuilt if the env config changes.

    Keeping the model resident avoids a full reload per transcription (which
    on GPU would also mean re-allocating a couple of GB of VRAM per job).
    """
    global _whisper_model, _whisper_key, _whisper_pipeline
    cfg = get_whisper_config()
    if _whisper_force_cpu:
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"
    key = (cfg["model_size"], cfg["device"], cfg["compute_type"])
    with _whisper_lock:
        if _whisper_model is None or _whisper_key != key:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _whisper_key = key
            _whisper_pipeline = None  # any cached batched wrapper points at the old model
    return _whisper_model, cfg["device"]


def _whisper_batch_size():
    """Batched-decode width: ``WHISPER_BATCH_SIZE`` (default 8, 1 disables)."""
    raw = os.environ.get("WHISPER_BATCH_SIZE", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _get_batched_pipeline(model):
    """BatchedInferencePipeline around the singleton model (None if unavailable).

    Batched decode is the single biggest code-side speedup (~3-4x on GPU):
    VAD speech chunks are transcribed in parallel instead of segment by
    segment, with the same word-timestamped output contract.
    """
    global _whisper_pipeline
    if _whisper_pipeline is None:
        try:
            from faster_whisper import BatchedInferencePipeline
        except (ImportError, AttributeError):
            return None  # install without the batched pipeline / test fake
        _whisper_pipeline = BatchedInferencePipeline(model=model)
    return _whisper_pipeline


def _media_duration_seconds(media_path) -> float:
    """Media duration in seconds via ffprobe (0.0 when it can't be determined)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(media_path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _run_whisper_once(media_path, **params):
    model, device = _get_whisper_model()
    batch = _whisper_batch_size()
    # Batched decode is only trusted on longer audio: very short files hit
    # batch edge cases (sub-second chunks, empty VAD output). Unknown
    # duration (0.0) also takes the sequential path.
    if batch > 1 and device != "cpu" and _media_duration_seconds(media_path) >= 60:
        pipeline = _get_batched_pipeline(model)
        if pipeline is not None:
            params = {**params, "batch_size": batch}
            model = pipeline
    gate = _ASR_GATE if device != "cpu" else _NULL_GATE
    with gate:
        segments, info = model.transcribe(media_path, **params)
        progress = _TranscribeProgress(getattr(info, "duration", 0))
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line.
        progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, **params):
    """Transcribe and FULLY materialize the segments inside the GPU gate.

    faster-whisper returns a lazy generator — decoding happens while
    iterating, so the gate must wrap list(segments), not just transcribe().
    Returns (segments_list, info).

    A CUDA failure (model load OOM or mid-decode) retries once on CPU and
    pins CPU for the rest of the process — the GPU is shared with other
    models, so a job must degrade instead of dying when VRAM runs out.
    """
    global _whisper_model, _whisper_force_cpu, _whisper_pipeline
    try:
        return _run_whisper_once(media_path, **params)
    except RuntimeError as e:
        if _whisper_force_cpu or "cuda" not in str(e).lower():
            raise
        print(f"⚠️ [ASR] whisper GPU failed ({e}) — retrying on CPU", flush=True)
        _whisper_force_cpu = True
        with _whisper_lock:
            _whisper_model = None  # drop the GPU model to release its VRAM
            _whisper_pipeline = None
        return _run_whisper_once(media_path, **params)


def _transcribe_with_whisper(media_path):
    # When the shared 16 kHz wav was already extracted (parakeet attempt or a
    # retry in this process), reuse it: decoding PCM is cheaper than
    # demuxing + decoding the original container inside the GPU gate.
    shared = _current_shared_wav_for(media_path)
    if shared:
        media_path = shared
    segments, info = run_whisper_transcription(media_path, **WHISPER_TRANSCRIBE_PARAMS)

    out_segments = []
    text_parts = []
    for segment in segments:
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in (segment.words or [])
        ]
        out_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": merge_continuation_words(words),
        })
        text_parts.append(segment.text.strip())

    return {
        "text": " ".join(part for part in text_parts if part),
        "language": info.language,
        "segments": out_segments,
    }


# --- parakeet ---------------------------------------------------------------

_parakeet_model = None
_parakeet_lock = threading.Lock()


def _get_parakeet_model():
    global _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is None:
            import onnx_asr
            model = onnx_asr.load_model(
                PARAKEET_MODEL_ID,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            vad = onnx_asr.load_vad("silero")
            _parakeet_model = model.with_vad(vad).with_timestamps()
    return _parakeet_model


def _extract_wav(media_path):
    """Parakeet wants 16kHz mono PCM wav; ffmpeg-extract to a temp file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)
    return wav_path


# --- shared 16 kHz wav extraction (extract once, reuse across backends) ------

_shared_wav_media = None   # (abs path, mtime) the cached wav was extracted from
_shared_wav_path = None


def _get_shared_wav(media_path):
    """16 kHz mono PCM wav for ``media_path``, extracted once and cached.

    Parakeet needs this format anyway, and handing the same wav to whisper
    (instead of the original container) skips a second decode of the media
    inside the serialized GPU gate. Single-entry cache keyed on path+mtime:
    a whisper fallback after a parakeet failure, and job retries in the same
    process, reuse the extraction instead of re-running ffmpeg.
    """
    global _shared_wav_media, _shared_wav_path
    try:
        key = (os.path.abspath(media_path), os.path.getmtime(media_path))
    except OSError:
        key = None
    if (key is not None and _shared_wav_media == key
            and _shared_wav_path and os.path.exists(_shared_wav_path)):
        return _shared_wav_path
    _release_shared_wav()
    wav_path = _extract_wav(media_path)
    _shared_wav_media, _shared_wav_path = key, wav_path
    return wav_path


def _release_shared_wav():
    """Delete the cached wav (called when the transcription job is done)."""
    global _shared_wav_media, _shared_wav_path
    if _shared_wav_path:
        try:
            os.remove(_shared_wav_path)
        except OSError:
            pass
    _shared_wav_media = None
    _shared_wav_path = None


def _current_shared_wav_for(media_path):
    """The cached wav for this media, or None (so whisper can reuse it)."""
    if not (_shared_wav_media and _shared_wav_path
            and os.path.exists(_shared_wav_path)):
        return None
    try:
        if _shared_wav_media[0] != os.path.abspath(media_path):
            return None
    except OSError:
        return None
    return _shared_wav_path


def _words_from_tokens(tokens, timestamps, seg_start, seg_end):
    """Group parakeet BPE tokens into words with absolute timestamps.

    Verified on the prod model: tokens already carry the leading-space
    word-start convention (" T", "odo", " el", ...) and timestamps are token
    START times in seconds relative to the VAD segment. A token without a
    leading space (subword continuations, punctuation like ",") belongs to
    the previous word — same semantics merge_continuation_words expects.
    Word end is inferred: next word's start, capped near the word's last
    token so a long inter-word silence doesn't stretch the highlight.
    """
    words = []
    last_token_ts = []
    for token, ts in zip(tokens, timestamps):
        if not token:
            continue
        abs_ts = float(ts) + seg_start
        if token.startswith(" ") or not words:
            words.append({
                "word": token if token.startswith(" ") else " " + token,
                "start": abs_ts,
            })
            last_token_ts.append(abs_ts)
        else:
            words[-1]["word"] += token
            last_token_ts[-1] = abs_ts

    for i, word in enumerate(words):
        next_start = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        cap = last_token_ts[i] + 0.6
        word["end"] = float(max(word["start"] + 0.05, min(next_start, cap)))

    return words


def _transcribe_with_parakeet(media_path):
    model = _get_parakeet_model()
    wav_path = _get_shared_wav(media_path)
    # 16kHz mono s16le wav -> 32000 bytes per second of audio.
    try:
        duration = os.path.getsize(wav_path) / 32000.0
    except OSError:
        duration = 0.0
    with _ASR_GATE:
        progress = _TranscribeProgress(duration)
        results = []
        for seg in model.recognize(wav_path):
            results.append(seg)
            progress.update(float(seg.end))
        progress.update(progress.total)
    # The cached wav is NOT removed here: a whisper fallback (or a job retry
    # in this process) reuses it. transcribe_media releases it when done.

    out_segments = []
    text_parts = []
    for seg in results:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = str(seg.text or "").strip()
        if not seg_text:
            continue
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": _words_from_tokens(
                list(seg.tokens or []), list(seg.timestamps or []),
                seg_start, seg_end,
            ),
        })
        text_parts.append(seg_text)

    text = " ".join(text_parts)
    return {
        "text": text,
        "language": _detect_language(text),
        "segments": out_segments,
    }


def _detect_language(text):
    """Parakeet doesn't report a language; classify the transcribed text.

    py3langid is pure-Python and returns ISO 639-1 codes compatible with the
    whisper codes the pipeline expects (thumbnail titles, Gemini prompts).
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def _parakeet_fallback_reason(transcript, duration_hint=None):
    """Return why the parakeet result is untrustworthy, or None if it's fine."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    language = transcript.get("language")
    if language not in PARAKEET_LANGS:
        return f"language '{language}' outside parakeet's supported set"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    # Real speech averages >100 wpm; under ~12 wpm on a long video means the
    # audio was mostly not recognized (e.g. unsupported language or music).
    if duration > 60 and total_words < duration * 0.2:
        return f"only {total_words} words in {duration:.0f}s of audio"
    return None


# --- public entry point -----------------------------------------------------

class NoAudioError(Exception):
    """The media has no audio track — nothing to transcribe."""


def _has_audio_stream(media_path) -> bool:
    """True if the file has at least one audio stream (ffprobe)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # probe failed — don't block, let the backend try


def _log_asr_rtf(transcript, started):
    """One RTF line per transcription: audio seconds vs wall seconds.

    The health metric for the ASR stack (docs/transcription_gemini_fix.md
    Part 1): RTF <= 0.1 is the prod target. Real-time factor regressions
    from model/config changes show up here immediately.
    """
    segments = transcript.get("segments") or []
    audio = float(segments[-1]["end"]) if segments else 0.0
    elapsed = time.time() - started
    if audio > 0:
        print(f"🎙️ [ASR] done: {audio:.0f}s of audio in {elapsed:.1f}s "
              f"(RTF {elapsed / audio:.2f})", flush=True)


def transcribe_media(media_path):
    """Transcribe with the configured backend, falling back to whisper."""
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    started = time.time()
    try:
        backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()

        if backend == "parakeet":
            try:
                transcript = _transcribe_with_parakeet(media_path)
                reason = _parakeet_fallback_reason(transcript)
                if reason is None:
                    print(f"🎙️ [ASR] parakeet ok: lang={transcript['language']} "
                          f"segments={len(transcript['segments'])}")
                    _log_asr_rtf(transcript, started)
                    return transcript
                print(f"⚠️ [ASR] parakeet result rejected ({reason}) — "
                      f"falling back to whisper")
            except Exception as e:
                print(f"⚠️ [ASR] parakeet failed ({type(e).__name__}: {e}) — "
                      f"falling back to whisper")

        transcript = _transcribe_with_whisper(media_path)
        _log_asr_rtf(transcript, started)
        return transcript
    finally:
        # The shared wav only helps while this job transcribes; drop it so
        # temp files don't accumulate across jobs in a long-lived worker.
        _release_shared_wav()
