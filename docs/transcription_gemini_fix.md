# Transcription speed + Gemini reliability — analysis & fix plan

Date: 2026-09-08 · Status: **IMPLEMENTED** (Part 1 A+B, Part 2 1+2+3; Part 1 Option C remains an ops/config task documented in `.env.example`)

Implementation notes / deviations from the original proposal:
- Batched whisper decode engages only on GPU + media ≥ 60s (short files and unknown durations take the sequential path); `WHISPER_BATCH_SIZE=1` disables it. `WHISPER_BATCH_SIZE` doc added to `.env.example`.
- The shared 16 kHz wav is cached per media file (path+mtime) and released when `transcribe_media` finishes, so whisper fallback and same-process retries reuse it without temp-file leaks.
- Model-chain switching only happens on capacity errors (503/429): empty bodies / schema misses repeat identically on another model, so switching would only burn quota.
- The failure-time fallback is gated by env (`LLM_FALLBACK_BASE_URL` is the operator opt-in) rather than a `cloud/config.py` flag, because the worker process never imports the `cloud` package — the env var is the mechanism both modes share.
- Retry helpers unified in `gemini_worker` (`is_transient_error`, `is_capacity_error`, `retry_delay`, `call_with_retry`); `thumbnail`, `editor`, `saasshorts` and `screencast_layout` now route through them. New tests: `tests/test_gemini_fallback.py` (14 tests).

Two problems, one document:

1. **Transcription is too slow.** Root-cause analysis of the ASR architecture
   (`main.transcribe_video` → `transcribe_backends.transcribe_media`) and a
   plan for maximum throughput.
2. **Gemini is flaky.** `503 UNAVAILABLE … high demand` kills jobs after 3
   quick retries on a *single* model. Plan to make the text-LLM layer
   resilient and add a second provider.

---

## Part 1 — Transcription speed

### 1.1 Current architecture (as built)

```
job → main.transcribe_video() → transcribe_backends.transcribe_media()
        ├─ ffprobe: audio stream present? ── no → NoAudioError
        ├─ TRANSCRIBE_BACKEND=whisper (default)
        │     └─ faster-whisper WhisperModel singleton
        │        get_whisper_config(): model=small, device=cpu, compute=int8  ← DEFAULT
        │        params: beam_size=5, vad_filter=True,
        │                condition_on_previous_text=False, word_timestamps=True
        └─ TRANSCRIBE_BACKEND=parakeet (opt-in, GPU image only)
              └─ onnx-asr parakeet-tdt-0.6b-v3, ~2x faster than whisper-turbo GPU,
                 auto-fallback to whisper on error / bad result / unsupported lang
```

**Observed pain** (uvicorn_out.txt, 845s-class uploads): whisper decode takes
~166 s for a ~10 min video on CPU — acceptable RTF (~3.6x realtime), but on the
cloud host with concurrent jobs and slower shared CPUs this stretches to the
minutes-long stalls seen in the dashboard (`25% (94s) → 50% (184s)`).

### 1.2 Issues found (root causes)

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| T1 | **Defaults are CPU.** `WHISPER_DEVICE` defaults to `cpu`, `int8`, model `small`. The GPU config (`large-v3-turbo / cuda / float16`) is only a comment in `.env.example` | `subtitles.py:get_whisper_config` | The single biggest loss. GPU float16 is ~8–10x faster than CPU int8 for the same model |
| T2 | **No batched inference.** `model.transcribe()` decodes segment-by-segment. faster-whisper ships `BatchedInferencePipeline` (batched VAD chunks), typically **3–4x faster** on GPU and even ~1.5–2x on CPU | `transcribe_backends.py:_run_whisper_once` | Largest pure-code win; zero contract change (same segments/words) |
| T3 | **Parakeet not enabled in prod.** `TRANSCRIBE_BACKEND` defaults to `whisper`; parakeet is ~2x faster than whisper-turbo-GPU and needs no word-timestamp decode tax | `.env.example`, `transcribe_backends.py:transcribe_media` | We already built and tested the fast path but don't run it |
| T4 | **No HF_TOKEN.** uvicorn_out shows `unauthenticated requests to the HF Hub` warnings — model downloads are rate-limited/slow, and can make the parakeet path *look* broken and silently fall back to whisper | env | Delays + misdiagnosed fallbacks |
| T5 | **Double media decode.** Whisper decodes audio via PyAV inside the gate; parakeet re-extracts a wav per call. Every byte of a 10 min file is decoded inside the serialized GPU section | `_run_whisper_once`, `_extract_wav` | A few seconds per job; matters at concurrency |
| T6 | **Progress only at segment boundaries.** Lazy segment iteration means long silences (VAD-skipped) produce no updates — looks "stuck" even when healthy | `_TranscribeProgress` | UX only (fixed to 1% already), listed for completeness |
| T7 | **No RTF telemetry.** Nothing logs "X s of audio took Y s (RTF 0.3)". We can't prove optimizations worked or catch regressions | — | Observability |
| T8 | **ASR gate = 1 job at a time.** Correct for VRAM safety, but combined with slow decode it serializes the whole fleet on the slowest stage | `_ASR_GATE` | Throughput cap |

### 1.3 Plan options

**Option A — Config-only speedup (safe, ~1 day)**
1. Prod env: `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE=float16`, `WHISPER_MODEL=large-v3-turbo` (already documented in `.env.example`).
2. Set `HF_TOKEN` so model downloads are cached/fast.
3. Set `TRANSCRIBE_BACKEND=parakeet` on GPU hosts (keep whisper fallback — it's automatic).
4. Add RTF logging at the end of every transcription: `🎙️ [ASR] done: 845s audio in 62s (RTF 0.07)`.
- **Expected**: from ~166 s/clip → **~15–25 s** (GPU turbo) → **~8–15 s** with parakeet.
- Risk: low. No code contract changes; fallbacks already exist.

**Option B — A + batched inference (recommended, ~1–2 days)**
Everything in A, plus:
5. Add `BatchedInferencePipeline` in `_run_whisper_once`: wrap the singleton model once (`WHISPER_BATCH_SIZE`, default 8; 1 disables batching → old path). Batched mode produces the same word-timestamped segments, so the transcript contract is unchanged.
6. Pre-extract 16 kHz mono wav **once** per media file and hand the same wav to whichever backend runs (removes T5, also speeds parakeet's `_extract_wav` by caching to the job dir).
7. Benchmark script (`ops/bench_asr.py`): fixed sample audio, prints RTF per backend/config, run manually to verify each change.
- **Expected**: whisper-GPU batched ≈ parakeet-class speed; with parakeet enabled, RTF ≈ 0.05–0.1 → a 10 min video transcribes in **30–60 s worst case**.
- Risk: low-medium. Batched pipeline has edge cases on very short audio (<30 s) — fall back to the sequential path when `info.duration < 60`.

**Option C — B + throughput work (only if multiple concurrent jobs matter)**
8. Benchmark `ASR_GPU_CONCURRENCY=2` with batch_size 4 per job (VRAM-dependent); keep 1 if VRAM contention causes OOM fallbacks.
9. Cache extracted wavs across job retries (T5).
- Risk: medium (VRAM pressure interacts with the shared-GPU models); only do with GPU telemetry in hand.

**Recommendation: B.** A alone stops the bleeding; B captures the biggest
remaining win in code and is fully backward-compatible.

### 1.4 Acceptance criteria

- 10 min 1080p upload transcribes < 60 s on the prod GPU host (RTF ≤ 0.1), measured by the bench script and visible in job logs.
- Transcript contract unchanged: `tests/test_transcribe_backends.py` passes unmodified (new tests added for the batching gate + wav reuse).
- No behavior change for unsupported-language videos (parakeet → whisper fallback still logged).
- CPU-only dev machines still work: batching off or CPU-safe, force-CPU path intact.

---

## Part 2 — Gemini reliability / new LLM

### 2.1 Current architecture (as built)

- **All LLM text stages** go through `main._run_gemini_stage(client, model_name, prompt, schema)` (`main.py:1561`): one model, `max_attempts = 3`, backoff **5s / 10s / 20s**, then the job **dies**. No jitter, no alternate model, no alternate provider.
- `llm_backend.py` already abstracts an OpenAI-compatible chat endpoint (Ollama, vLLM, OpenRouter…) for the **moment picker** only, selected by env (`LLM_BASE_URL`), not by failure.
- Vision stages (layout picker, screencast content detector, silent-video path) and `saasshorts.py`/`editor.py`/`thumbnail.py` call Gemini directly with their own (or no) retry logic.

### 2.2 Issues found (root causes)

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| G1 | **Only one model is tried.** `GEMINI_MODEL` (flash-lite) 503 "high demand" is a *capacity* error: the same model in the same region usually 503s again seconds later. 3 quick retries rarely outlive a demand spike | `main.py:_run_gemini_stage` | Job-fatal after ~15 s of retries |
| G2 | **No jitter, no Retry-After.** Fixed 5/10/20s backoff; Google sends `retryDelay` in 429/503 payloads that is ignored. Thundering-herd retries at the same tick | `main.py:1611–1614` | Retries mostly wasted |
| G3 | **No provider fallback.** `llm_backend` exists but is a *configuration-time* choice, not a *failure-time* fallback. A Gemini outage = total outage even when a working OpenRouter key is configured | `llm_backend.active()` gating | Single point of failure |
| G4 | **Inconsistent retry coverage.** `saasshorts.py`, `editor.py`, `thumbnail.py`, `screencast_layout.py` each handle (or don't) transient errors differently | various | One stage's outage kills the whole job |
| G5 | **No health memory.** Every job rediscovers the same 503 storm from scratch; no circuit breaker, no "Gemini was down 3 min ago, go straight to fallback" | — | Every job pays the full retry tax during an outage |

### 2.3 Plan options

**Option 1 — Model fallback chain inside Gemini (~0.5 day)**
1. `GEMINI_FALLBACK_MODELS` (default `gemini-3.1-flash-lite, gemini-2.5-flash-lite`): on a transient failure of model N, retry on model N+1 within the same attempt budget. Add both models to `MODEL_PRICES` in `clip_selection.py` so cost stays honest.
2. Backoff: exponential **with jitter**, honor `Retry-After`, and raise `max_attempts` to 5 for capacity errors (`UNAVAILABLE`/`429`) with a hard total-deadline of ~90 s.
- Risk: low. flash-lite → 2.5-flash-lite is a near-drop-in for the schema'd JSON stages.

**Option 2 — 1 + cross-provider fallback for text stages (recommended, ~1–2 days)**
3. Extend `llm_backend.generate_json` to be the **failure-time fallback** for every text-only stage: `_run_gemini_stage` catches the final Gemini failure and, when `LLM_FALLBACK_BASE_URL` is set (OpenRouter recommended — one key gives Gemini *and* OpenAI *and* Anthropic models), retries there with the same pydantic schema (the plumbing already exists and is tested — `tests/test_llm_backend.py`).
4. Cost bookkeeping: fallback responses get the fallback model's price from `MODEL_PRICES` (add OpenRouter catalog entries) so billing stays truthful.
5. Cloud mode: allow the fallback provider in paid mode only when the operator configures it (`cloud/config.py` flag), keeping the "no key, no compute" rule.
- Risk: low-medium. Main care: different JSON-strictness → we already validate against the same schema and retry, so the contract holds.

**Option 3 — 2 + health-based routing / circuit breaker (optional hardening, ~1 day)**
6. In-process circuit breaker: 3 consecutive Gemini capacity failures ⇒ route text stages directly to the fallback for 5 min (configurable), then re-probe Gemini. Keeps outages from taxing every job with 90 s of doomed retries.
7. Unify the ad-hoc Gemini call sites (`saasshorts`, `editor`, `thumbnail`, `screencast_layout`) onto a shared retry helper with the same policy (fixes G4).
- Risk: medium — more surface, but all behind the existing `_run_gemini_stage` shape.

**Recommendation: 1 + 2, then 3 if outages recur.** 1+2 requires no new infra,
keeps Gemini as primary (cost), and turns an outage into a slowdown instead of
a failure.

### 2.4 Acceptance criteria

- A simulated constant-503 on the primary model completes the job via fallback model/provider without user-visible failure; job logs show the failover decision.
- No stage retries forever: total retry budget bounded (~90 s Gemini + ~90 s fallback).
- All existing tests pass; new unit tests cover: model-chain fallback, jitter/backoff, provider fallback, cost attribution for the fallback model.
- Config documented in `.env.example` (`GEMINI_FALLBACK_MODELS`, `LLM_FALLBACK_BASE_URL`).

---

## Suggested implementation order (after approval)

1. Part 1, Option A steps 1–4 (env + RTF log).
2. Part 1, Option B steps 5–7 (batching + shared wav extraction + bench).
3. Part 2, Option 1 (Gemini model chain + backoff).
4. Part 2, Option 2 (cross-provider fallback).
5. Revisit Part 2 Option 3 / Part 1 Option C only if prod data justifies it.