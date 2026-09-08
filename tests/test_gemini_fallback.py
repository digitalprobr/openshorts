"""Gemini resilience: model fallback chain, provider fallback, circuit breaker.

Implements docs/transcription_gemini_fix.md Part 2. Prod kept losing jobs to
503 UNAVAILABLE "high demand" after 3 quick retries on a single model; these
tests pin the new contract: capacity errors get more attempts per model, the
next model in GEMINI_FALLBACK_MODELS takes over, a configured fallback
provider (LLM_FALLBACK_BASE_URL) catches a fully exhausted chain, and the
in-process breaker routes straight to the fallback while it is open.
"""
import pytest

import gemini_worker
import llm_backend

# main pulls in cv2/torch/mediapipe at import time; the minimal CI env lacks
# them, so skip there. Runs fully in the container/local where deps exist.
main = pytest.importorskip("main")


class _FakeResponse:
    """Mimics a good genai response for the parsed-JSON success path."""

    def __init__(self, payload):
        self.parsed = payload
        self.candidates = []
        self.prompt_feedback = None
        self.usage_metadata = None


class _FakeModels:
    """generate_content stub.

    Raises ``error`` forever by default; with ``payload`` set, the first
    ``blips`` calls raise and the call after that succeeds.
    """

    def __init__(self, error=None, blips=0, payload=None):
        self.error = error or "503 UNAVAILABLE: high demand"
        self.blips = blips
        self.payload = payload
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.blips:
            raise RuntimeError(self.error)
        if self.payload is None:
            raise RuntimeError(self.error)
        return _FakeResponse(self.payload)


def types_client(models):
    import types as _types
    return _types.SimpleNamespace(models=models)


def _schema_payload():
    import types as _types
    return _types.SimpleNamespace(
        model_dump=lambda: {"windows": [{"id": "w0", "score": 90}]})


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """No sleeps, empty breaker, no fallback provider unless a test opts in."""
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    monkeypatch.setattr(gemini_worker.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm_backend, "_breaker_failures", 0)
    monkeypatch.setattr(llm_backend, "_breaker_open_until", 0.0)
    monkeypatch.delenv("LLM_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)


# --- model fallback chain ----------------------------------------------------

def test_capacity_error_gets_five_attempts_per_model(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")  # single-model chain
    models = _FakeModels()
    with pytest.raises(RuntimeError, match="high demand"):
        main._run_gemini_stage(types_client(models), "m", "prompt", object)
    assert models.calls == 5


def test_chain_switches_models_on_capacity_error(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "b-model, c-model")
    models = _FakeModels()
    with pytest.raises(RuntimeError):
        main._run_gemini_stage(types_client(models), "a-model", "prompt", object)
    # 5 capacity attempts per model x 3 models in the chain.
    assert models.calls == 15


def test_chain_switch_logs_the_next_model(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite")
    models = _FakeModels()
    with pytest.raises(RuntimeError):
        main._run_gemini_stage(types_client(models), "gemini-3.1-flash-lite",
                               "prompt", object)
    out = capsys.readouterr().out
    assert "switching to gemini-2.5-flash-lite" in out


def test_non_capacity_transient_does_not_switch_models(monkeypatch):
    # Empty bodies / schema misses repeat identically on another model, so a
    # chain switch would only burn quota — fail after the usual 3 attempts.
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "b-model")
    models = _FakeModels(error="empty response body")
    with pytest.raises(RuntimeError, match="empty response body"):
        main._run_gemini_stage(types_client(models), "m", "prompt", object)
    assert models.calls == 3


def test_recovers_from_capacity_blips(monkeypatch):
    models = _FakeModels(blips=2, payload={"windows": [{"id": "w0", "score": 90}]})
    parsed, _ = main._run_gemini_stage(types_client(models), "m", "prompt",
                                       _schema_payload())
    assert models.calls == 3
    assert parsed["windows"][0]["score"] == 90

# --- failure-time provider fallback ------------------------------------------

def test_exhausted_chain_falls_back_to_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "http://fb.test/v1")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "openai/gpt-4o-mini")
    seen = {}

    def fake_generate(prompt, schema, model=None, endpoint=None, api_key=None):
        seen.update(model=model, endpoint=endpoint)
        return {"windows": []}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    models = _FakeModels()
    parsed, _ = main._run_gemini_stage(types_client(models), "m", "prompt", object)
    assert models.calls == 5  # Gemini chain ran first...
    assert seen == {"model": "openai/gpt-4o-mini", "endpoint": "http://fb.test/v1"}
    assert parsed == {"windows": []}  # ...and the fallback saved the stage.


def test_no_fallback_configured_raises_after_the_chain(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    models = _FakeModels()
    with pytest.raises(RuntimeError, match="high demand"):
        main._run_gemini_stage(types_client(models), "m", "prompt", object)
    assert models.calls == 5


# --- circuit breaker ----------------------------------------------------------

def test_open_breaker_routes_straight_to_the_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "http://fb.test/v1")
    for _ in range(3):  # threshold
        llm_backend.record_gemini_failure()
    assert llm_backend.gemini_degraded() is True

    class _MustNotCall:
        def __getattr__(self, name):
            raise AssertionError("Gemini must not be called while the breaker is open")

    called = {}
    monkeypatch.setattr(
        llm_backend, "generate_json",
        lambda *a, **kw: (called.setdefault("called", True),
                          ({"windows": []}, None))[1])
    parsed, _ = main._run_gemini_stage(_MustNotCall(), "m", "prompt", object)
    assert parsed == {"windows": []}
    assert called == {"called": True}


def test_breaker_never_opens_without_a_fallback(monkeypatch):
    for _ in range(5):
        llm_backend.record_gemini_failure()
    assert llm_backend.gemini_degraded() is False


def test_success_resets_the_breaker(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "http://fb.test/v1")
    llm_backend.record_gemini_failure()
    llm_backend.record_gemini_failure()
    llm_backend.record_gemini_success()
    llm_backend.record_gemini_failure()
    assert llm_backend.gemini_degraded() is False


# --- shared retry policy ------------------------------------------------------

def test_call_with_retry_gives_capacity_errors_five_attempts():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 5:
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    assert gemini_worker.call_with_retry(flaky) == "ok"
    assert attempts["n"] == 5


def test_call_with_retry_non_transient_fails_fast():
    attempts = {"n": 0}

    def boom():
        attempts["n"] += 1
        raise ValueError("400 INVALID_ARGUMENT")

    with pytest.raises(ValueError):
        gemini_worker.call_with_retry(boom)
    assert attempts["n"] == 1


def test_retry_delay_honors_google_retrydelay():
    msg = "503 {'error': {'retryDelay': '26s'}}"
    assert gemini_worker.retry_delay(msg, 1) == 26.0


def test_retry_delay_is_exponential_with_jitter():
    d = gemini_worker.retry_delay("503 no hint", 1)
    assert 4.0 <= d <= 6.0  # 5s ±20%
    d3 = gemini_worker.retry_delay("503 no hint", 3)
    assert d3 <= 36.0  # capped at 30s ±20%