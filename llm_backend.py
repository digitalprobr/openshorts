"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


# --- failure-time fallback provider (docs/transcription_gemini_fix.md Pt 2) --
#
# The endpoint above is a configuration-time choice: when set, text stages
# bypass Gemini entirely. The one below is a FAILURE-time choice: Gemini stays
# primary, and when its whole model chain fails (or the circuit breaker is
# open) the stage re-runs here instead of killing the job. OpenRouter is the
# natural fit — one key reaches Gemini AND OpenAI AND Anthropic models.

def fallback_base_url() -> str:
    return (os.environ.get("LLM_FALLBACK_BASE_URL") or "").strip().rstrip("/")


def fallback_model() -> str:
    return (os.environ.get("LLM_FALLBACK_MODEL") or "").strip() or "openai/gpt-4o-mini"


def fallback_active() -> bool:
    """True when a fallback provider is configured (and not explicitly disabled).

    The base URL IS the operator's opt-in — cloud/paid operators who don't
    want fallback compute simply leave LLM_FALLBACK_BASE_URL unset. Set
    LLM_FALLBACK_ENABLED=0 to disable an already-configured fallback without
    touching the URL.
    """
    if not fallback_base_url():
        return False
    enabled = (os.environ.get("LLM_FALLBACK_ENABLED") or "1").strip().lower()
    return enabled not in ("0", "false", "no")


# --- Gemini circuit breaker (in-process) -------------------------------------
#
# A job runs several Gemini stages per process; after 3 consecutive capacity
# failures every later stage would burn its ~90s retry budget on the same
# doomed calls. When a fallback provider exists, the breaker short-circuits
# Gemini straight to it for a cooldown window, then lets one stage re-probe.
# Scope note: job subprocesses each start fresh, so the breaker protects
# multi-stage jobs, not the whole fleet.

_breaker_failures = 0
_breaker_open_until = 0.0


def _breaker_threshold() -> int:
    try:
        return max(1, int(os.environ.get("LLM_BREAKER_THRESHOLD", "3")))
    except ValueError:
        return 3


def _breaker_cooldown() -> float:
    try:
        return max(5.0, float(os.environ.get("LLM_BREAKER_COOLDOWN", "300")))
    except ValueError:
        return 300.0


def gemini_degraded() -> bool:
    """True while the breaker is open: skip Gemini, use the fallback directly."""
    return time.time() < _breaker_open_until


def record_gemini_failure():
    """One more Gemini capacity failure for this process (breaker bookkeeping)."""
    global _breaker_failures, _breaker_open_until
    if not fallback_active():
        return  # nowhere to route to; the breaker would only lose resilience
    _breaker_failures += 1
    if _breaker_failures >= _breaker_threshold():
        cooldown = _breaker_cooldown()
        _breaker_open_until = time.time() + cooldown
        _breaker_failures = 0
        print(f"⚡ Gemini circuit OPEN for {cooldown:.0f}s — text stages will "
              f"use the fallback provider directly, then re-probe Gemini")


def record_gemini_success():
    """Gemini answered — close the breaker (probe traffic keeps it honest)."""
    global _breaker_failures
    _breaker_failures = 0


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers(key: Optional[str] = None) -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (key or os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  *, endpoint: Optional[str] = None, api_key: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.

    ``endpoint``/``api_key`` default to the self-host configuration
    (``LLM_BASE_URL``/``LLM_API_KEY``); the Gemini failure-time fallback in
    ``main`` passes the fallback provider's values instead. ``local`` in the
    cost dict is False for those calls, and token prices are looked up from
    ``clip_selection.MODEL_PRICES`` so fallback usage is billed honestly.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{endpoint or base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers(api_key))
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    cost = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": endpoint is None,
    }
    # Real pricing when the model is in the catalog (OpenRouter fallback models
    # are), so usage through the failure-time fallback is billed honestly.
    prices = gemini_worker.lookup_model_prices(model)
    if prices:
        in_price, out_price = prices
        input_cost = input_tokens / 1_000_000 * in_price
        output_cost = output_tokens / 1_000_000 * out_price
        cost.update({
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        })
    return validated, cost
