"""
Unified LLM client with automatic Groq → Gemini Flash fallback.
 
Design decisions
----------------
Single entry point: chat_complete(messages, json_mode=False)
    All agents and the sentiment classifier call this one function. The
    fallback logic lives here and nowhere else.
 
Why the openai SDK for both providers:
    Groq exposes https://api.groq.com/openai/v1 and Gemini exposes
    https://generativelanguage.googleapis.com/v1beta/openai/ — both are
    intentional OpenAI-compatible drop-in endpoints. The openai Python
    client accepts base_url + api_key at construction time, so switching
    providers is literally two string changes with zero prompt logic changes.
    Using groq-sdk + google-generativeai would require maintaining two
    different call signatures and response parsers.
 
Why Groq as primary:
    14,400 req/day free tier on LPU hardware (Liquid Processing Units) —
    custom silicon built specifically for autoregressive inference. Sub-100ms
    time-to-first-token, the most generous free tier of any provider.
 
Why Gemini 2.0 Flash as fallback:
    1,500 req/day free tier, fast, and exposes an OpenAI-compatible endpoint
    making the fallback a zero-cost abstraction. The 10:1 ratio (Groq:Gemini
    limits) means Gemini is genuinely only used as emergency backup.
 
Why llama-3.1-8b-instruct on Groq:
    8B parameters is the sweet spot for structured JSON extraction tasks:
    small enough to run at sub-100ms on LPU hardware, large enough to follow
    JSON schema instructions reliably without CoT prompting.
 
Why gemini-2.0-flash-exp on Gemini:
    The experimental Flash model is the fastest Gemini model available on
    the free tier and produces reliable structured output comparable to
    llama-3.1-8b for classification tasks.
 
Fallback triggers:
    - HTTP 429 (rate limit): Groq quota exhausted → switch to Gemini.
    - openai.APITimeoutError after TIMEOUT_S seconds: Groq LPU congestion
      (rare but happens during peak hours) → switch to Gemini.
    - Any other openai.APIError from Groq → switch to Gemini.
      We catch broadly here because a flaky provider should never crash
      an agent mid-run.
 
LLM_PROVIDER env var:
    Setting LLM_PROVIDER=gemini bypasses Groq entirely and starts directly
    on Gemini. This is used in testing to verify the fallback path without
    having to exhaust Groq's rate limit.
 
json_mode:
    When True, passes response_format={"type": "json_object"} to the API.
    Both Groq and Gemini support this parameter via their OpenAI-compatible
    endpoints. The caller is responsible for prompting the model to produce
    JSON — json_mode only enforces that the response is valid JSON, it does
    not inject a schema.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.1-8b-instruct"
 
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_MODEL = "gemini-2.0-flash-exp"

TIMEOUT_S = 30.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_groq_client() -> openai.OpenAI:
    """
    Build an OpenAI-compatible client pointed at Groq's endpoint.
 
    Reads GROQ_API_KEY from the environment at call time (not at import time)
    so that tests can set the env var after import without module reload.
 
    Raises
    ------
    RuntimeError
        If GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Create a free key at https://console.groq.com and add it to .env: "
            "GROQ_API_KEY=gsk_..."
        )
    return openai.OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL, timeout=TIMEOUT_S)

def _get_gemini_client() -> openai.OpenAI:
    """
    Build an OpenAI-compatible client pointed at Gemini's endpoint.
 
    Reads GEMINI_API_KEY from the environment at call time.
 
    Raises
    ------
    RuntimeError
        If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Create a free key at https://aistudio.google.com and add it to .env: "
            "GEMINI_API_KEY=AIza..."
        )
    return openai.OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL, timeout=TIMEOUT_S)

def _call_provider(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    json_mode: bool,
) -> str:
    """
    Execute a single chat completion request and return the response text.
 
    Parameters
    ----------
    client:
        Pre-configured openai.OpenAI instance (Groq or Gemini).
    model:
        Model identifier string for that provider.
    messages:
        OpenAI-format message list: [{"role": "user", "content": "..."}].
    json_mode:
        If True, instructs the API to return valid JSON only.
 
    Returns
    -------
    str
        The content string from the first choice.
 
    Raises
    ------
    openai.APIError
        Any API-level error (rate limit, timeout, server error). The caller
        (chat_complete) decides whether to retry on the fallback provider.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if content is None:
        raise ValueError(f"Provider returned empty content for model {model}")
    return content

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chat_complete(
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
) -> dict[str, object]:
    """
    Send a chat completion request using Groq (primary) with automatic
    fallback to Gemini Flash on rate limit or timeout.
 
    Parameters
    ----------
    messages:
        OpenAI-format message list. Example:
            [
                {"role": "system", "content": "You are a climate analyst."},
                {"role": "user",   "content": "Classify the sentiment of: ..."},
            ]
    json_mode:
        If True, the response is guaranteed to be valid JSON. The caller
        must include JSON output instructions in the prompt — json_mode
        only enforces the format, not the schema.
 
    Returns
    -------
    dict with keys:
        "content"  : str   — the model's response text
        "provider" : str   — which provider was actually used ("groq" | "gemini")
        "model"    : str   — the exact model string used
 
    Raises
    ------
    RuntimeError
        If both providers fail. The error message includes both failure
        reasons so the caller can log them.
 
    Examples
    --------
    >>> result = chat_complete(
    ...     [{"role": "user", "content": "Say hello"}]
    ... )
    >>> result["provider"]
    'groq'
    >>> result["content"]
    'Hello! How can I help you today?'
    """
    forced_provider = os.getenv("LLM_PROVIDER", "groq").lower()

    providers = [
        ("groq",   _get_groq_client,   _GROQ_MODEL),
        ("gemini", _get_gemini_client, _GEMINI_MODEL),
    ]

    if forced_provider=="gemini":
        providers = [("gemini", _get_gemini_client, _GEMINI_MODEL)]

    errors: list[str] = []

    for provider_name, get_client, model in providers:
        try:
            client = get_client()
            content = _call_provider(client, model, messages, json_mode)

            logger.info(
                "chat_complete: provider=%s model=%s json_mode=%s",
                provider_name,
                model,
                json_mode,
            )
            return {
                "content": content,
                "provider": provider_name,
                "model": model,
            }
        except openai.RateLimitError as exc:
            msg = f"{provider_name} rate limit: {exc}"
            logger.warning("chat_complete: %s — trying next provider.", msg)
            errors.append(msg)
        except openai.APITimeoutError as exc:
            msg = f"{provider_name} timeout after {TIMEOUT_S}s: {exc}"
            logger.warning("chat_complete: %s — trying next provider.", msg)
            errors.append(msg)
        except openai.APIError as exc:
            msg = f"{provider_name} API error: {exc}"
            logger.warning("chat_complete: %s — trying next provider.", msg)
            errors.append(msg)
        except RuntimeError as exc:
            msg = f"{provider_name} config error: {exc}"
            logger.warning("chat_complete: %s — trying next provider.", msg)
            errors.append(msg)

    raise RuntimeError(
        "All LLM providers failed. Errors:\n" + "\n".join(f"  - {e}" for e in errors)
    )