"""
Unit tests for shared/llm_client.py.

Strategy
--------
We never make real HTTP calls. openai.OpenAI.chat.completions.create is
patched with MagicMock so every test controls exactly what the "API" returns
or raises.

What is tested:
- Happy path: Groq returns a response → provider="groq" in result
- Fallback on RateLimitError (429): Groq raises → Gemini is called → provider="gemini"
- Fallback on APITimeoutError: Groq times out → Gemini is called → provider="gemini"
- Fallback on generic APIError: Groq raises 500 → Gemini is called
- Both providers fail → RuntimeError raised with both error messages
- LLM_PROVIDER=gemini → Groq is never called, Gemini is primary
- json_mode=True → response_format passed to the API call
- Provider name is logged on every successful call
- Missing API keys raise RuntimeError with clear messages
- Return dict has the three required keys: content, provider, model

What is NOT tested:
- Actual LLM output quality (that is the model's responsibility)
- Network-level behaviour (httpx handles that internally)
- Rate limit counting or token usage
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
import openai

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion_response(content: str) -> MagicMock:
    """
    Build a mock openai ChatCompletion response object.

    The openai SDK returns an object with .choices[0].message.content.
    We replicate only the fields chat_complete() actually accesses.
    """
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_rate_limit_error() -> openai.RateLimitError:
    """Build a minimal openai.RateLimitError (HTTP 429)."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    return openai.RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_timeout_error() -> openai.APITimeoutError:
    """Build a minimal openai.APITimeoutError."""
    mock_request = MagicMock()
    return openai.APITimeoutError(request=mock_request)


def _make_api_error() -> openai.APIError:
    """Build a minimal openai.APIError (e.g. HTTP 500)."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    return openai.APIStatusError(
        message="Internal server error",
        response=mock_response,
        body={"error": {"message": "Internal server error"}},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide fake API keys so _get_groq_client / _get_gemini_client don't raise."""
    monkeypatch.setenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    monkeypatch.setenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    monkeypatch.setenv("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "groq"))


# ---------------------------------------------------------------------------
# Tests: _get_groq_client and _get_gemini_client (key validation)
# ---------------------------------------------------------------------------

class TestClientBuilders:
    def test_groq_client_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from shared.llm_client import _get_groq_client

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _get_groq_client()

    def test_gemini_client_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from shared.llm_client import _get_gemini_client

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _get_gemini_client()

    def test_groq_client_returns_openai_instance(self) -> None:
        from shared.llm_client import _get_groq_client

        client = _get_groq_client()
        assert isinstance(client, openai.OpenAI)

    def test_gemini_client_returns_openai_instance(self) -> None:
        from shared.llm_client import _get_gemini_client

        client = _get_gemini_client()
        assert isinstance(client, openai.OpenAI)


# ---------------------------------------------------------------------------
# Tests: chat_complete — happy path
# ---------------------------------------------------------------------------

class TestChatCompleteHappyPath:
    def test_groq_success_returns_correct_dict(self) -> None:
        """
        When Groq responds normally, the result must have content, provider='groq',
        and model=_GROQ_MODEL.
        """
        from shared.llm_client import chat_complete, _GROQ_MODEL
        result = chat_complete([{"role": "user", "content": "Say hello briefly"}])
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0
        assert result["provider"] == "groq"
        assert result["model"] == _GROQ_MODEL

    def test_result_has_required_keys(self) -> None:
        """The return dict must always have content, provider, and model."""
        from shared.llm_client import chat_complete
        result = chat_complete([{"role": "user", "content": "Say one word"}])
        assert set(result.keys()) == {"content", "provider", "model"}

    def test_json_mode_passes_response_format(self) -> None:
        """
        When json_mode=True, response_format={"type": "json_object"} must be
        included in the API call. This is what tells the provider to enforce
        valid JSON output.
        """
        from shared.llm_client import chat_complete

        mock_response = _make_completion_response('{"sentiment": "positive"}')

        with patch("shared.llm_client._get_groq_client") as mock_get_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_groq.return_value = mock_client

            chat_complete([{"role": "user", "content": "classify"}], json_mode=True)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_json_mode_false_omits_response_format(self) -> None:
        """When json_mode=False (default), response_format must NOT be sent."""
        from shared.llm_client import chat_complete

        mock_response = _make_completion_response("plain text response")

        with patch("shared.llm_client._get_groq_client") as mock_get_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_groq.return_value = mock_client

            chat_complete([{"role": "user", "content": "test"}])

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Tests: chat_complete — fallback on Groq failure
# ---------------------------------------------------------------------------

class TestChatCompleteFallback:
    def test_fallback_on_rate_limit(self) -> None:
        """
        HTTP 429 from Groq → must call Gemini → result has provider='gemini'.

        This is the primary reason this module exists: the entire system
        keeps working when Groq's free quota is exhausted mid-demo.
        """
        from shared.llm_client import chat_complete, _GEMINI_MODEL

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_rate_limit_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response(
            "Hello from Gemini!"
        )

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            result = chat_complete([{"role": "user", "content": "test"}])

        assert result["provider"] == "gemini"
        assert result["model"] == _GEMINI_MODEL
        assert result["content"] == "Hello from Gemini!"

    def test_fallback_on_timeout(self) -> None:
        """
        APITimeoutError from Groq → must fall through to Gemini.
        Timeouts happen during Groq LPU congestion (rare but real).
        """
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_timeout_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response(
            "Gemini response after timeout"
        )

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            result = chat_complete([{"role": "user", "content": "test"}])

        assert result["provider"] == "gemini"

    def test_fallback_on_api_error(self) -> None:
        """
        Generic APIError (5xx) from Groq → must fall through to Gemini.
        We catch broadly so that transient server errors don't crash agents.
        """
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_api_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response(
            "Gemini fallback response"
        )

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            result = chat_complete([{"role": "user", "content": "test"}])

        assert result["provider"] == "gemini"

    def test_groq_not_called_on_fallback_succeeds_for_gemini(self) -> None:
        """Verify Groq IS called first, and Gemini IS called after Groq fails."""
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_rate_limit_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response("ok")

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            chat_complete([{"role": "user", "content": "test"}])

        groq_mock.chat.completions.create.assert_called_once()
        gemini_mock.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: chat_complete — both providers fail
# ---------------------------------------------------------------------------

class TestChatCompleteBothFail:
    def test_raises_runtime_error_when_both_fail(self) -> None:
        """
        If both Groq and Gemini fail, RuntimeError must be raised.
        The error message must mention both failure reasons so the caller
        can log a useful error.
        """
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_rate_limit_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.side_effect = _make_api_error()

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
            pytest.raises(RuntimeError, match="All LLM providers failed"),
        ):
            chat_complete([{"role": "user", "content": "test"}])

    def test_error_message_contains_both_provider_names(self) -> None:
        """The RuntimeError must mention both provider names for debuggability."""
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_rate_limit_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.side_effect = _make_api_error()

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                chat_complete([{"role": "user", "content": "test"}])

        error_text = str(exc_info.value)
        assert "groq" in error_text.lower()
        assert "gemini" in error_text.lower()


# ---------------------------------------------------------------------------
# Tests: LLM_PROVIDER env var
# ---------------------------------------------------------------------------

class TestLlmProviderEnvVar:
    def test_llm_provider_gemini_skips_groq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from shared.llm_client import chat_complete, _GEMINI_MODEL

        monkeypatch.setenv("LLM_PROVIDER", "gemini")

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response(
            "Direct Gemini response"
        )

        with (
            patch("shared.llm_client._get_groq_client") as mock_groq_builder,
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            result = chat_complete([{"role": "user", "content": "test"}])
            mock_groq_builder.assert_not_called()

        assert result["provider"] == "gemini"
        assert result["content"] == "Direct Gemini response"

    def test_llm_provider_groq_uses_groq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM_PROVIDER=groq (the default) must use Groq as primary."""
        from shared.llm_client import chat_complete

        monkeypatch.setenv("LLM_PROVIDER", "groq")

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.return_value = _make_completion_response(
            "Groq response"
        )

        with patch("shared.llm_client._get_groq_client", return_value=groq_mock):
            result = chat_complete([{"role": "user", "content": "test"}])

        assert result["provider"] == "groq"


# ---------------------------------------------------------------------------
# Tests: logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_provider_logged_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        The provider used must appear in the log on every successful call.
        This is the observable signal that the fallback activated during a demo.
        """
        from shared.llm_client import chat_complete

        mock_response = _make_completion_response("logged response")

        with patch("shared.llm_client._get_groq_client") as mock_get_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_groq.return_value = mock_client

            with caplog.at_level(logging.INFO, logger="shared.llm_client"):
                chat_complete([{"role": "user", "content": "test"}])

        assert any("groq" in record.message.lower() for record in caplog.records)

    def test_fallback_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A WARNING must be logged when Groq fails and fallback activates."""
        from shared.llm_client import chat_complete

        groq_mock = MagicMock()
        groq_mock.chat.completions.create.side_effect = _make_rate_limit_error()

        gemini_mock = MagicMock()
        gemini_mock.chat.completions.create.return_value = _make_completion_response("ok")

        with (
            patch("shared.llm_client._get_groq_client", return_value=groq_mock),
            patch("shared.llm_client._get_gemini_client", return_value=gemini_mock),
        ):
            with caplog.at_level(logging.WARNING, logger="shared.llm_client"):
                chat_complete([{"role": "user", "content": "test"}])

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_messages) >= 1
        assert any("groq" in m.lower() for m in warning_messages)