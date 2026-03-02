"""Tests for true streaming with mid-stream fallback."""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure settings are loaded before importing server
os.environ.setdefault("NADIRCLAW_SIMPLE_MODEL", "claude-haiku-4-20250514")
os.environ.setdefault("NADIRCLAW_COMPLEX_MODEL", "claude-opus-4-20250514")

from nadirclaw.server import (
    RateLimitExhausted,
    _stream_with_fallback,
)


def _make_request(messages=None):
    """Create a minimal ChatCompletionRequest-like object."""
    from nadirclaw.server import ChatCompletionRequest
    return ChatCompletionRequest(
        messages=messages or [{"role": "user", "content": "Hello"}],
        stream=True,
    )


async def _collect_events(async_gen):
    """Collect all SSE events from an async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


def _parse_sse_events(events):
    """Parse SSE event dicts into decoded data."""
    results = []
    for evt in events:
        data = evt["data"]
        if data == "[DONE]":
            results.append("[DONE]")
        else:
            results.append(json.loads(data))
    return results


class TestStreamWithFallback:

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    async def test_successful_stream(self, mock_dispatch):
        """Primary model streams successfully — no fallback needed."""
        async def _fake_stream(model, request, provider):
            yield {"role": "assistant", "content": "Hello "}, None, None
            yield {"content": "world"}, None, None
            yield {}, {"prompt_tokens": 10, "completion_tokens": 5}, "stop"

        mock_dispatch.return_value = _fake_stream("m", None, None)

        request = _make_request()
        analysis = {"tier": "simple"}
        events = await _collect_events(
            _stream_with_fallback("model-a", request, "openai", analysis, "req-1")
        )
        parsed = _parse_sse_events(events)

        # Should have content chunks + finish + [DONE]
        assert parsed[-1] == "[DONE]"
        assert any(
            isinstance(p, dict) and p.get("choices", [{}])[0].get("delta", {}).get("content") == "Hello "
            for p in parsed
        )
        assert "fallback_from" not in analysis

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    @patch("nadirclaw.server.settings")
    async def test_pre_content_fallback(self, mock_settings, mock_dispatch):
        """If primary fails before content, falls back to next model."""
        mock_settings.FALLBACK_CHAIN = ["model-b"]

        call_count = 0

        async def _fake_dispatch(model, request, provider):
            nonlocal call_count
            call_count += 1
            if model == "model-a":
                raise RateLimitExhausted(model="model-a", retry_after=60)
            # Fallback model works
            yield {"role": "assistant", "content": "From fallback"}, None, None
            yield {}, {"prompt_tokens": 8, "completion_tokens": 3}, "stop"

        mock_dispatch.side_effect = _fake_dispatch

        request = _make_request()
        analysis = {"tier": "simple"}

        with patch("nadirclaw.credentials.detect_provider", return_value="anthropic"):
            events = await _collect_events(
                _stream_with_fallback("model-a", request, "openai", analysis, "req-2")
            )

        parsed = _parse_sse_events(events)
        assert parsed[-1] == "[DONE]"

        # Should have content from fallback
        content_chunks = [
            p for p in parsed
            if isinstance(p, dict) and p.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        assert any("From fallback" in c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert analysis.get("fallback_from") == "model-a"

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    @patch("nadirclaw.server.settings")
    async def test_mid_stream_failure(self, mock_settings, mock_dispatch):
        """If model fails mid-stream, adds error notice and stops (can't restart)."""
        mock_settings.FALLBACK_CHAIN = ["model-b"]

        async def _failing_stream(model, request, provider):
            yield {"role": "assistant", "content": "Starting..."}, None, None
            raise Exception("Connection lost")

        mock_dispatch.side_effect = _failing_stream

        request = _make_request()
        analysis = {"tier": "simple"}
        events = await _collect_events(
            _stream_with_fallback("model-a", request, "openai", analysis, "req-3")
        )
        parsed = _parse_sse_events(events)

        assert parsed[-1] == "[DONE]"

        # Should contain error notice
        all_content = ""
        for p in parsed:
            if isinstance(p, dict):
                content = p.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    all_content += content
        assert "interrupted" in all_content.lower() or "error" in all_content.lower()
        assert analysis.get("_stream_error") == "Connection lost"

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    @patch("nadirclaw.server.settings")
    async def test_all_models_exhausted(self, mock_settings, mock_dispatch):
        """If all models fail pre-content, yields an error message."""
        mock_settings.FALLBACK_CHAIN = ["model-b"]

        async def _always_fail(model, request, provider):
            raise RateLimitExhausted(model=model, retry_after=60)

        mock_dispatch.side_effect = _always_fail

        request = _make_request()
        analysis = {"tier": "simple"}

        with patch("nadirclaw.credentials.detect_provider", return_value="anthropic"):
            events = await _collect_events(
                _stream_with_fallback("model-a", request, "openai", analysis, "req-4")
            )

        parsed = _parse_sse_events(events)
        assert parsed[-1] == "[DONE]"

        # Should have error content
        all_content = ""
        for p in parsed:
            if isinstance(p, dict):
                content = p.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    all_content += content
        assert "unavailable" in all_content.lower()
        assert analysis.get("_stream_error") == "all_models_exhausted"

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    @patch("nadirclaw.server.settings")
    async def test_no_fallback_chain(self, mock_settings, mock_dispatch):
        """If no fallback chain and primary fails, yields error."""
        mock_settings.FALLBACK_CHAIN = []

        async def _fail(model, request, provider):
            raise RateLimitExhausted(model=model, retry_after=60)

        mock_dispatch.side_effect = _fail

        request = _make_request()
        analysis = {"tier": "simple"}
        events = await _collect_events(
            _stream_with_fallback("model-a", request, "openai", analysis, "req-5")
        )
        parsed = _parse_sse_events(events)
        assert parsed[-1] == "[DONE]"
        assert analysis.get("_stream_error") == "all_models_exhausted"

    @pytest.mark.asyncio
    @patch("nadirclaw.server._dispatch_model_stream")
    async def test_usage_tracked(self, mock_dispatch):
        """Usage from the stream is captured in analysis_info."""
        async def _stream(model, request, provider):
            yield {"role": "assistant", "content": "Hi"}, None, None
            yield {}, {"prompt_tokens": 15, "completion_tokens": 8}, "stop"

        mock_dispatch.return_value = _stream("m", None, None)

        request = _make_request()
        analysis = {"tier": "simple"}
        events = await _collect_events(
            _stream_with_fallback("model-a", request, "openai", analysis, "req-6")
        )

        assert analysis["_stream_usage"]["prompt_tokens"] == 15
        assert analysis["_stream_usage"]["completion_tokens"] == 8
        assert analysis["_stream_model"] == "model-a"
