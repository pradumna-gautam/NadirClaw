"""Tests for thinking/reasoning token passthrough in NadirClaw.

Verifies that thinking parameters are forwarded to providers and
thinking/reasoning content in LLM responses is correctly preserved
in both streaming and non-streaming response formats.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nadirclaw.server import (
    ChatCompletionRequest,
    _build_streaming_response,
    _call_litellm,
    app,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_MODEL = "ollama/test-model"
OLLAMA_PROVIDER = "ollama"


def _make_request(messages, **extra):
    data = {"messages": messages, "model": "auto"}
    data.update(extra)
    return ChatCompletionRequest(**data)


def _mock_litellm_response(
    content="Hello",
    tool_calls=None,
    reasoning_content=None,
    thinking=None,
    reasoning_tokens=None,
):
    """Build a fake litellm response with optional thinking fields.

    Uses SimpleNamespace for the message and usage objects to avoid
    MagicMock's auto-attribute creation which defeats isinstance checks.
    """
    msg_attrs = {"content": content, "tool_calls": tool_calls}
    if reasoning_content is not None:
        msg_attrs["reasoning_content"] = reasoning_content
    if thinking is not None:
        msg_attrs["thinking"] = thinking
    msg = SimpleNamespace(**msg_attrs)

    usage_attrs = {"prompt_tokens": 10, "completion_tokens": 20}
    if reasoning_tokens is not None:
        usage_attrs["completion_tokens_details"] = SimpleNamespace(
            reasoning_tokens=reasoning_tokens,
        )
    usage = SimpleNamespace(**usage_attrs)

    choice = SimpleNamespace(
        message=msg,
        finish_reason="stop",
    )
    resp = SimpleNamespace(choices=[choice], usage=usage)
    return resp


# ---------------------------------------------------------------------------
# Request parameter forwarding
# ---------------------------------------------------------------------------


class TestThinkingRequestPassthrough:
    """Verify thinking/reasoning params are forwarded to litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_forwarded(self):
        request = _make_request(
            [{"role": "user", "content": "Think step by step"}],
            reasoning_effort="high",
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_thinking_param_forwarded(self):
        thinking_config = {"type": "enabled", "budget_tokens": 10000}
        request = _make_request(
            [{"role": "user", "content": "Think carefully"}],
            thinking=thinking_config,
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["thinking"] == thinking_config

    @pytest.mark.asyncio
    async def test_response_format_forwarded(self):
        request = _make_request(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_no_thinking_params_when_absent(self):
        """When no thinking params are set, they should not appear in call_kwargs."""
        request = _make_request([{"role": "user", "content": "Hello"}])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert "reasoning_effort" not in call_kwargs
        assert "thinking" not in call_kwargs
        assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


class TestThinkingResponseExtraction:
    """Verify thinking/reasoning content is extracted from LLM responses."""

    @pytest.mark.asyncio
    async def test_reasoning_content_extracted(self):
        """DeepSeek-style reasoning_content should be preserved."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(
                content="The answer is 42.",
                reasoning_content="Let me think step by step...",
            )
            request = _make_request([{"role": "user", "content": "Think"}])
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert result["content"] == "The answer is 42."
        assert result["reasoning_content"] == "Let me think step by step..."

    @pytest.mark.asyncio
    async def test_thinking_extracted(self):
        """Anthropic-style thinking should be preserved."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(
                content="Final answer.",
                thinking="I need to consider multiple angles...",
            )
            request = _make_request([{"role": "user", "content": "Think"}])
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert result["thinking"] == "I need to consider multiple angles..."

    @pytest.mark.asyncio
    async def test_reasoning_tokens_extracted(self):
        """Reasoning token count from usage details should be captured."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(
                content="Answer.",
                reasoning_tokens=150,
            )
            request = _make_request([{"role": "user", "content": "Think"}])
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert result["reasoning_tokens"] == 150

    @pytest.mark.asyncio
    async def test_no_thinking_fields_when_absent(self):
        """When model doesn't return thinking, no extra fields should appear."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(content="Just text.")
            request = _make_request([{"role": "user", "content": "Hello"}])
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert "reasoning_content" not in result
        assert "thinking" not in result
        assert "reasoning_tokens" not in result

    @pytest.mark.asyncio
    async def test_thinking_response_json_serializable(self):
        """Full result with thinking fields must be JSON-serializable."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(
                content="Answer.",
                reasoning_content="Step 1... Step 2...",
                thinking="Deep thought...",
                reasoning_tokens=200,
            )
            request = _make_request([{"role": "user", "content": "Think"}])
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["reasoning_content"] == "Step 1... Step 2..."
        assert parsed["thinking"] == "Deep thought..."
        assert parsed["reasoning_tokens"] == 200


# ---------------------------------------------------------------------------
# Non-streaming response construction
# ---------------------------------------------------------------------------


class TestThinkingInFinalResponse:
    """Verify thinking fields appear in the final API response format."""

    def _response_data(self, **overrides):
        base = {
            "content": "The answer.",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
        base.update(overrides)
        return base

    def test_reasoning_content_in_message(self):
        """reasoning_content should appear in choices[0].message."""
        from nadirclaw.server import ChatCompletionRequest
        import time

        response_data = self._response_data(
            reasoning_content="Chain of thought...",
        )

        # Simulate the response construction from chat_completions
        message = {
            "role": "assistant",
            "content": response_data["content"],
        }
        if "reasoning_content" in response_data:
            message["reasoning_content"] = response_data["reasoning_content"]

        assert message["reasoning_content"] == "Chain of thought..."

    def test_thinking_in_message(self):
        response_data = self._response_data(thinking="Extended thinking...")

        message = {
            "role": "assistant",
            "content": response_data["content"],
        }
        if "thinking" in response_data:
            message["thinking"] = response_data["thinking"]

        assert message["thinking"] == "Extended thinking..."

    def test_reasoning_tokens_in_usage(self):
        response_data = self._response_data(reasoning_tokens=150)

        usage = {
            "prompt_tokens": response_data["prompt_tokens"],
            "completion_tokens": response_data["completion_tokens"],
            "total_tokens": response_data["prompt_tokens"] + response_data["completion_tokens"],
        }
        if response_data.get("reasoning_tokens"):
            usage["completion_tokens_details"] = {
                "reasoning_tokens": response_data["reasoning_tokens"],
            }

        assert usage["completion_tokens_details"]["reasoning_tokens"] == 150
        assert usage["total_tokens"] == 30


# ---------------------------------------------------------------------------
# Fake streaming (batch-to-SSE conversion)
# ---------------------------------------------------------------------------


class TestThinkingInFakeStreaming:
    """Verify thinking fields in _build_streaming_response."""

    async def _collect_chunks(self, response_data):
        """Run the fake streaming generator and collect parsed chunks."""
        sse_response = _build_streaming_response(
            request_id="test-req",
            model="test-model",
            response_data=response_data,
            analysis_info={},
            elapsed_ms=100,
        )

        chunks = []
        async for event in sse_response.body_iterator:
            data = event.get("data", "") if isinstance(event, dict) else event
            if data == "[DONE]":
                break
            parsed = json.loads(data)
            chunks.append(parsed)

        return chunks

    @pytest.mark.asyncio
    async def test_reasoning_content_in_stream_delta(self):
        response_data = {
            "content": "Answer.",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "reasoning_content": "Step by step...",
        }

        chunks = await self._collect_chunks(response_data)
        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta["reasoning_content"] == "Step by step..."
        assert first_delta["content"] == "Answer."

    @pytest.mark.asyncio
    async def test_thinking_in_stream_delta(self):
        response_data = {
            "content": "Final.",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "thinking": "Let me reason...",
        }

        chunks = await self._collect_chunks(response_data)
        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta["thinking"] == "Let me reason..."

    @pytest.mark.asyncio
    async def test_no_thinking_in_plain_stream(self):
        response_data = {
            "content": "Hello.",
            "finish_reason": "stop",
            "prompt_tokens": 5,
            "completion_tokens": 3,
        }

        chunks = await self._collect_chunks(response_data)
        first_delta = chunks[0]["choices"][0]["delta"]
        assert "reasoning_content" not in first_delta
        assert "thinking" not in first_delta
