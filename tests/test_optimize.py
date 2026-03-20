"""Tests for nadirclaw.optimize — Context Optimize transforms."""

import json

import pytest

from nadirclaw.optimize import (
    OptimizeResult,
    _dedup_system_prompts,
    _dedup_tool_schemas,
    _minify_json_in_content,
    _normalize_whitespace,
    _trim_chat_history,
    optimize_messages,
)


# ======================================================================
# JSON minification
# ======================================================================

class TestJsonMinification:
    def test_minifies_pretty_json(self):
        content = '{\n  "key": "value",\n  "num": 42\n}'
        result, changed = _minify_json_in_content(content)
        assert changed is True
        assert json.loads(result) == {"key": "value", "num": 42}
        assert "\n" not in result

    def test_leaves_non_json_alone(self):
        content = "Hello world, no JSON here"
        result, changed = _minify_json_in_content(content)
        assert result == content
        assert changed is False

    def test_preserves_json_values(self):
        original = {"nested": {"a": [1, 2, 3]}, "b": "hello world"}
        content = json.dumps(original, indent=4)
        result, changed = _minify_json_in_content(content)
        assert changed is True
        assert json.loads(result) == original

    def test_mixed_text_and_json(self):
        obj = {"tool": "search", "query": "hello"}
        content = f"Here is the result:\n{json.dumps(obj, indent=2)}\nEnd of result."
        result, changed = _minify_json_in_content(content)
        assert changed is True
        assert "Here is the result:" in result
        assert "End of result." in result
        # The JSON part should be compact
        compact = json.dumps(obj, separators=(",", ":"))
        assert compact in result

    def test_already_compact_json_unchanged(self):
        content = '{"a":1,"b":2}'
        result, changed = _minify_json_in_content(content)
        assert changed is False
        assert result == content

    def test_array_minification(self):
        content = '[\n  1,\n  2,\n  3\n]'
        result, changed = _minify_json_in_content(content)
        assert changed is True
        assert json.loads(result) == [1, 2, 3]

    def test_short_content_skipped(self):
        content = "short"
        result, changed = _minify_json_in_content(content)
        assert changed is False

    def test_invalid_json_braces_left_alone(self):
        content = "function() { return x; }"
        result, changed = _minify_json_in_content(content)
        # Should not crash; content preserved
        assert "function()" in result


# ======================================================================
# Whitespace normalization
# ======================================================================

class TestWhitespaceNormalization:
    def test_collapses_blank_lines(self):
        content = "line1\n\n\n\n\nline2"
        result, changed = _normalize_whitespace(content)
        assert changed is True
        assert result == "line1\n\nline2"

    def test_collapses_multi_spaces(self):
        content = "word1     word2    word3"
        result, changed = _normalize_whitespace(content)
        assert changed is True
        assert result == "word1 word2 word3"

    def test_preserves_code_blocks(self):
        content = "text\n```\n  indented    code\n```\nmore text"
        result, changed = _normalize_whitespace(content)
        assert "  indented    code" in result

    def test_empty_content(self):
        result, changed = _normalize_whitespace("")
        assert changed is False
        assert result == ""

    def test_already_clean(self):
        content = "clean text\nwith normal spacing"
        result, changed = _normalize_whitespace(content)
        assert changed is False


# ======================================================================
# System prompt deduplication
# ======================================================================

class TestSystemPromptDedup:
    def test_removes_duplicate_system_in_user_msg(self):
        system_text = "You are a helpful assistant that answers questions about Python."
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"Context: {system_text}\n\nWhat is a list?"},
        ]
        result, changed = _dedup_system_prompts(messages)
        assert changed is True
        assert result[0]["content"] == system_text  # system preserved
        assert system_text not in result[1]["content"]  # removed from user msg
        assert "What is a list?" in result[1]["content"]

    def test_no_false_positives_on_partial_match(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about helpful things."},
        ]
        result, changed = _dedup_system_prompts(messages)
        assert changed is False

    def test_short_system_prompt_ignored(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Be brief. What is Python?"},
        ]
        result, changed = _dedup_system_prompts(messages)
        assert changed is False  # system prompt too short (<20 chars)

    def test_no_system_messages(self):
        messages = [{"role": "user", "content": "hello"}]
        result, changed = _dedup_system_prompts(messages)
        assert changed is False


# ======================================================================
# Tool schema deduplication
# ======================================================================

class TestToolSchemaDedup:
    def test_dedup_identical_schemas(self):
        schema = json.dumps({
            "name": "search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }, indent=2)
        messages = [
            {"role": "user", "content": f"Use this tool:\n{schema}"},
            {"role": "user", "content": f"Also use:\n{schema}"},
        ]
        result, changed = _dedup_tool_schemas(messages)
        assert changed is True
        # First occurrence preserved, second replaced
        assert "search" in result[0]["content"]
        assert '[see tool "search" schema above]' in result[1]["content"]

    def test_different_schemas_preserved(self):
        schema1 = json.dumps({"name": "search", "parameters": {}}, indent=2)
        schema2 = json.dumps({"name": "browse", "parameters": {}}, indent=2)
        messages = [
            {"role": "user", "content": f"Tool 1:\n{schema1}"},
            {"role": "user", "content": f"Tool 2:\n{schema2}"},
        ]
        result, changed = _dedup_tool_schemas(messages)
        assert changed is False

    def test_non_schema_json_ignored(self):
        content = json.dumps({"data": [1, 2, 3]}, indent=2)
        messages = [
            {"role": "user", "content": content},
            {"role": "user", "content": content},
        ]
        result, changed = _dedup_tool_schemas(messages)
        assert changed is False  # not tool schemas


# ======================================================================
# Chat history trimming
# ======================================================================

class TestChatHistoryTrim:
    def test_short_conversation_untouched(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        result, changed = _trim_chat_history(messages, max_turns=40)
        assert changed is False
        assert result == messages

    def test_long_conversation_trimmed(self):
        messages = [{"role": "system", "content": "sys"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"question {i}"})
            messages.append({"role": "assistant", "content": f"answer {i}"})

        result, changed = _trim_chat_history(messages, max_turns=5)
        assert changed is True
        assert len(result) < len(messages)
        # System message preserved
        assert result[0]["content"] == "sys"
        # First turn preserved
        assert result[1]["content"] == "question 0"
        # Placeholder present
        assert any("trimmed" in m.get("content", "") for m in result)
        # Last turns preserved
        assert result[-1]["content"] == "answer 49"

    def test_system_message_preserved(self):
        messages = [{"role": "system", "content": "important system prompt"}]
        for i in range(20):
            messages.append({"role": "user", "content": f"q{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})

        result, changed = _trim_chat_history(messages, max_turns=3)
        assert changed is True
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "important system prompt"


# ======================================================================
# optimize_messages — integration
# ======================================================================

class TestOptimizeMessages:
    def test_off_mode_noop(self):
        messages = [{"role": "user", "content": "hello"}]
        result = optimize_messages(messages, mode="off")
        assert result.messages is messages  # same reference, no copy
        assert result.tokens_saved == 0
        assert result.optimizations_applied == []
        assert result.mode == "off"

    def test_safe_mode_minifies_json(self):
        pretty = json.dumps({"key": "value", "nested": {"a": 1}}, indent=4)
        messages = [{"role": "user", "content": pretty}]
        result = optimize_messages(messages, mode="safe")
        assert result.tokens_saved > 0
        assert "json_minify" in result.optimizations_applied
        # Content is lossless
        assert json.loads(result.messages[0]["content"]) == {"key": "value", "nested": {"a": 1}}

    def test_safe_mode_normalizes_whitespace(self):
        messages = [{"role": "user", "content": "line1\n\n\n\n\nline2     word"}]
        result = optimize_messages(messages, mode="safe")
        assert "whitespace_normalize" in result.optimizations_applied

    def test_aggressive_includes_safe_transforms(self):
        pretty = json.dumps({"key": "value"}, indent=4)
        messages = [{"role": "user", "content": pretty}]
        result = optimize_messages(messages, mode="aggressive")
        assert result.mode == "aggressive"
        assert result.tokens_saved > 0
        assert "json_minify" in result.optimizations_applied

    def test_no_mutation_of_input(self):
        original_content = json.dumps({"a": 1}, indent=4)
        messages = [{"role": "user", "content": original_content}]
        optimize_messages(messages, mode="safe")
        # Original should be unchanged
        assert messages[0]["content"] == original_content

    def test_result_type(self):
        result = optimize_messages([{"role": "user", "content": "hi"}], mode="safe")
        assert isinstance(result, OptimizeResult)
        assert isinstance(result.messages, list)
        assert isinstance(result.original_tokens, int)
        assert isinstance(result.optimized_tokens, int)
        assert isinstance(result.tokens_saved, int)
        assert isinstance(result.optimizations_applied, list)

    def test_multimodal_content_preserved(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }]
        result = optimize_messages(messages, mode="safe")
        # Non-text parts should be preserved
        assert result.messages[0]["content"] == messages[0]["content"]

    def test_empty_messages(self):
        result = optimize_messages([], mode="safe")
        assert result.messages == []
        assert result.tokens_saved == 0


# ======================================================================
# Semantic deduplication (aggressive mode)
# ======================================================================

class TestSemanticDedup:
    def test_near_duplicate_messages_deduped(self):
        long_content = (
            "Please explain how Python decorators work in great detail. "
            "I need a comprehensive example with code showing how to create a decorator "
            "that wraps a function, preserves its signature, and handles both positional "
            "and keyword arguments correctly. Also show how to use functools.wraps."
        )
        near_dup = (
            "Can you explain how Python decorators work in detail? "
            "I need a comprehensive example with code showing how to create a decorator "
            "that wraps a function, preserves the signature, and handles both positional "
            "and keyword arguments correctly. Please also show functools.wraps usage."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "Decorators in Python are functions that modify the behavior of other functions..."},
            {"role": "user", "content": near_dup},
        ]
        result = optimize_messages(messages, mode="aggressive")
        assert "semantic_dedup" in result.optimizations_applied
        assert result.tokens_saved > 0
        # The near-duplicate user message should be replaced with a reference
        assert "similar to earlier" in result.messages[3]["content"]

    def test_different_messages_preserved(self):
        messages = [
            {"role": "user", "content": "Explain Python decorators with a detailed example showing function wrapping and closures."},
            {"role": "assistant", "content": "Here is an explanation of decorators..."},
            {"role": "user", "content": "Now explain JavaScript promises and async/await patterns with error handling examples."},
        ]
        result = optimize_messages(messages, mode="aggressive")
        # Different topics should NOT be deduped
        assert "JavaScript promises" in result.messages[2]["content"]

    def test_system_messages_never_deduped(self):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant that writes Python code and explains concepts clearly."},
            {"role": "user", "content": "You are a helpful coding assistant that writes Python code and explains concepts clearly. Now help me."},
        ]
        result = optimize_messages(messages, mode="aggressive")
        # System message must always be preserved as-is
        assert result.messages[0]["role"] == "system"
        assert "helpful coding assistant" in result.messages[0]["content"]

    def test_short_messages_skipped(self):
        messages = [
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "yes"},
        ]
        result = optimize_messages(messages, mode="aggressive")
        # Short messages should not trigger semantic dedup
        assert "semantic_dedup" not in result.optimizations_applied

    def test_safe_mode_does_not_run_semantic(self):
        messages = [
            {"role": "user", "content": "Please explain how Python decorators work and give me a detailed example with code."},
            {"role": "assistant", "content": "Decorators are..."},
            {"role": "user", "content": "Can you explain how Python decorators work? I need a detailed example with code."},
        ]
        result = optimize_messages(messages, mode="safe")
        assert "semantic_dedup" not in result.optimizations_applied


# ======================================================================
# Aggressive accuracy — unique details must survive dedup
# ======================================================================

class TestAggressiveAccuracy:
    """Verify aggressive mode preserves critical differences in similar messages."""

    def test_refined_instruction_preserved(self):
        """User refines 'return indices' → 'return values, not indices'."""
        messages = [
            {"role": "system", "content": "You are a Python coding assistant."},
            {"role": "user", "content": (
                "Write a Python function that takes a list of integers and returns "
                "the two numbers that add up to a target sum. Use a hash map for "
                "O(n) time complexity. Handle edge cases like empty lists and "
                "duplicates. Return the indices of the two numbers."
            )},
            {"role": "assistant", "content": "Here is the two_sum function."},
            {"role": "user", "content": (
                "Write a Python function that takes a list of integers and returns "
                "the two numbers that add up to a target sum. Use a hash map for "
                "O(n) time complexity. Handle edge cases like empty lists and "
                "duplicates. Return the actual values, not indices."
            )},
        ]
        result = optimize_messages(messages, mode="aggressive")
        last = result.messages[-1]["content"]
        # The key refinement MUST survive
        assert "values" in last, f"Lost 'values' in: {last}"
        assert "not indices" in last, f"Lost 'not indices' in: {last}"
        assert result.tokens_saved > 0

    def test_format_change_preserved(self):
        """User changes output format from JSON to CSV."""
        messages = [
            {"role": "user", "content": (
                "Query the users table and return all users who signed up in the "
                "last 30 days. Include their name, email, signup date, and plan type. "
                "Format the output as JSON with proper indentation."
            )},
            {"role": "assistant", "content": "Here is the query result in JSON format."},
            {"role": "user", "content": (
                "Query the users table and return all users who signed up in the "
                "last 30 days. Include their name, email, signup date, and plan type. "
                "Format the output as CSV with headers."
            )},
        ]
        result = optimize_messages(messages, mode="aggressive")
        last = result.messages[-1]["content"]
        assert "CSV" in last, f"Lost 'CSV' format instruction in: {last}"

    def test_language_change_preserved(self):
        """User changes target language from Python to Rust."""
        messages = [
            {"role": "user", "content": (
                "Implement a binary search tree with insert, delete, and search "
                "operations. Include proper error handling and unit tests. "
                "Write it in Python using classes and type hints."
            )},
            {"role": "assistant", "content": "Here's the BST in Python."},
            {"role": "user", "content": (
                "Implement a binary search tree with insert, delete, and search "
                "operations. Include proper error handling and unit tests. "
                "Write it in Rust using generics and traits."
            )},
        ]
        result = optimize_messages(messages, mode="aggressive")
        last = result.messages[-1]["content"]
        assert "Rust" in last, f"Lost 'Rust' language instruction in: {last}"

    def test_no_dedup_when_replacement_larger(self):
        """If the deduped version would be larger, keep the original."""
        # Very short but just above MIN_CONTENT_LEN threshold — diff overhead > savings
        messages = [
            {"role": "user", "content": "a " * 35 + "hello world ending one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "a " * 35 + "hello world ending two"},
        ]
        result = optimize_messages(messages, mode="aggressive")
        if "semantic_dedup" in result.optimizations_applied:
            # If it did dedup, the result must be smaller
            assert result.tokens_saved > 0

    def test_exact_duplicate_fully_compacted(self):
        """Exact duplicate with zero diff should be compacted maximally."""
        content = (
            "Explain the difference between TCP and UDP protocols in detail. "
            "Cover reliability, ordering, connection setup, and common use cases "
            "like streaming, gaming, file transfer, and web browsing. Include "
            "performance considerations and when to choose each protocol."
        )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "TCP vs UDP explained..."},
            {"role": "user", "content": content},
        ]
        result = optimize_messages(messages, mode="aggressive")
        last = result.messages[-1]["content"]
        assert "similar to earlier" in last
        assert "Key differences" not in last  # no diff for exact duplicates
        assert result.tokens_saved > 10
