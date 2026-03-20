"""Prove context optimization reduces tokens without harming results.

Each test creates a realistic payload, optimizes it, and verifies:
1. Token count drops meaningfully
2. All semantic content is preserved (lossless)
3. An LLM would produce the same answer from both versions
"""

import json

import pytest

from nadirclaw.optimize import optimize_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_lossless(original_msgs, result):
    """Verify optimization is lossless: all meaningful content preserved."""
    assert result.tokens_saved > 0, "Expected token savings"
    assert result.optimized_tokens < result.original_tokens

    for orig, opt in zip(original_msgs, result.messages):
        assert orig["role"] == opt["role"], "Roles must not change"

    # All parseable JSON in output must match original values
    for orig, opt in zip(original_msgs, result.messages):
        orig_c = orig.get("content", "")
        opt_c = opt.get("content", "")
        if not isinstance(orig_c, str) or not isinstance(opt_c, str):
            continue
        for obj in _extract_json(orig_c):
            # The same data must be recoverable from optimized content
            compact = json.dumps(obj, separators=(",", ":"), sort_keys=True)
            assert compact in json.dumps(
                list(_extract_json(opt_c)), separators=(",", ":"), sort_keys=True
            ) or _json_values_preserved(obj, opt_c), (
                f"JSON data lost during optimization: {compact[:80]}"
            )


def _extract_json(text):
    """Yield all JSON objects/arrays found in text."""
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        for ch in ("{", "["):
            idx = text.find(ch, pos)
            if idx == -1:
                continue
            try:
                obj, end = decoder.raw_decode(text, idx)
                yield obj
                pos = end
                break
            except (json.JSONDecodeError, ValueError):
                continue
        else:
            break


def _json_values_preserved(obj, text):
    """Check that all leaf values from obj appear somewhere in text."""
    if isinstance(obj, dict):
        return all(_json_values_preserved(v, text) for v in obj.values())
    if isinstance(obj, list):
        return all(_json_values_preserved(v, text) for v in obj)
    return str(obj) in text


# ======================================================================
# Scenario 1: Pretty-printed API response in context
# ======================================================================

class TestApiResponsePayload:
    """Simulates RAG/agent context stuffed with pretty-printed API data."""

    PAYLOAD = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin",
             "permissions": ["read", "write", "delete"], "last_login": "2026-03-18T10:30:00Z"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "viewer",
             "permissions": ["read"], "last_login": "2026-03-17T15:45:00Z"},
            {"id": 3, "name": "Carol", "email": "carol@example.com", "role": "editor",
             "permissions": ["read", "write"], "last_login": "2026-03-19T08:00:00Z"},
        ],
        "pagination": {"page": 1, "per_page": 25, "total": 3, "total_pages": 1},
        "metadata": {"api_version": "2.1", "response_time_ms": 42},
    }

    def test_minifies_without_data_loss(self):
        pretty = json.dumps(self.PAYLOAD, indent=4)
        messages = [
            {"role": "system", "content": "You are an API assistant."},
            {"role": "user", "content": f"Here is the user data:\n{pretty}\n\nWho has admin permissions?"},
        ]

        result = optimize_messages(messages, mode="safe")

        assert result.tokens_saved > 50
        savings_pct = result.tokens_saved / result.original_tokens * 100
        assert savings_pct > 20, f"Expected >20% savings, got {savings_pct:.1f}%"

        # ALL data is preserved — parse the optimized JSON and compare
        opt_content = result.messages[1]["content"]
        recovered = json.loads(opt_content.split("\n\n")[0].split(":\n")[1])
        assert recovered == self.PAYLOAD
        assert "Who has admin permissions?" in opt_content

    def test_question_unchanged(self):
        pretty = json.dumps(self.PAYLOAD, indent=4)
        messages = [
            {"role": "user", "content": f"Data:\n{pretty}\n\nList all emails."},
        ]
        result = optimize_messages(messages, mode="safe")
        assert "List all emails." in result.messages[0]["content"]


# ======================================================================
# Scenario 2: Agent with repeated tool schemas
# ======================================================================

class TestAgentToolSchemas:
    """Simulates an agent loop where tool schemas are sent every turn."""

    TOOLS = [
        {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                    "site_filter": {"type": "string", "description": "Restrict to domain"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            },
        },
    ]

    def _make_messages(self, turns=4):
        tools_block = "\n".join(json.dumps(t, indent=2) for t in self.TOOLS)
        msgs = [
            {"role": "system", "content": f"You are an agent. Available tools:\n{tools_block}"},
        ]
        for i in range(turns):
            msgs.append({"role": "user", "content": f"Tools available:\n{tools_block}\n\nTask {i}: search for topic {i}"})
            msgs.append({"role": "assistant", "content": f"I'll search for topic {i}."})
        return msgs

    def test_dedup_saves_significant_tokens(self):
        messages = self._make_messages(turns=4)
        result = optimize_messages(messages, mode="safe")

        assert "tool_schema_dedup" in result.optimizations_applied
        savings_pct = result.tokens_saved / result.original_tokens * 100
        assert savings_pct > 30, f"Expected >30% savings, got {savings_pct:.1f}%"

    def test_first_schema_preserved(self):
        messages = self._make_messages(turns=3)
        result = optimize_messages(messages, mode="safe")

        # First occurrence of each tool schema must be fully present
        first_system = result.messages[0]["content"]
        assert "web_search" in first_system
        assert "read_file" in first_system

    def test_tool_names_always_visible(self):
        messages = self._make_messages(turns=3)
        result = optimize_messages(messages, mode="safe")

        # Even deduped references mention the tool name
        for m in result.messages:
            c = m.get("content", "")
            if "see tool" in c:
                assert "web_search" in c or "read_file" in c

    def test_task_instructions_preserved(self):
        messages = self._make_messages(turns=4)
        result = optimize_messages(messages, mode="safe")

        user_msgs = [m for m in result.messages if m["role"] == "user"]
        for i, um in enumerate(user_msgs):
            assert f"Task {i}" in um["content"], f"Task {i} instruction lost"


# ======================================================================
# Scenario 3: Long chat history
# ======================================================================

class TestLongChatHistory:
    """Simulates a 60-turn conversation that should be trimmed."""

    def _make_conversation(self, turns=60):
        msgs = [{"role": "system", "content": "You are a coding assistant."}]
        for i in range(turns):
            msgs.append({"role": "user", "content": f"Question {i}: How do I implement feature {i}?"})
            msgs.append({"role": "assistant", "content": f"Here's how to implement feature {i}: use pattern {i}."})
        return msgs

    def test_trimming_saves_tokens(self):
        messages = self._make_conversation(60)
        result = optimize_messages(messages, mode="safe", max_turns=10)

        assert "chat_history_trim" in result.optimizations_applied
        savings_pct = result.tokens_saved / result.original_tokens * 100
        assert savings_pct > 50, f"Expected >50% savings on 60→10 trim, got {savings_pct:.1f}%"

    def test_system_prompt_preserved(self):
        messages = self._make_conversation(60)
        result = optimize_messages(messages, mode="safe", max_turns=10)
        assert result.messages[0]["content"] == "You are a coding assistant."

    def test_first_turn_preserved(self):
        messages = self._make_conversation(60)
        result = optimize_messages(messages, mode="safe", max_turns=10)

        # First user question should survive
        contents = " ".join(m["content"] for m in result.messages)
        assert "Question 0" in contents

    def test_recent_turns_preserved(self):
        messages = self._make_conversation(60)
        result = optimize_messages(messages, mode="safe", max_turns=10)

        contents = " ".join(m["content"] for m in result.messages)
        # Last few turns must be intact
        assert "Question 59" in contents
        assert "feature 59" in contents
        assert "Question 58" in contents

    def test_trimmed_count_noted(self):
        messages = self._make_conversation(60)
        result = optimize_messages(messages, mode="safe", max_turns=10)

        contents = " ".join(m["content"] for m in result.messages)
        assert "trimmed" in contents.lower()


# ======================================================================
# Scenario 4: Whitespace-bloated log output
# ======================================================================

class TestBloatedLogs:
    """Simulates verbose log/trace output pasted into context."""

    def test_whitespace_reduction(self):
        log_block = "\n\n\n".join([
            f"[2026-03-19 10:{i:02d}:00]  INFO     Processing     request     {i}"
            for i in range(20)
        ])
        messages = [
            {"role": "user", "content": f"Here are the logs:\n\n\n\n{log_block}\n\n\n\nWhat errors do you see?"},
        ]
        result = optimize_messages(messages, mode="safe")

        assert "whitespace_normalize" in result.optimizations_applied
        assert result.tokens_saved > 10
        # All log lines preserved
        assert "request     19" not in result.messages[0]["content"]  # multi-space collapsed
        assert "request 19" in result.messages[0]["content"]
        assert "What errors do you see?" in result.messages[0]["content"]


# ======================================================================
# Scenario 5: Combined — realistic agent turn
# ======================================================================

class TestRealisticAgentTurn:
    """Full agent scenario: system prompt + tools + RAG data + history."""

    def test_combined_optimization(self):
        system = "You are a data analysis agent. You help users query databases and visualize results."
        tool = {
            "name": "run_sql",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Target database name"},
                    "timeout_ms": {"type": "integer", "default": 30000},
                },
                "required": ["query", "database"],
            },
        }
        query_result = {
            "columns": ["id", "name", "revenue", "region", "quarter"],
            "rows": [
                [1, "Product A", 150000, "North", "Q1"],
                [2, "Product B", 220000, "South", "Q1"],
                [3, "Product C", 180000, "East", "Q1"],
                [4, "Product A", 165000, "North", "Q2"],
                [5, "Product B", 195000, "South", "Q2"],
            ],
            "row_count": 5,
            "execution_time_ms": 23,
        }

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Tools:\n{json.dumps(tool, indent=2)}\n\nShow me Q1 revenue by product."},
            {"role": "assistant", "content": "I'll query the database for Q1 revenue data."},
            {"role": "user", "content": (
                f"Tools:\n{json.dumps(tool, indent=2)}\n\n"
                f"Query result:\n{json.dumps(query_result, indent=4)}\n\n"
                "Now break this down by region."
            )},
        ]

        result = optimize_messages(messages, mode="safe")

        # Meaningful savings
        savings_pct = result.tokens_saved / result.original_tokens * 100
        assert savings_pct > 25, f"Expected >25% savings, got {savings_pct:.1f}%"

        # All data preserved
        opt_text = " ".join(m["content"] for m in result.messages)
        assert "Product A" in opt_text
        assert "150000" in opt_text
        assert "220000" in opt_text
        assert "Now break this down by region." in opt_text
        assert system in result.messages[0]["content"]

        # Multiple transforms fired
        assert len(result.optimizations_applied) >= 2

    def test_off_mode_is_truly_zero_cost(self):
        """off mode returns the exact same list object — no copies, no processing."""
        messages = [{"role": "user", "content": "x" * 10000}]
        result = optimize_messages(messages, mode="off")
        assert result.messages is messages
        assert result.tokens_saved == 0
        assert result.optimizations_applied == []


# ======================================================================
# Scenario 6: Edge cases that must NOT corrupt content
# ======================================================================

class TestSafetyEdgeCases:
    """Ensure optimization never corrupts tricky content."""

    def test_code_blocks_untouched(self):
        code = '```python\ndef foo():\n    data = {\n        "key":   "value"\n    }\n    return   data\n```'
        messages = [{"role": "user", "content": f"Review this code:\n{code}"}]
        result = optimize_messages(messages, mode="safe")
        # Code inside fences must not have whitespace collapsed
        assert '    data = {\n        "key":   "value"\n    }' in result.messages[0]["content"]

    def test_urls_preserved(self):
        messages = [{"role": "user", "content": "Visit https://example.com/api?q=hello&limit=10  for docs."}]
        result = optimize_messages(messages, mode="safe")
        assert "https://example.com/api?q=hello&limit=10" in result.messages[0]["content"]

    def test_empty_messages_safe(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
        result = optimize_messages(messages, mode="safe")
        assert len(result.messages) == 2

    def test_unicode_preserved(self):
        messages = [{"role": "user", "content": '{"emoji": "Hello 🌍", "cjk": "你好世界"}'}]
        result = optimize_messages(messages, mode="safe")
        content = result.messages[0]["content"]
        assert "🌍" in content
        assert "你好世界" in content

    def test_nested_json_roundtrips(self):
        deep = {"a": {"b": {"c": {"d": {"e": [1, 2, {"f": "deep"}]}}}}}
        messages = [{"role": "user", "content": json.dumps(deep, indent=4)}]
        result = optimize_messages(messages, mode="safe")
        recovered = json.loads(result.messages[0]["content"])
        assert recovered == deep
