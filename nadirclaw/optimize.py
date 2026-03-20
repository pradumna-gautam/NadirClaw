"""Context Optimize — compact bloated context before LLM dispatch.

Modes
-----
- ``off``        No processing (zero overhead).
- ``safe``       Deterministic, lossless transforms only.
- ``aggressive`` All safe transforms + semantic deduplication via embeddings.

All public functions operate on plain ``list[dict]`` messages so the module
has no dependency on FastAPI, Pydantic, or the rest of the server.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """Returned by :func:`optimize_messages`."""
    messages: list[dict]
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    mode: str
    optimizations_applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token estimation — tiktoken (accurate) with len//4 fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude-family BPE

    def _estimate_tokens_str(text: str) -> int:
        return max(1, len(_enc.encode(text, disallowed_special=())))
except Exception:                       # pragma: no cover — missing or broken tiktoken
    def _estimate_tokens_str(text: str) -> int:
        return max(1, len(text) // 4)


def _estimate_tokens_messages(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += _estimate_tokens_str(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += _estimate_tokens_str(part.get("text", ""))
        # role overhead
        total += 4
    return total


# ---------------------------------------------------------------------------
# Transform 1 — System-prompt deduplication
# ---------------------------------------------------------------------------

def _dedup_system_prompts(messages: list[dict]) -> tuple[list[dict], bool]:
    """Remove system-prompt text that is duplicated verbatim in later messages."""
    system_texts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and len(content) >= 20:
                system_texts.append(content)

    if not system_texts:
        return messages, False

    changed = False
    result: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            result.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, str):
            result.append(m)
            continue
        new_content = content
        for sys_text in system_texts:
            if sys_text in new_content:
                new_content = new_content.replace(sys_text, "").strip()
                changed = True
        if new_content != content:
            result.append({**m, "content": new_content})
        else:
            result.append(m)
    return result, changed


# ---------------------------------------------------------------------------
# Transform 2 — Tool-schema deduplication
# ---------------------------------------------------------------------------

def _dedup_tool_schemas(messages: list[dict]) -> tuple[list[dict], bool]:
    """Replace repeated identical tool/function schemas with a short reference."""
    seen_schemas: dict[str, int] = {}  # canonical JSON → first-seen message index
    changed = False
    result: list[dict] = []

    for idx, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, str) or len(content) < 50:
            result.append(m)
            continue

        new_content = content
        # Find JSON objects that look like tool schemas (contain "name" and
        # "parameters" or "function" keys)
        for match_obj in _iter_json_objects(content):
            obj, start, end = match_obj
            if not isinstance(obj, dict):
                continue
            # Heuristic: looks like a tool schema
            if not (_is_tool_schema(obj)):
                continue
            canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            if canonical in seen_schemas:
                ref = f'[see tool "{obj.get("name", "?")}" schema above]'
                new_content = new_content[:start] + ref + new_content[end:]
                changed = True
            else:
                seen_schemas[canonical] = idx

        if new_content != content:
            result.append({**m, "content": new_content})
        else:
            result.append(m)

    return result, changed


def _is_tool_schema(obj: dict) -> bool:
    """Heuristic: dict looks like a tool/function schema."""
    if "name" in obj and ("parameters" in obj or "input_schema" in obj):
        return True
    if "function" in obj and isinstance(obj["function"], dict):
        return True
    return False


# ---------------------------------------------------------------------------
# Transform 3 — JSON minification
# ---------------------------------------------------------------------------

def _minify_json_in_content(content: str) -> tuple[str, bool]:
    """Find JSON objects/arrays in text and re-serialize compactly.

    Uses ``json.JSONDecoder.raw_decode`` to handle JSON embedded in prose.
    Only replaces when the compact form is actually shorter.
    Skips content inside fenced code blocks (``` ... ```).
    """
    if not content or len(content) < 10:
        return content, False

    # Split on code fences — only process non-code segments
    parts = re.split(r"(```[^\n]*\n.*?```)", content, flags=re.DOTALL)
    changed = False
    result_segments: list[str] = []

    for i, segment in enumerate(parts):
        if segment.startswith("```"):
            # Code block — leave untouched
            result_segments.append(segment)
        else:
            minified, seg_changed = _minify_json_segment(segment)
            result_segments.append(minified)
            if seg_changed:
                changed = True

    return "".join(result_segments), changed


def _minify_json_segment(text: str) -> tuple[str, bool]:
    """Minify JSON in a single non-code-block text segment."""
    if not text or len(text) < 10:
        return text, False

    decoder = json.JSONDecoder()
    changed = False
    result_parts: list[str] = []
    pos = 0

    while pos < len(text):
        next_brace = len(text)
        for ch in ("{", "["):
            idx = text.find(ch, pos)
            if idx != -1 and idx < next_brace:
                next_brace = idx

        if next_brace == len(text):
            result_parts.append(text[pos:])
            break

        result_parts.append(text[pos:next_brace])

        try:
            obj, end_idx = decoder.raw_decode(text, next_brace)
            compact = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
            original_slice = text[next_brace:end_idx]
            if len(compact) < len(original_slice):
                result_parts.append(compact)
                changed = True
            else:
                result_parts.append(original_slice)
            pos = end_idx
        except (json.JSONDecodeError, ValueError):
            result_parts.append(text[next_brace])
            pos = next_brace + 1

    return "".join(result_parts), changed


# ---------------------------------------------------------------------------
# Transform 4 — Whitespace normalization
# ---------------------------------------------------------------------------

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def _normalize_whitespace(content: str) -> tuple[str, bool]:
    """Collapse excessive blank lines and spaces, preserving code blocks."""
    if not content:
        return content, False

    lines = content.split("\n")
    in_code_block = False
    out_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue
        # Collapse multi-spaces outside code blocks
        out_lines.append(_MULTI_SPACES.sub(" ", line))

    result = "\n".join(out_lines)
    # Collapse 3+ consecutive blank lines → 2
    result = _MULTI_BLANK_LINES.sub("\n\n", result)
    return result, result != content


# ---------------------------------------------------------------------------
# Transform 5 — Chat-history trimming
# ---------------------------------------------------------------------------

def _trim_chat_history(
    messages: list[dict], max_turns: int = 40
) -> tuple[list[dict], bool]:
    """Trim long conversations, keeping system msgs + first turn + last N turns.

    A "turn" is a user message followed by zero or more non-user messages
    (assistant, tool, etc.).
    """
    # Separate system messages from the rest
    system_msgs: list[dict] = []
    conversation: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_msgs.append(m)
        else:
            conversation.append(m)

    # Count user turns
    user_indices = [i for i, m in enumerate(conversation) if m.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return messages, False

    # Keep first turn (up to second user message) and last max_turns-1 turns
    first_turn_end = user_indices[1] if len(user_indices) > 1 else len(conversation)
    first_turn = conversation[:first_turn_end]

    # Last (max_turns - 1) turns start from the user_indices[-(max_turns-1)] position
    keep_from = max_turns - 1
    last_start_idx = user_indices[-keep_from] if keep_from <= len(user_indices) else 0
    last_turns = conversation[last_start_idx:]

    trimmed_count = len(user_indices) - max_turns
    placeholder = {
        "role": "system",
        "content": f"[...{trimmed_count} earlier turns trimmed for context optimization...]",
    }

    result = system_msgs + first_turn + [placeholder] + last_turns
    return result, True


# ---------------------------------------------------------------------------
# JSON object iterator (shared utility)
# ---------------------------------------------------------------------------

def _iter_json_objects(text: str):
    """Yield (parsed_obj, start, end) for each top-level JSON value in *text*."""
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        # Find next { or [
        next_brace = len(text)
        for ch in ("{", "["):
            idx = text.find(ch, pos)
            if idx != -1 and idx < next_brace:
                next_brace = idx
        if next_brace == len(text):
            break
        try:
            obj, end_idx = decoder.raw_decode(text, next_brace)
            yield obj, next_brace, end_idx
            pos = end_idx
        except (json.JSONDecodeError, ValueError):
            pos = next_brace + 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transform 6 — Semantic deduplication (aggressive mode only)
# ---------------------------------------------------------------------------

_SEMANTIC_SIMILARITY_THRESHOLD = 0.85  # cosine similarity above this = "same"
_MIN_CONTENT_LEN_FOR_SEMANTIC = 60     # skip short messages


def _extract_diff_phrases(earlier: str, later: str) -> str:
    """Return the *changed* phrases from *later* relative to *earlier*.

    Uses ``difflib.SequenceMatcher`` on word tokens to find inserted or
    replaced runs of words.  This captures fine-grained edits like
    "return indices" → "return actual values, not indices" without
    treating the whole message as unique.
    """
    from difflib import SequenceMatcher

    a_words = earlier.split()
    b_words = later.split()
    sm = SequenceMatcher(None, a_words, b_words, autojunk=False)

    diff_parts: list[str] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            diff_parts.append(" ".join(b_words[j1:j2]))

    return " ".join(diff_parts)


def _semantic_dedup(
    messages: list[dict],
    threshold: float = _SEMANTIC_SIMILARITY_THRESHOLD,
) -> tuple[list[dict], bool]:
    """Deduplicate near-similar messages while preserving unique details.

    Compares each user/assistant message to all prior messages of the same
    role.  If cosine similarity exceeds *threshold*, the later message is
    replaced with a compact reference **plus any sentences that differ** from
    the earlier message.  This keeps token savings high while avoiding
    accuracy loss from losing refinements the user made.

    Requires ``sentence-transformers`` (loaded lazily via the shared encoder).
    System messages and short messages are never deduplicated.
    """
    try:
        from nadirclaw.encoder import get_shared_encoder_sync
        import numpy as np
    except ImportError:
        # sentence-transformers not installed — skip silently
        return messages, False

    # Collect candidate texts and their indices
    candidates: list[tuple[int, str]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            continue
        content = m.get("content")
        if not isinstance(content, str) or len(content) < _MIN_CONTENT_LEN_FOR_SEMANTIC:
            continue
        candidates.append((i, content))

    if len(candidates) < 2:
        return messages, False

    encoder = get_shared_encoder_sync()
    texts = [c[1] for c in candidates]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    changed = False
    removed: set[int] = set()  # candidate indices that were deduped
    result = list(messages)

    for j in range(1, len(candidates)):
        if j in removed:
            continue
        idx_j = candidates[j][0]
        role_j = messages[idx_j].get("role")
        emb_j = embeddings[j]

        for k in range(j):
            if k in removed:
                continue
            idx_k = candidates[k][0]
            if messages[idx_k].get("role") != role_j:
                continue

            sim = float(np.dot(emb_j, embeddings[k]))
            if sim >= threshold:
                # Build compact replacement: reference + unique diff
                preview = texts[k][:60].replace("\n", " ")
                diff = _extract_diff_phrases(texts[k], texts[j])
                if diff:
                    replacement = (
                        f'[similar to earlier message: "{preview}..."]\n'
                        f"Key differences: {diff}"
                    )
                else:
                    replacement = f'[similar to earlier message: "{preview}..."]'

                # Only replace if we actually save tokens
                if _estimate_tokens_str(replacement) < _estimate_tokens_str(texts[j]):
                    result[idx_j] = {
                        **messages[idx_j],
                        "content": replacement,
                    }
                    removed.add(j)
                    changed = True
                break  # one match is enough

    return result, changed


_SAFE_TRANSFORMS = [
    ("system_prompt_dedup", lambda msgs, **_: _dedup_system_prompts(msgs)),
    ("tool_schema_dedup", lambda msgs, **_: _dedup_tool_schemas(msgs)),
]

# Content-level transforms (operate on individual message content strings)
_SAFE_CONTENT_TRANSFORMS = [
    ("json_minify", _minify_json_in_content),
    ("whitespace_normalize", _normalize_whitespace),
]


def optimize_messages(
    messages: list[dict],
    mode: str = "off",
    max_turns: int = 40,
) -> OptimizeResult:
    """Optimize a list of message dicts for token reduction.

    Parameters
    ----------
    messages
        List of ``{"role": "...", "content": "..."}`` dicts.
    mode
        ``"off"`` (no-op), ``"safe"`` (lossless), or ``"aggressive"``
        (safe + semantic deduplication via sentence embeddings).
    max_turns
        Maximum conversation turns to keep when trimming history.

    Returns
    -------
    OptimizeResult
        Contains optimized messages and savings metrics.
    """
    original_tokens = _estimate_tokens_messages(messages)

    if mode == "off":
        return OptimizeResult(
            messages=messages,
            original_tokens=original_tokens,
            optimized_tokens=original_tokens,
            tokens_saved=0,
            mode="off",
        )

    applied: list[str] = []

    # Deep copy messages to avoid mutating input
    msgs = [{**m} for m in messages]

    # --- Message-level transforms (safe) ---
    for name, fn in _SAFE_TRANSFORMS:
        msgs, did_change = fn(msgs)
        if did_change:
            applied.append(name)

    # --- Content-level transforms (safe) ---
    for name, fn in _SAFE_CONTENT_TRANSFORMS:
        content_changed = False
        for i, m in enumerate(msgs):
            content = m.get("content")
            if not isinstance(content, str) or len(content) < 10:
                continue
            new_content, changed = fn(content)
            if changed:
                msgs[i] = {**m, "content": new_content}
                content_changed = True
        if content_changed:
            applied.append(name)

    # --- Aggressive-only transforms ---
    if mode == "aggressive":
        msgs, did_semantic = _semantic_dedup(msgs)
        if did_semantic:
            applied.append("semantic_dedup")

    # --- Chat history trimming ---
    msgs, did_trim = _trim_chat_history(msgs, max_turns=max_turns)
    if did_trim:
        applied.append("chat_history_trim")

    optimized_tokens = _estimate_tokens_messages(msgs)

    return OptimizeResult(
        messages=msgs,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        tokens_saved=max(0, original_tokens - optimized_tokens),
        mode=mode,
        optimizations_applied=applied,
    )
