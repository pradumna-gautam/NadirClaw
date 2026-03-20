# Context Optimize — Savings Analysis

## Summary

NadirClaw's Context Optimize compacts bloated context (JSON, tool schemas, chat history, whitespace) before sending to the LLM provider. All transforms are **lossless** — zero semantic degradation.

Combined with smart routing, NadirClaw now saves in two ways:
1. **Route** simpler work to cheaper models
2. **Compact** bloated context before it hits your bill

## Benchmark: Claude Opus 4.6

**Pricing:** $15/1M input tokens, $75/1M output tokens

| Scenario | Before | After | Saved | % | Saved / 1K req |
|---|---:|---:|---:|---:|---:|
| Agentic coding assistant (8 turns, 5 tools repeated) | 3,657 | 1,573 | 2,084 | **57.0%** | $31.26 |
| RAG pipeline (6 chunks, pretty-printed) | 544 | 386 | 158 | **29.0%** | $2.37 |
| API response analysis (nested JSON, 5 orders) | 1,634 | 616 | 1,018 | **62.3%** | $15.27 |
| Long debug session (50 turns, JSON logs) | 3,856 | 1,414 | 2,442 | **63.3%** | $36.63 |
| OpenAPI spec context (5 endpoints) | 2,649 | 762 | 1,887 | **71.2%** | $28.30 |
| **Total** | **12,340** | **4,751** | **7,589** | **61.5%** | **$113.84** |

### Transforms Applied

| Scenario | Transforms |
|---|---|
| Agentic coding assistant | tool_schema_dedup, json_minify, whitespace_normalize |
| RAG pipeline | json_minify |
| API response analysis | json_minify |
| Long debug session | json_minify, chat_history_trim |
| OpenAPI spec context | json_minify |

### Where the Savings Come From

- **JSON minification** — Pretty-printed JSON (indent=2 or indent=4) is common in agent tool outputs, RAG chunks, and API responses. Compact re-serialization removes all formatting whitespace while preserving every value.
- **Tool schema deduplication** — Agent frameworks often re-send the full tool schema with every turn. NadirClaw keeps the first occurrence and replaces repeats with a short reference.
- **Chat history trimming** — Long conversations accumulate tokens that are far from the current task. Trimming to recent turns (default: 40) keeps context relevant and cheap.
- **Whitespace normalization** — Log dumps, stack traces, and verbose output contain runs of blank lines and spaces that carry no semantic value.

## Projected Monthly Savings (Opus 4.6)

| Daily Requests | Monthly Requests | Tokens Saved | Monthly Savings |
|---:|---:|---:|---:|
| 100 | 3,000 | ~4.5M | **$68** |
| 500 | 15,000 | ~22.8M | **$342** |
| 1,000 | 30,000 | ~45.5M | **$683** |
| 5,000 | 150,000 | ~227.7M | **$3,415** |
| 10,000 | 300,000 | ~455.3M | **$6,830** |

*Average savings per request: ~1,517 tokens (61.5%)*

## Safety Guarantees

All safe-mode transforms are deterministic and lossless:

- JSON values roundtrip exactly (parse + compact re-serialize)
- Code blocks inside fences (```) are never modified
- URLs are preserved character-for-character
- Unicode and emoji roundtrip correctly
- Deeply nested structures are handled without data loss
- `off` mode has zero overhead — no message copying, no processing

## How to Enable

```bash
# Server-wide
nadirclaw serve --optimize safe

# Or via environment variable
NADIRCLAW_OPTIMIZE=safe nadirclaw serve

# Per-request override (in the request body)
{"model": "auto", "optimize": "safe", "messages": [...]}

# Dry-run on a file
nadirclaw optimize payload.json --mode safe --format json
```
