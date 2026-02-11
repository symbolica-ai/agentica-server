"""Anthropic prompt caching: add cache_control breakpoints to inference requests.

Anthropic models require explicit cache_control breakpoints to enable prompt caching.
This module adds up to 4 breakpoints using a walking-backwards strategy to maximize
cache coverage across multi-turn conversations.

Breakpoints are free to add and Anthropic silently ignores them below the minimum
token threshold, so there is no downside to always applying them.

Done as faithfully as possible according to the Anthropic & Openrouter documentation:
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://openrouter.ai/docs/guides/best-practices/prompt-caching
"""

from __future__ import annotations

import copy

from agentica_internal.session_manager_messages import CacheTTL
from anthropic.types import MessageParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputItemParam

MAX_BREAKPOINTS = 4
LOOKBACK_WINDOW = 20


def _cache_control(cache_ttl: CacheTTL | None) -> dict:
    """Build the cache_control dict, optionally including a TTL."""
    if cache_ttl is not None:
        return {"type": "ephemeral", "ttl": cache_ttl}
    return {"type": "ephemeral"}


def _get_breakpoint_indices(length: int, system_idx: int | None) -> list[int]:
    """Calculate indices where cache_control breakpoints should be placed.

    Strategy: always mark the system message, then walk backwards from the end
    placing a breakpoint every LOOKBACK_WINDOW items (up to MAX_BREAKPOINTS total).
    """
    if length == 0:
        return []

    indices: set[int] = set()

    if system_idx is not None:
        indices.add(system_idx)

    budget = MAX_BREAKPOINTS - len(indices)
    pos = length - 1
    placed = 0
    while placed < budget and pos >= 0:
        if pos != system_idx:
            indices.add(pos)
            placed += 1
        pos -= LOOKBACK_WINDOW

    return sorted(indices)


# ---------------------------------------------------------------------------
# Chat Completions API
# ---------------------------------------------------------------------------


def apply_cache_control_chat_completions(
    messages: list[ChatCompletionMessageParam],
    cache_ttl: CacheTTL | None = None,
) -> list[ChatCompletionMessageParam]:
    """Return a new message list with cache_control breakpoints for Anthropic.

    Shallow-copies the list; deep-copies only the messages that need annotation.
    String content is converted to multipart format as required by the API.
    """
    if not messages:
        return messages

    system_idx: int | None = None
    for i, msg in enumerate(messages):
        if msg.get('role') == 'system':
            system_idx = i
            break

    indices = _get_breakpoint_indices(len(messages), system_idx)
    if not indices:
        return messages

    result = list(messages)
    for idx in indices:
        result[idx] = _annotate_chat_message(copy.deepcopy(messages[idx]), cache_ttl)
    return result


def _annotate_chat_message(
    msg: ChatCompletionMessageParam,
    cache_ttl: CacheTTL | None = None,
) -> ChatCompletionMessageParam:
    """Add cache_control to the last text part of a Chat Completions message."""
    content = msg.get('content')
    if content is None:
        return msg

    cc = _cache_control(cache_ttl)
    if isinstance(content, str):
        msg['content'] = [{"type": "text", "text": content, "cache_control": cc}]  #  type: ignore openai's SDK does not do caching like this, so their SDK doesn't type for it
    elif isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') == 'text':
                part['cache_control'] = cc  #  type: ignore openai's API does not do caching like this, so their SDK doesn't type for it
                break
    return msg


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def apply_cache_control_responses(
    input_items: list[ResponseInputItemParam],
    cache_ttl: CacheTTL | None = None,
) -> list[ResponseInputItemParam]:
    """Return a new input list with cache_control breakpoints for Anthropic.

    Only annotates items with type "message". When a breakpoint position falls on
    a non-message item, the nearest message-type item looking backwards is used.
    """
    if not input_items:
        return input_items

    system_idx: int | None = None
    for i, item in enumerate(input_items):
        if (
            isinstance(item, dict)
            and item.get('type') == 'message'
            and item.get('role') == 'system'
        ):
            system_idx = i
            break

    indices = _get_breakpoint_indices(len(input_items), system_idx)
    if not indices:
        return input_items

    resolved: set[int] = set()
    for idx in indices:
        actual = _find_message_index(input_items, idx)
        if actual is not None:
            resolved.add(actual)

    if not resolved:
        return input_items

    result = list(input_items)
    for idx in resolved:
        result[idx] = _annotate_response_message(copy.deepcopy(input_items[idx]), cache_ttl)
    return result


def _find_message_index(items: list[ResponseInputItemParam], start: int) -> int | None:
    """Find nearest message-type item at or before *start*."""
    for i in range(start, -1, -1):
        item = items[i]
        if isinstance(item, dict) and item.get('type') == 'message':
            return i
    return None


def _annotate_response_message(
    item: ResponseInputItemParam,
    cache_ttl: CacheTTL | None = None,
) -> ResponseInputItemParam:
    """Add cache_control to the last text part of a Responses API message."""
    if not isinstance(item, dict):
        return item

    content = item.get('content')
    if content is None:
        return item

    cc = _cache_control(cache_ttl)
    if isinstance(content, str):
        text_type = 'output_text' if item.get('role') == 'assistant' else 'input_text'
        item['content'] = [{"type": text_type, "text": content, "cache_control": cc}]  #  type: ignore openai's SDK does not do caching like this, so their SDK doesn't type for it
    elif isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') in ('input_text', 'output_text', 'text'):
                part['cache_control'] = cc  #  type: ignore openai's SDK does not do caching like this, so their SDK doesn't type for it
                break
    return item


# ---------------------------------------------------------------------------
# Anthropic Messages API (native format)
# ---------------------------------------------------------------------------


def apply_cache_control_messages(
    messages: list[MessageParam],
    cache_ttl: CacheTTL | None = None,
) -> list[MessageParam]:
    """Return a new message list with cache_control breakpoints for Anthropic native API.

    Annotates the last text or tool_result content block in target messages
    with cache_control: {"type": "ephemeral"}.

    Note: MessageParam is a TypedDict with 'role' and 'content' keys. The content
    can be a str or a list of ContentBlockParam (also TypedDicts). We deep-copy
    and mutate the TypedDicts at breakpoint positions to add cache_control.
    """
    if not messages:
        return messages

    # No system messages in Anthropic messages list (they're stored separately),
    # so system_idx is always None here
    indices = _get_breakpoint_indices(len(messages), system_idx=None)
    if not indices:
        return messages

    result = list(messages)
    for idx in indices:
        result[idx] = _annotate_anthropic_message(copy.deepcopy(messages[idx]), cache_ttl)
    return result


def _annotate_anthropic_message(
    msg: MessageParam, cache_ttl: CacheTTL | None = None
) -> MessageParam:
    """Add cache_control to the last text/tool_result content block of an Anthropic message."""
    content = msg.get('content')
    if content is None:
        return msg

    cc = _cache_control(cache_ttl)
    if isinstance(content, str):
        msg['content'] = [{"type": "text", "text": content, "cache_control": cc}]  # type: ignore[list-item]
    elif isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') in (
                'text',
                'tool_result',
                'tool_use',
            ):
                part['cache_control'] = cc  # type: ignore[typeddict-unknown-key]
                break
    return msg
