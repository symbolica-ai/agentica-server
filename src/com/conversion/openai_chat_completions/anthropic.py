from typing import TYPE_CHECKING, Any

from com.deltas import *
from com.roles import *

if TYPE_CHECKING:
    from openai.types import chat  # noqa: F401

__all__ = [
    'make_text_block',
    'make_system_block',
    'make_user_block',
    'make_assistant_text_block',
    'make_tool_text_block',
    'make_tool_result',
    'make_assistant_tool_results',
    'make_assistant_tool_blocks',
    'make_system',
    'make_user',
    'make_assistant_tool',
    'convert_from_role',
    'convert_from_deltas',
]


# === Blocks ===


def make_text_block(content: str) -> 'chat.ChatCompletionContentPartTextParam':
    """
    Makes a text content block.

    Args:
        content: The content to add.

    Returns:
        The text content block.
    """
    from openai.types.chat import ChatCompletionContentPartTextParam

    return ChatCompletionContentPartTextParam(type='text', text=content)


def make_system_block(delta: Delta) -> 'chat.ChatCompletionContentPartTextParam':
    """
    Makes a text content block for a system message.

    Args:
        delta: The delta to convert.

    Returns:
        The text content block.
    """
    if delta.content is None:
        raise ValueError("System message content is required")
    return make_text_block(delta.content)


def make_user_block(delta: Delta) -> 'chat.ChatCompletionContentPartTextParam':
    """
    Makes a text content block for a user message.
    Checks if the user role has a name attached to it and raises an error if they do as this is not supported by Anthropic.

    Args:
        delta: The delta to convert.

    Returns:
        The text content block.
    """
    if delta.content is None:
        raise ValueError("User message content is required")
    role = delta.name
    assert isinstance(role, UserRole)
    if role.username:
        raise ValueError("Anthropic does not support names in user messages")
    return make_text_block(delta.content)


def make_assistant_text_block(delta: Delta) -> 'chat.ChatCompletionContentPartTextParam':
    """
    Makes a text content block for an assistant message.

    Args:
        delta: The delta to convert.

    Returns:
        The text content block.
    """
    return make_text_block(delta.content)


def make_tool_text_block(result: str | Any) -> 'chat.ChatCompletionContentPartTextParam':
    """
    Makes text content block for a tool result. The tool result is stringified to text.

    Args:
        result: The result to convert.

    Returns:
        The text content block.
    """
    return make_text_block(result if isinstance(result, str) else str(result))


def make_tool_result(id: str, result: str | Any) -> 'chat.ChatCompletionToolMessageParam':
    """
    Makes a tool result.

    Args:
        id: The id of the tool call.
        result: The result to convert.

    Returns:
        The tool result message param.
    """
    from openai.types.chat import ChatCompletionToolMessageParam

    return ChatCompletionToolMessageParam(
        role='tool',
        content=[make_tool_text_block(result)],
        tool_call_id=id,
    )


def make_assistant_tool_results(
    end: EndGenCallableTypes,
) -> list['chat.ChatCompletionToolMessageParam']:
    """
    Makes tool results associated to the tool calls of a callable end for a generated delta.
    Tool calls with the same id will be appended as blocks of the same tool result.

    Args:
        end: The end to convert.

    Returns:
        The tool results.
    """
    tool_results_content = []

    # If tool calls were executed
    if end.results is not None:
        for id, result in zip(end.ids, end.results):
            # If the tool result is for a new tool call id, create a new tool result content block
            if not tool_results_content or tool_results_content[-1]['tool_call_id'] != id:
                tool_results_content.append(make_tool_result(id, result))
                continue

            # Otherwise, append the result to the last tool result content block
            tool_results_content[-1]['content'].append(make_tool_text_block(result))
    return tool_results_content


def make_assistant_tool_blocks(
    delta: Delta,
) -> tuple[
    'chat.ChatCompletionContentPartTextParam',
    list['chat.ChatCompletionMessageToolCallParam'],
    list['chat.ChatCompletionToolMessageParam'],
]:
    """
    Makes
    - a text content block for an assistant message,
    - tool call blocks for an assistant message from a generated delta with a callable end and
    - and any tool result messages associated to those tool calls

    Args:
        delta: The delta to convert.

    Returns:
        The text content block, tool calls, and tool results.
    """
    call_end = delta.get_end(EndGenCallableTypes)
    if not call_end:
        return make_assistant_text_block(delta), [], []
    from .utils_json import make_assistant_tool_call_blocks

    return (
        make_assistant_text_block(delta),
        make_assistant_tool_call_blocks(call_end),
        make_assistant_tool_results(call_end),
    )


# === Messages ===


def make_system(delta: Delta) -> 'chat.ChatCompletionSystemMessageParam':
    """
    Makes a system message.

    Args:
        delta: The delta to convert.

    Returns:
        The system message.
    """
    from openai.types.chat import ChatCompletionSystemMessageParam

    if delta.content is None:
        raise ValueError("System message content is required")
    return ChatCompletionSystemMessageParam(content=[make_system_block(delta)], role='system')


def make_user(delta: Delta) -> 'chat.ChatCompletionUserMessageParam':
    """
    Makes a user message.

    Args:
        delta: The delta to convert.

    Returns:
        The user message.
    """
    from openai.types.chat import ChatCompletionUserMessageParam

    return ChatCompletionUserMessageParam(content=[make_user_block(delta)], role='user')


def make_assistant_tool(
    delta: Delta,
) -> list['chat.ChatCompletionAssistantMessageParam | chat.ChatCompletionToolMessageParam']:
    """
    Makes an assistant message. Also makes tool result messages associated to the tool calls of a callable end for a generated delta.

    Args:
        delta: The delta to convert.

    Returns:
        The assistant message.
    """
    from openai.types.chat import ChatCompletionAssistantMessageParam

    res = ChatCompletionAssistantMessageParam(
        role='assistant',
        content=[make_assistant_text_block(delta)],
    )
    end = delta.get_end(EndGenCallableTypes)
    if not end:
        return [res]
    from .utils_json import make_assistant_tool_call_blocks

    res['tool_calls'] = make_assistant_tool_call_blocks(end)
    tool_res = make_assistant_tool_results(end)
    return [res] + tool_res


def convert_from_role(delta: Delta) -> str:
    """
    If the delta is a generated delta, it returns 'assistant'. Otherwise, it returns the role of the delta as a string.

    Args:
        delta: The delta to convert.

    Returns:
        The role.
    """
    if isinstance(delta, GeneratedDelta):
        return 'assistant'
    match delta.name:
        case UserRole():
            return 'user'
        case AgentRole():
            return 'assistant'
        case SystemRole():
            return 'system'
        case _:
            raise ValueError(f"Invalid role: {delta.name}")


def convert_from_deltas(deltas: list[Delta]) -> list['chat.ChatCompletionMessageParam']:
    """
    Converts a list of deltas to a list of messages for the OpenAI Chat Completions API specifically for Anthropic providers.

    This also includes features for Anthropic-specific caching:
    - Having one system message with multiple content blocks
    - Only switching ChatCompletionMessageParam type when roles change, otherwise appending content blocks

    Args:
        deltas: The list of deltas to convert.

    Returns:
        The list of messages with the system message at the beginning.
    """
    messages = []
    system_message: 'chat.ChatCompletionSystemMessageParam | None' = None
    current_message = None

    for delta in deltas:
        delta_role: str = convert_from_role(delta)
        current_role: str | None = current_message['role'] if current_message else None

        # Create new message blocks if changing role or if it is the first message (system or otherwise)
        if ((current_message is None or delta_role != current_role) and delta_role != 'system') or (
            system_message is None and delta_role == 'system'
        ):
            # Convert system message
            if delta_role == 'system':
                system_message = make_system(delta)

            # Convert user message
            elif delta_role == 'user':
                messages.append(make_user(delta))
                current_message = messages[-1]

            # Convert assistant message
            elif delta_role == 'assistant':
                messages.extend(make_assistant_tool(delta))
                current_message = messages[-1]
            continue

        # Glue deltas together by creating and appending message blocks
        if delta_role == 'system':
            system_message['content'].append(make_system_block(delta))

        elif delta_role == 'user':
            current_message['content'].append(make_user_block(delta))

        elif delta_role == 'assistant':
            content, tool_calls, tool_results = make_assistant_tool_blocks(delta)
            current_message['content'].append(content)

            # Add tool call blocks to assistant message if not already present
            if 'tool_calls' not in current_message:
                current_message['tool_calls'] = []
            current_message['tool_calls'].extend(tool_calls)

            # Add tool result messages and change current message
            if tool_results:
                messages.extend(tool_results)
                current_message = messages[-1]

    if system_message is not None:
        messages.insert(0, system_message)
    return messages
