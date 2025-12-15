from typing import TYPE_CHECKING

from com.deltas import *
from com.roles import *

if TYPE_CHECKING:
    from openai.types import chat  # noqa: F401

__all__ = [
    'convert_from_deltas',
]


# === Messages ===


def make_system(delta: Delta) -> 'chat.ChatCompletionSystemMessageParam':
    """
    Makes a system message.

    Args:
        delta: The delta to add.

    Returns:
        The system message.
    """
    from openai.types.chat import ChatCompletionSystemMessageParam

    if delta.content is None:
        raise ValueError("System message content is required")
    return ChatCompletionSystemMessageParam(
        content=delta.content,
        role='system',
    )


def make_user(delta: Delta) -> 'chat.ChatCompletionUserMessageParam':
    """
    Makes a user message.

    Args:
        delta: The delta to convert.

    Returns:
        The user message.
    """
    from openai.types.chat import ChatCompletionUserMessageParam

    if delta.content is None:
        raise ValueError("User message content is required")
    role = delta.name
    assert isinstance(role, UserRole)
    user_msg = ChatCompletionUserMessageParam(
        content=delta.content,
        role='user',
    )
    if role.username:
        user_msg['name'] = role.username
    return user_msg


def make_assistant(
    delta: Delta,
) -> list['chat.ChatCompletionAssistantMessageParam | chat.ChatCompletionToolMessageParam']:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionToolMessageParam,
    )

    """
    Makes an assistant message.

    Args:
        delta: The delta to convert.

    Returns:
        The assistant message.
    """
    results = []
    content = delta.content
    if isinstance(delta, GeneratedDelta) and delta.reasoning_content is not None:
        content = content or ''
        content += delta.reasoning_content
    res = ChatCompletionAssistantMessageParam(
        role='assistant',
        content=content,
        refusal=delta.refusal if isinstance(delta, GeneratedDelta) else None,
    )
    results.append(res)
    if isinstance(delta, GeneratedDelta):
        if audio := delta.audio:
            if not ('id' in audio and isinstance(audio['id'], str)):
                raise ValueError("Audio message must have an string id")
            results[0]['audio'] = {'id': audio['id']}

    if call_end := delta.get_end(EndGenCallableTypes):
        from .utils_json import make_assistant_tool_call_blocks

        results[0]['tool_calls'] = make_assistant_tool_call_blocks(call_end)
        if call_end.results:
            for _, id, _, result in zip(
                call_end.constraints, call_end.ids, call_end.content, call_end.results
            ):
                results.append(
                    ChatCompletionToolMessageParam(
                        role='tool',
                        content=result if isinstance(result, str) else str(result),
                        tool_call_id=id,
                    )
                )
    return results


def make_developer(delta: Delta) -> 'chat.ChatCompletionDeveloperMessageParam':
    """
    Makes a developer message.

    Args:
        delta: The delta to convert.

    Returns:
        The developer message.
    """
    from openai.types.chat import ChatCompletionDeveloperMessageParam

    if delta.content is None:
        raise ValueError("Developer message content is required")
    return ChatCompletionDeveloperMessageParam(
        content=delta.content,
        role='developer',
    )


def convert_from_delta(delta: Delta) -> list['chat.ChatCompletionMessageParam']:
    """
    Converts a single delta to a list of messages.

    Args:
        delta: The delta to convert.

    Returns:
        The list of messages.
    """
    if isinstance(delta, GeneratedDelta):
        return make_assistant(delta)  # type: ignore
    match delta.name:
        case SystemRole():
            return [make_system(delta)]
        case AgentRole():
            return make_assistant(delta)  # type: ignore
        case UserRole():
            return [make_user(delta)]
        case _:
            raise ValueError(f"Invalid delta: {delta.name}")


def convert_from_role(role: GenRole) -> str:
    """
    Returns the role as a string.

    Args:
        role: The role to convert.

    Returns:
        The role as a string.
    """
    match role:
        case UserRole():
            return 'user'
        case AgentRole():
            return 'assistant'
        case SystemRole():
            return 'system'
        case _:
            raise ValueError(f"Invalid role: {role}")


def convert_from_deltas(deltas: list[Delta]) -> list['chat.ChatCompletionMessageParam']:
    """
    Converts a list of deltas to a list of messages for the OpenAI Chat Completions API specifically for OpenAI providers.

    Args:
        deltas: The list of deltas to convert.

    Returns:
        The list of messages with the system message at the beginning.
    """
    from openai.types.chat import ChatCompletionMessageParam

    messages = []
    current_message: ChatCompletionMessageParam | None = None
    current_role: GenRole | None = None
    for delta in deltas:
        if current_message is None or current_role != delta.name:
            current_role = delta.name
            current_messages = convert_from_delta(delta)
            current_message = current_messages[-1]
            messages.extend(current_messages)
            continue

        assert current_message is not None
        # Glue deltas together
        if delta.content:
            prev = current_message.get('content', '') or ''
            assert isinstance(prev, str)
            current_message['content'] = prev + delta.content
        if isinstance(delta, GeneratedDelta) and delta.reasoning_content is not None:
            prev = current_message.get('content', '') or ''
            assert isinstance(prev, str)
            current_message['content'] = prev + delta.reasoning_content
    return messages
