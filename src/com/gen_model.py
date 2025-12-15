import json
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from com.deltas import *
from com.roles import *

if TYPE_CHECKING:
    from agentica_internal.internal_errors import GenerationError

    from com.apis import API
    from com.context import Context
    from inference import InferenceEndpoint
    from messages import GenAIDeltaEvent, InvocationNotifier


__all__ = [
    'GenModel',
    'GenModelErrorHandling',
    'GenCacheKey',
]


type SendGenErrFn = Callable[[GenerationError], Awaitable[None]]

_default_cache_key = lambda: GenCacheKey()
_default_error_handling = lambda: GenModelErrorHandling()


@dataclass
class GenModel:
    """
    Model inference / generation context. Should rename to ModelContext or something.
    """

    iid: str
    model: str
    api: 'API'
    endpoint: 'InferenceEndpoint'
    deltas: list['Delta']
    inference_rounds_count: int
    max_rounds: int | None  # Maximum number of rounds of inference (unlimited if None)
    max_invocation_tokens: int | None  # Total tokens allowed for the invocation (unlimited if None)
    max_inference_tokens: int | None  # Max tokens for a single inference (unlimited if None)
    max_completion_tokens: int | None  # Subtracted from each round of inference (unlimited if None)
    # `max_tokens` becomes `min(max_inference_tokens, max_completion_tokens)`
    send_gen_err: SendGenErrFn
    guided: bool = False
    streaming: bool = False
    cache_key: 'GenCacheKey' = field(default_factory=_default_cache_key)
    error_handling: 'GenModelErrorHandling' = field(default_factory=_default_error_handling)

    _parent: 'Context | None' = None

    def max_tokens(self) -> int | None:
        if self.max_inference_tokens is None:
            # if no max per round of inference, use the max per completion
            return self.max_completion_tokens
        if self.max_completion_tokens is None:
            # if no max per completion, use the max per round of inference
            return self.max_inference_tokens
        # if both are set, use the minimum of the two for limiting the inference round
        return min(self.max_inference_tokens, self.max_completion_tokens)

    def finish_invocation(self) -> None:
        self.max_completion_tokens = self.max_invocation_tokens
        self.inference_rounds_count = 0

    def spend_tokens(self, tokens: int) -> None:
        """Subtract the given number of tokens from our limit."""
        mct = self.max_completion_tokens
        if mct is None:
            return
        mct = max(mct - tokens, 0)
        self.max_completion_tokens = mct

    async def push_delta(self, delta: 'Delta') -> None:
        self.deltas.append(delta)
        if self._parent and self._parent.invocation:
            # Log delta events under the invocation span for all Delta types
            # Pass whether this is an implicit system message
            is_implicit = self._parent._sending_system_message
            event = _delta_to_genai_delta(self._parent.invocation, delta, implicit=is_implicit)
            if event:
                await self._parent.invocation.log_genai_delta(event)
        await self._log_append(delta)

    async def _log_append(self, delta: 'Delta') -> None:
        if not self._parent:
            return
        info: dict[str, Any] = {
            'id': delta.id,
            'role': delta.name.name,
            'content': delta.content,
        }
        if isinstance(delta, GeneratedDelta):
            info['refusal'] = delta.refusal
            if isinstance(delta.end, EndGenCallableTypes):
                tool_calls: list[Any] = []
                for tool_id, constraint, content in zip(
                    delta.end.ids, delta.end.constraints, delta.end.content
                ):
                    tool_calls.append(
                        {
                            'id': tool_id,
                            'name': constraint.executable.name,
                            'content': content,
                        }
                    )
                info['tool_calls'] = tool_calls
        if isinstance(delta.name, UserRole):
            info['username'] = delta.name.username

        await self._parent.log('delta', info)

    def use_cache_key(self, condition: Callable[['GenCacheKey'], bool]) -> str:
        """
        Use the last cache key if it satisfies the condition, otherwise create a new cache key.
        """
        # Note cache keys are created with a default first_request_time of 0 so time dependent
        # conditions are always true on the first request
        if not condition(self.cache_key):
            self.cache_key = GenCacheKey(first_request_time=time.time())
        self.cache_key.num_requests += 1
        return self.cache_key.key

    def __str__(self):
        from textwrap import indent

        strs = ['GenContext(']
        add = strs.append
        current_role = None
        tab = '\t'
        for d in self.deltas:
            role_switched = d.name != current_role
            if role_switched:
                add("\n\n") if current_role else None
                add(tab + str(d.name) + "\n")
                current_role = d.name
            if d.content:
                add(indent(d.content, tab * 2))
            if isinstance(d, GeneratedDelta):
                has_executable_results = (
                    isinstance(d.end, EndGenCallableTypes) and d.end.results is not None
                )
                if d.reasoning_content:
                    add(indent(f"<think>{d.reasoning_content}</think>", tab * 2))
                if has_executable_results:
                    for result in d.end.results:  # type: ignore
                        add(indent(f"<executable_result>{result}</executable_result>", tab * 2))
                if not (d.content or d.reasoning_content or has_executable_results):
                    add(tab + "#BLANK#")
        add("\n\n")
        add(')')
        return ''.join(strs)


@dataclass
class GenModelErrorHandling:
    """
    Error handling configuration.
    """

    max_retries: int = 1
    rate_limit_delay: float = 1
    rate_limit_exponential_base: float = 2
    rate_limit_jitter: bool = True
    read_timeout: int | None = None


@dataclass
class GenCacheKey:
    """
    A cache key for a generation.
    """

    key: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_request_time: float = 0
    num_requests: int = 0


def _delta_to_genai_delta(
    invocation: 'InvocationNotifier', delta: Delta, implicit: bool = False
) -> 'GenAIDeltaEvent | None':
    """Convert any Delta type to GenAIDeltaEvent for OpenTelemetry logging.

    Supports both base Delta and GeneratedDelta types. Events are attached to
    the invocation span as gen_ai.choice.delta events.

    Args:
        invocation: The invocation notifier to get iid from
        delta: The delta to convert (base Delta or GeneratedDelta)
        implicit: Whether this delta is part of implicit system messages (few-shot examples)
    """
    from messages import GenAIDeltaEvent, GenAIUsage

    role = delta.name.name if hasattr(delta.name, "name") else None
    username = delta.name.username if isinstance(delta.name, UserRole) else None
    usage = GenAIUsage.from_dict(delta.usage) if isinstance(delta.usage, dict) else None

    # Extract GeneratedDelta-specific fields if available
    reasoning_content = None
    tool_calls = None

    if isinstance(delta, GeneratedDelta):
        reasoning_content = delta.reasoning_content

        # Extract tool calls from GenEndCallableTypes
        if isinstance(delta.end, EndGenCallableTypes):
            tool_calls = []
            for tool_id, constraint, content in zip(
                delta.end.ids, delta.end.constraints, delta.end.content
            ):
                arguments = content
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments, default=str)
                    except TypeError:
                        arguments = str(arguments)
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": constraint.executable.name,
                            "arguments": arguments,
                        },
                    }
                )

    return GenAIDeltaEvent(
        iid=invocation.iid,
        delta_id=delta.id,
        role=role,
        content=delta.content,
        username=username,
        reasoning_content=reasoning_content,
        usage=usage,
        tool_calls=tool_calls,
        raw=None,
        implicit=implicit,
    )
