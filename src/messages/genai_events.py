"""Structured GenAI logging events used for observability and tracing."""

from dataclasses import dataclass
from typing import Any

from openai.types.responses import ResponseUsage


@dataclass(slots=True)
class GenAIUsage:
    """Token usage information produced by a model invocation."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    def to_payload(self) -> dict[str, int]:
        payload: dict[str, int] = {}
        if self.prompt_tokens is not None:
            payload["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            payload["completion_tokens"] = self.completion_tokens
        if self.total_tokens is not None:
            payload["total_tokens"] = self.total_tokens
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenAIUsage":
        # Support both Chat Completions (prompt_tokens/completion_tokens) and
        # Responses API (input_tokens/output_tokens) field names
        cached_tokens = None
        reasoning_tokens = None
        if input_details := data.get("input_tokens_details"):
            cached_tokens = _int_or_none(input_details.get("cached_tokens"))
        if output_details := data.get("output_tokens_details"):
            reasoning_tokens = _int_or_none(output_details.get("reasoning_tokens"))
        return cls(
            prompt_tokens=_int_or_none(data.get("prompt_tokens") or data.get("input_tokens")),
            completion_tokens=_int_or_none(
                data.get("completion_tokens") or data.get("output_tokens")
            ),
            total_tokens=_int_or_none(data.get("total_tokens")),
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        # Include token details in nested structure for client-side _fetch_usage
        if self.cached_tokens is not None:
            result["input_tokens_details"] = {"cached_tokens": self.cached_tokens}
        if self.reasoning_tokens is not None:
            result["output_tokens_details"] = {"reasoning_tokens": self.reasoning_tokens}
        return result

    @classmethod
    def from_response_usage(cls, usage: ResponseUsage) -> "GenAIUsage":
        cached_tokens = None
        reasoning_tokens = None
        if usage.input_tokens_details:
            cached_tokens = usage.input_tokens_details.cached_tokens
        if usage.output_tokens_details:
            reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        return cls(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )


@dataclass(slots=True)
class GenAIChatEvent:
    """Represents a single chat completion invocation.

    Includes optional server.address and server.port per OpenTelemetry GenAI spec.
    """

    iid: str
    inference_id: str
    model: str
    provider: str | None
    input_messages: list[dict[str, Any]]
    output_messages: list[dict[str, Any]] | None = None
    usage: GenAIUsage | None = None
    streaming: bool = False
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    server_address: str | None = None  # Server hostname or IP per OTel spec
    server_port: int | None = None  # Server port per OTel spec

    def to_payload(self, uid: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uid": uid,
            "iid": self.iid,
            "inference_id": self.inference_id,
            "model": self.model,
            "provider": self.provider,
            "streaming": self.streaming,
            "input_messages": self.input_messages,
        }
        if self.output_messages is not None:
            payload["output_messages"] = self.output_messages
        if self.usage is not None:
            payload["usage"] = self.usage.to_payload()
        if self.request is not None:
            payload["request"] = self.request
        if self.response is not None:
            payload["response"] = self.response
        if self.server_address is not None:
            payload["server_address"] = self.server_address
        if self.server_port is not None:
            payload["server_port"] = self.server_port
        return payload


@dataclass(slots=True)
class GenAIToolEvent:
    """Represents execution of a tool/function as part of an invocation."""

    iid: str
    tool_id: str
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any] | list[Any] | str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_payload(self, uid: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uid": uid,
            "iid": self.iid,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "input": self.input,
        }
        if self.output is not None:
            payload["output"] = self.output
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload


@dataclass(slots=True)
class GenAIDeltaEvent:
    """Represents a agent or user text delta emitted during an invocation."""

    iid: str
    delta_id: str | None
    role: str | None
    content: str | None
    username: str | None = None
    reasoning_content: str | None = None
    usage: GenAIUsage | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    implicit: bool = (
        False  # True if this is part of implicit system messages (e.g., few-shot examples)
    )

    def to_payload(self, uid: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uid": uid,
            "iid": self.iid,
            "delta_id": self.delta_id,
            "role": self.role,
            "content": self.content,
        }
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        if self.usage is not None:
            payload["usage"] = self.usage.to_payload()
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        if self.raw is not None:
            payload["raw"] = self.raw
        if self.implicit:
            payload["implicit"] = self.implicit
        return payload


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (OverflowError, ValueError):
            return None
    return None
