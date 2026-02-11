"""Hybrid notifier that combines legacy logging and OpenTelemetry tracing."""

import json
import logging
from collections.abc import Awaitable
from typing import Any, Callable

from agentica_internal.multiplex_protocol import MultiplexServerMessage
from agentica_internal.session_manager_messages import AllServerMessage, CreateAgentRequest
from agentica_internal.session_manager_messages.session_manager_messages import (
    InteractionEvent,
    SMInferenceUsageMessage,
    SMMonadMessage,
)
from openai.types.responses import ResponseUsage

from .genai_events import GenAIChatEvent, GenAIDeltaEvent, GenAIToolEvent, GenAIUsage
from .notifier import Notifier
from .otel_notifier import OTelNotifier

logger = logging.getLogger(__name__)


class HybridNotifier:
    """Notifier that delegates to both legacy Notifier and OTelNotifier."""

    uid: str
    send_mx_message: Callable[[MultiplexServerMessage], Awaitable[None]]
    legacy: Notifier
    otel: OTelNotifier | None

    def __init__(
        self,
        uid: str,
        send_mx_message: Callable[[MultiplexServerMessage], Awaitable[None]],
        legacy_notifier: Notifier,
        otel_notifier: OTelNotifier | None = None,
    ) -> None:
        self.uid = uid
        self.send_mx_message = send_mx_message
        self.legacy = legacy_notifier
        self.otel = otel_notifier

    def bind_invocation(self, iid: str) -> "InvocationNotifier":
        """Create a notifier bound to a specific invocation."""
        return InvocationNotifier(self, iid)

    async def append_to_log(self, msg: AllServerMessage) -> None:
        """Forward message to the legacy notifier."""
        await self.legacy.append_to_log(msg)

    async def on_inference_request(
        self,
        inference_id: str,
        iid: str,
        request_str: str,
        timeout: int | None = None,
    ) -> None:
        await self.legacy.on_inference_request(inference_id, iid, request_str, timeout)

    async def on_inference_response(
        self,
        inference_id: str,
        iid: str,
        response_str: str,
    ) -> None:
        await self.legacy.on_inference_response(inference_id, iid, response_str)

    async def on_inference_error(
        self,
        inference_id: str,
        iid: str,
        err: BaseException,
        message: str,
    ) -> None:
        await self.legacy.on_inference_error(inference_id, iid, err, message)
        if self.otel:
            await self.otel.on_inference_error(inference_id, iid, err, message)

    async def on_enter(
        self, iid: str, parent_uid: str | None = None, parent_iid: str | None = None
    ) -> None:
        await self.legacy.on_enter(iid)
        if self.otel:
            await self.otel.on_enter(iid, parent_uid=parent_uid, parent_iid=parent_iid)

    async def on_exception(self, iid: str, err: str) -> None:
        await self.legacy.on_exception(iid, err)
        if self.otel:
            await self.otel.on_exception(iid, err)

    async def on_exit(self, iid: str) -> None:
        await self.legacy.on_exit(iid)
        if self.otel:
            await self.otel.on_exit(iid)

    async def on_create_agent(self, body: CreateAgentRequest) -> None:
        await self.legacy.on_create_agent(body)
        if self.otel:
            await self.otel.on_create_agent(body)

    async def on_destroy_agent(self) -> None:
        await self.legacy.on_destroy_agent()

    async def _log_structured(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":"))
        await self.legacy.append_to_log(
            SMMonadMessage(
                uid=self.uid,
                iid=str(payload.get("iid", "unknown")),
                body=body,
            )
        )

    async def log_genai_chat(self, event: GenAIChatEvent) -> None:
        payload = event.to_payload(self.uid)
        await self._log_structured(payload)
        if self.otel:
            await self.otel.on_genai_chat(event)

    async def log_genai_tool(self, event: GenAIToolEvent) -> None:
        payload = event.to_payload(self.uid)
        await self._log_structured(payload)
        if self.otel:
            await self.otel.on_genai_tool(event)

    async def log_genai_delta(self, event: GenAIDeltaEvent) -> None:
        payload = event.to_payload(self.uid)
        await self._log_structured(payload)
        if self.otel:
            await self.otel.on_genai_delta(event)

    async def log_interaction(
        self,
        iid: str,
        event: InteractionEvent,
    ) -> None:
        await self.legacy.log_interaction(iid, event)
        if self.otel:
            await self.otel.on_interaction_event(iid, event)


class InvocationNotifier:
    """Helper bound to a specific invocation for structured logging."""

    def __init__(self, parent: HybridNotifier, iid: str) -> None:
        self._parent = parent
        self.iid = iid
        self._model: str | None = None
        self._provider: str | None = None
        self._current_request: dict[str, Any] | None = None
        self._current_inference_id: str | None = None
        self._streaming: bool = False
        self._server_address: str | None = None
        self._server_port: int | None = None

    @property
    def uid(self) -> str:
        return self._parent.uid

    async def log_message(self, msg: AllServerMessage) -> None:
        await self._parent.append_to_log(msg)

    async def log_monad(self, body: str) -> None:
        await self._parent.append_to_log(
            SMMonadMessage(
                uid=self.uid,
                iid=self.iid,
                body=body,
            )
        )

    async def log_usage(self, usage: ResponseUsage) -> None:
        await self.log_message(
            SMInferenceUsageMessage(
                uid=self.uid, iid=self.iid, usage=usage.model_dump(exclude_none=True)
            )
        )

    async def log_invocation_error(self, error_type: str, error_message: str | None = None) -> None:
        from agentica_internal.session_manager_messages.session_manager_messages import (
            SMInvocationErrorMessage,
        )

        await self._parent.append_to_log(
            SMInvocationErrorMessage(
                uid=self.uid,
                iid=self.iid,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def start_inference(
        self,
        *,
        inference_id: str,
        request: dict[str, Any],
        streaming: bool,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> None:
        self._current_inference_id = inference_id
        self._current_request = request
        self._streaming = streaming
        self._server_address = server_address
        self._server_port = server_port
        if self._model is None:
            self._model = _as_str(request.get("model"))
        if self._provider is None and self._model:
            self._provider = _derive_provider(self._model)

    async def log_genai_chat(self, event: GenAIChatEvent) -> None:
        await self._parent.log_genai_chat(self._apply_defaults(event))

    async def log_genai_tool(self, event: GenAIToolEvent) -> None:
        await self._parent.log_genai_tool(event)

    async def log_genai_delta(self, event: GenAIDeltaEvent) -> None:
        await self._parent.log_genai_delta(event)

    async def log_interaction(
        self,
        event: InteractionEvent,
    ) -> None:
        await self._parent.log_interaction(self.iid, event)

    def with_agent_metadata(self, *, model: str | None, provider: str | None) -> None:
        if model:
            self._model = model
        if provider:
            self._provider = provider

    def create_chat_event(
        self,
        *,
        inference_id: str | None = None,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        output_messages: list[dict[str, Any]] | None = None,
        usage: ResponseUsage | None = None,
        streaming: bool | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> GenAIChatEvent:
        req = request or self._current_request or {}
        messages = _ensure_list_of_messages(req.get("messages"))
        inf_id = inference_id or self._current_inference_id or "unknown"
        out_msgs = output_messages
        if out_msgs is None and response is not None:
            out_msgs = _extract_output_messages(response)
        usage_obj = usage
        if usage_obj is None and response is not None:
            usage_obj = _extract_usage(response)
        stream_flag = streaming if streaming is not None else self._streaming

        return GenAIChatEvent(
            iid=self.iid,
            inference_id=inf_id,
            model=self._model or "",
            provider=self._provider,
            input_messages=messages,
            output_messages=out_msgs,
            usage=usage and GenAIUsage.from_response_usage(usage) or None,
            streaming=stream_flag,
            request=req if req else None,
            response=response,
            server_address=server_address or self._server_address,
            server_port=server_port or self._server_port,
        )

    def create_delta_event(self, chunk: dict[str, Any]) -> GenAIDeltaEvent | None:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
        if not isinstance(delta, dict):
            delta = {}
        role = _as_str(delta.get("role"), None)
        username = _as_str(delta.get("username"), None)
        content = delta.get("content")
        if isinstance(content, list):
            # Some providers emit list of content parts
            try:
                content = json.dumps(content, separators=(",", ":"))
            except TypeError:
                content = str(content)
        elif content is not None and not isinstance(content, str):
            content = str(content)
        tool_calls = delta.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            tool_calls = [tool_calls]  # type: ignore[list-item]
        usage = choice.get("usage") if isinstance(choice, dict) else None
        usage_obj = _extract_usage(usage) if isinstance(usage, dict) else None
        return GenAIDeltaEvent(
            iid=self.iid,
            delta_id=_as_str(chunk.get("id"), None),
            role=role,
            content=content,
            username=username,
            reasoning_content=_as_str(delta.get("reasoning_content"), None),
            usage=usage_obj,
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
            raw=chunk,
        )

    def current_inference_id(self) -> str | None:
        return self._current_inference_id

    def current_request(self) -> dict[str, Any] | None:
        return self._current_request

    def streaming(self) -> bool:
        return self._streaming

    def _apply_defaults(self, event: GenAIChatEvent) -> GenAIChatEvent:
        model = (
            event.model
            or self._model
            or _as_str(event.request.get("model") if event.request else None)
        )
        provider = event.provider if event.provider is not None else self._provider
        if not provider and model:
            provider = _derive_provider(model)
        return GenAIChatEvent(
            iid=event.iid,
            inference_id=event.inference_id,
            model=model or "",
            provider=provider,
            input_messages=event.input_messages,
            output_messages=event.output_messages,
            usage=event.usage,
            streaming=event.streaming,
            request=event.request,
            response=event.response,
        )


def _ensure_list_of_messages(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    return []


def _extract_output_messages(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        messages: list[dict[str, Any]] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                messages.append(message)
            else:
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    # Convert delta-style message to message structure if possible
                    role = delta.get("role")
                    content = delta.get("content")
                    if role or content:
                        msg = {}
                        if isinstance(role, str):
                            msg["role"] = role
                        if content is not None:
                            msg["content"] = content
                        messages.append(msg)
        if messages:
            return messages
    role = response.get("role")
    content = response.get("content")
    if isinstance(role, str) or content is not None:
        message: dict[str, Any] = {}
        if isinstance(role, str):
            message["role"] = role
        if content is not None:
            message["content"] = content
        tool_calls = response.get("tool_calls")
        if isinstance(tool_calls, list):
            message["tool_calls"] = tool_calls
        return [message]
    return None


def _extract_usage(data: dict[str, Any] | None) -> GenAIUsage | None:
    if not data:
        return None
    return GenAIUsage.from_dict(data)


def _derive_provider(model: str) -> str | None:
    if ":" in model:
        return model.split(":", 1)[0]
    if "/" in model:
        prefix = model.split("/", 1)[0]
        return prefix if prefix else None
    return None


def _as_str(value: object, default: str | None = "") -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return default
