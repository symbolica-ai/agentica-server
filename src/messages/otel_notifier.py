"""OpenTelemetry-based notifier for distributed tracing."""

import json
import logging
import os
from typing import Any, Callable

from agentica_internal.session_manager_messages import CreateAgentRequest
from agentica_internal.session_manager_messages.session_manager_messages import (
    InteractionCodeBlock,
    InteractionEvent,
    InteractionExecuteResult,
)
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .genai_events import GenAIChatEvent, GenAIDeltaEvent, GenAIToolEvent

logger = logging.getLogger(__name__)

_CAPTURE_CONTENT: bool = os.getenv("OTEL_GENAI_CAPTURE_CONTENT", "false").lower() in {
    "1",
    "true",
    "yes",
}


class OTelNotifier:
    """OpenTelemetry-based notifier that creates spans and events for tracing.

    This notifier runs alongside the legacy Notifier to provide distributed tracing
    without breaking existing logging functionality.
    """

    uid: str
    tracer: trace.Tracer
    _session_span: trace.Span | None  # Session span, created lazily on first invocation
    _invocation_spans: dict[str, trace.Span]
    _agent_span: trace.Span | None
    _llm_spans: dict[str, trace.Span]  # inference_id -> span
    _tool_spans: dict[str, trace.Span]  # tool_id -> span
    _register_invocation_span: Callable | None  # Callback to register invocation spans
    _get_invocation_span: (
        Callable[[str, str], trace.Span | None] | None
    )  # Callback to get invocation span by (uid, iid)
    _parent_context: Any  # Parent context from WebSocket, used if no parent invocation
    _model_info: dict[str, str]  # Model information for session span attributes

    def __init__(
        self,
        uid: str,
        tracer: trace.Tracer,
        parent_context: Any = None,
        model_info: dict[str, str] | None = None,
        register_invocation_span: Callable | None = None,
        get_invocation_span: Callable[[str, str], trace.Span | None] | None = None,
    ):
        """Initialize OTelNotifier.

        Args:
            uid: Agent/function unique identifier
            tracer: OpenTelemetry tracer instance
            parent_context: Optional parent context from WebSocket connection
            model_info: Optional model information for session span attributes
            register_invocation_span: Optional callback to register invocation spans
            get_invocation_span: Optional callback to get invocation span by (uid, iid)
        """
        self.uid = uid
        self.tracer = tracer
        self._session_span = None  # Created lazily on first invocation
        self._invocation_spans = {}
        self._agent_span = None
        self._llm_spans = {}
        self._tool_spans = {}
        self._register_invocation_span = register_invocation_span
        self._get_invocation_span = get_invocation_span
        self._parent_context = parent_context
        self._model_info = model_info or {}

    async def on_create_agent(self, body: CreateAgentRequest) -> None:
        """Store agent creation configuration for session span attributes.

        The session span will be created lazily on first invocation, so we
        just store the configuration details here.

        Args:
            body: Agent creation request details
        """
        # Store config for later use when session span is created
        self._model_info["agent.config.streaming"] = str(body.streaming)
        if body.doc:
            self._model_info["agent.config.doc_length"] = str(len(body.doc))
        if body.system:
            self._model_info["agent.config.system_length"] = str(len(body.system))

    async def on_enter(
        self, iid: str, parent_uid: str | None = None, parent_iid: str | None = None
    ) -> None:
        """Create a new span for invocation enter.

        On the first invocation, also creates the session span. The session span will
        nest under the parent invocation if parent_uid/parent_iid are provided.

        Args:
            iid: Invocation unique identifier
            parent_uid: Optional parent agent UID (for nesting and metadata)
            parent_iid: Optional parent invocation ID (for nesting and metadata)
        """
        # On first invocation, create the session span
        if self._session_span is None:
            session_parent_ctx = None

            # If this agent was spawned from another agent's invocation, nest under that
            if parent_uid and parent_iid and self._get_invocation_span:
                parent_inv_span = self._get_invocation_span(parent_uid, parent_iid)
                if parent_inv_span:
                    session_parent_ctx = trace.set_span_in_context(parent_inv_span)

            session_attributes = {
                "gen_ai.agent.id": self.uid,
            }
            session_attributes.update(self._model_info)

            # Add parent agent ID as metadata if provided
            if parent_uid:
                session_attributes["gen_ai.parent.agent.id"] = parent_uid

            self._session_span = self.tracer.start_span(
                name="gen_ai.agent.session",
                context=session_parent_ctx,
                attributes=session_attributes,
            )
            logger.debug(f"OTel: Started session span for agent {self.uid}")

        # Invocation spans always nest under their own agent's session span
        ctx = trace.set_span_in_context(self._session_span)

        span_attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.id": self.uid,
        }
        if parent_uid:
            span_attributes["gen_ai.parent.agent.id"] = parent_uid
        if parent_iid:
            span_attributes["gen_ai.parent.invocation.id"] = parent_iid

        span = self.tracer.start_span(
            name=f"gen_ai.invocation",
            context=ctx,
            attributes=span_attributes,
        )
        self._invocation_spans[iid] = span

        if self._register_invocation_span:
            self._register_invocation_span(self.uid, iid, span)

        logger.debug(f"OTel: Started span for invocation {iid}")

    async def on_exit(self, iid: str) -> None:
        """End the invocation span with success status.

        Args:
            iid: Invocation unique identifier
        """
        span = self._invocation_spans.pop(iid, None)
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()
            logger.debug(f"OTel: Ended span for invocation {iid}")

    async def on_exception(self, iid: str, err: str) -> None:
        """Record error in invocation span.

        Args:
            iid: Invocation unique identifier
            err: Error message
        """
        span = self._invocation_spans.get(iid)
        if span:
            # Create an exception from the error string
            exception = Exception(err)
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(err)))
            span.set_attribute("error.type", type(exception).__name__)
            span.set_attribute("error.message", str(err))
            logger.debug(f"OTel: Recorded exception for invocation {iid}")

    def end_session_span(self) -> None:
        """End the session span.

        Called when the agent WebSocket connection closes.
        """
        if self._session_span:
            self._session_span.set_status(Status(StatusCode.OK))
            self._session_span.end()
            logger.debug(f"OTel: Ended session span for agent {self.uid}")
            self._session_span = None

    async def on_inference_error(
        self,
        inference_id: str,
        iid: str,
        err: BaseException,
        message: str,
    ) -> None:
        """Record inference error on the LLM span and end it.

        Errors are recorded using span.record_exception() and span.set_status(),
        following OpenTelemetry semantic conventions for error handling.
        The span is then ended since the inference is complete (with error).

        Args:
            inference_id: Inference request unique identifier
            iid: Invocation unique identifier
            err: Exception that occurred
            message: Error message
        """
        # Mark LLM span as error and end it
        llm_span = self._llm_spans.pop(inference_id, None)
        if llm_span:
            llm_span.record_exception(err)
            llm_span.set_status(Status(StatusCode.ERROR, str(message)))
            llm_span.set_attribute("error.type", type(err).__name__)
            llm_span.set_attribute("error.message", str(message))
            llm_span.end()
            logger.debug(f"OTel: Ended gen_ai.chat span with error for inference_id={inference_id}")

    async def on_interaction_event(
        self,
        iid: str,
        event: InteractionEvent,
    ) -> None:
        logger.debug("OTel: on_interaction_event called - iid=%s, event=%s", iid, event)
        parent_span = self._invocation_spans.get(iid)
        if parent_span is None:
            logger.warning(
                f"OTel: No parent span found for iid={iid}, available iids: {list(self._invocation_spans.keys())}"
            )
            return
        logger.debug(f"OTel: Found parent span for iid={iid}")

        # Use standard gen_ai.execute_tool spans for REPL code execution
        if isinstance(event, InteractionCodeBlock):
            # Start a new tool span for code execution
            tool_id = event.exec_id
            ctx = trace.set_span_in_context(parent_span)

            tool_span = self.tracer.start_span(
                name="execute_tool python_repl",
                context=ctx,
                kind=SpanKind.INTERNAL,
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.agent.id": self.uid,
                    "gen_ai.tool.name": "python_repl",
                    "gen_ai.tool.type": "extension",
                },
            )
            if _CAPTURE_CONTENT:
                tool_span.set_attribute(
                    "gen_ai.tool.call.arguments",
                    json.dumps({"code": event.code}, separators=(",", ":")),
                )
            # Store span for later completion
            self._tool_spans[tool_id] = tool_span
            logger.debug(f"OTel: Created gen_ai.execute_tool span for tool_id={tool_id}")

        elif isinstance(event, InteractionExecuteResult):
            # Find and complete the most recent tool span
            if self._tool_spans:
                tool_id = event.exec_id
                tool_span = self._tool_spans.get(tool_id)
                if tool_span:
                    if _CAPTURE_CONTENT:
                        tool_span.set_attribute(
                            "gen_ai.tool.call.result",
                            json.dumps({"result": event.result}, separators=(",", ":")),
                        )
                    tool_span.set_status(Status(StatusCode.OK))
                    tool_span.end()
                    self._tool_spans.pop(tool_id, None)
                    logger.debug(f"OTel: Ended gen_ai.execute_tool span for tool_id={tool_id}")
                else:
                    logger.warning(f"OTel: No tool span found for tool_id={tool_id}")
            else:
                logger.warning(f"OTel: No tool spans available for result payload")

    async def on_genai_chat(self, event: GenAIChatEvent) -> None:
        logger.debug(
            f"OTel: on_genai_chat called - inference_id={event.inference_id}, model={event.model}, streaming={event.streaming}"
        )
        span = self._ensure_llm_span(
            inference_id=event.inference_id,
            iid=event.iid,
            provider=event.provider,
            model=event.model,
            streaming=event.streaming,
            input_messages=event.input_messages,
            server_address=event.server_address,
            server_port=event.server_port,
        )
        if span is None:
            logger.warning(f"OTel: Failed to create LLM span for inference_id={event.inference_id}")
            return
        logger.debug(f"OTel: Created gen_ai.chat span for inference_id={event.inference_id}")

        if _CAPTURE_CONTENT and event.output_messages is not None:
            self._add_message_events(span, event.output_messages, "output")

        if event.usage is not None:
            if event.usage.prompt_tokens is not None:
                span.set_attribute("gen_ai.usage.prompt_tokens", event.usage.prompt_tokens)
            if event.usage.completion_tokens is not None:
                span.set_attribute("gen_ai.usage.completion_tokens", event.usage.completion_tokens)
            if event.usage.total_tokens is not None:
                span.set_attribute("gen_ai.usage.total_tokens", event.usage.total_tokens)

        # End span when we have complete output (both streaming and non-streaming)
        if event.output_messages is not None:
            span.set_status(Status(StatusCode.OK))
            span.end()
            _ = self._llm_spans.pop(event.inference_id, None)
            logger.debug(
                f"OTel: Ended gen_ai.chat span for inference_id={event.inference_id} (streaming={event.streaming})"
            )

    async def on_genai_tool(self, event: GenAIToolEvent) -> None:
        span = self._ensure_tool_span(event.tool_id, event.iid, event.tool_name)
        if span is None:
            return

        if _CAPTURE_CONTENT:
            span.set_attribute(
                "gen_ai.tool.call.arguments",
                json.dumps(event.input, default=str, separators=(",", ":")),
            )

        if event.output is not None and _CAPTURE_CONTENT:
            span.set_attribute(
                "gen_ai.tool.call.result",
                json.dumps(event.output, default=str, separators=(",", ":")),
            )

        if event.error_type or event.error_message:
            message = event.error_message or "Tool execution error"
            span.set_status(Status(StatusCode.ERROR, message))
            if event.error_type:
                span.set_attribute("error.type", event.error_type)
            span.set_attribute("error.message", message)
        else:
            span.set_status(Status(StatusCode.OK))

        span.end()
        _ = self._tool_spans.pop(event.tool_id, None)

    async def on_genai_delta(self, event: GenAIDeltaEvent) -> None:
        """Record a text delta event (content gated by OTEL_GENAI_CAPTURE_CONTENT)."""
        span = self._invocation_spans.get(event.iid)
        if span is None:
            return

        # Use experimental event name for streaming chunks
        # This matches emerging patterns but is not yet standardized
        attributes: dict[str, Any] = {}
        if event.delta_id is not None:
            attributes["gen_ai.choice.id"] = event.delta_id
        if event.role is not None:
            attributes["gen_ai.choice.role"] = event.role
        if event.username is not None:
            attributes["gen_ai.choice.name"] = event.username

        # Potentially sensitive fields are only recorded when capture is enabled
        if _CAPTURE_CONTENT:
            if event.content is not None:
                attributes["gen_ai.choice.content"] = event.content
            if event.reasoning_content is not None:
                attributes["gen_ai.choice.reasoning_content"] = event.reasoning_content
            if event.tool_calls is not None:
                attributes["gen_ai.choice.tool_calls"] = json.dumps(
                    event.tool_calls, default=str, separators=(",", ":")
                )
        if event.usage is not None:
            usage_payload = event.usage.to_payload()
            for key, value in usage_payload.items():
                attributes[f"gen_ai.usage.{key}"] = value
        if _CAPTURE_CONTENT and event.raw is not None:
            attributes["gen_ai.choice.raw"] = json.dumps(
                event.raw, default=str, separators=(",", ":")
            )
        # Mark if this is an implicit system message (few-shot examples, etc.)
        if event.implicit:
            attributes["gen_ai.choice.implicit"] = True

        # Using 'gen_ai.choice.delta' as experimental event name
        span.add_event("gen_ai.choice.delta", attributes=attributes)

    def _map_provider_name(self, provider: str | None, model: str) -> str | None:
        """Map provider/model to OpenTelemetry GenAI provider name.

        Returns one of the spec-compliant values: openai, anthropic, aws.bedrock,
        azure.ai.openai, azure.ai.inference, cohere, deepseek, gcp.gemini,
        gcp.vertex_ai, gcp.gen_ai, groq, ibm.watsonx.ai, mistral_ai, perplexity, x_ai.

        See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
        """
        if not provider:
            return None

        provider_lower = provider.lower()
        model_lower = model.lower()

        # Map openrouter to actual provider based on model
        if provider_lower == "openrouter":
            if "gpt" in model_lower or "openai" in model_lower:
                return "openai"
            elif "claude" in model_lower or "anthropic" in model_lower:
                return "anthropic"
            elif "gemini" in model_lower or "palm" in model_lower:
                return "gcp.gemini"
            elif "mistral" in model_lower:
                return "mistral_ai"
            elif "deepseek" in model_lower:
                return "deepseek"
            elif "cohere" in model_lower:
                return "cohere"
            elif "groq" in model_lower:
                return "groq"
            elif "perplexity" in model_lower:
                return "perplexity"
            elif "grok" in model_lower or "x.ai" in model_lower:
                return "x_ai"
            # Default to openai for openrouter
            return "openai"

        # Direct provider mappings per OTel spec
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "bedrock": "aws.bedrock",
            "aws.bedrock": "aws.bedrock",
            "aws_bedrock": "aws.bedrock",
            "azure": "azure.ai.openai",
            "azure.openai": "azure.ai.openai",
            "azure_openai": "azure.ai.openai",
            "azure.ai.openai": "azure.ai.openai",
            "azure.ai.inference": "azure.ai.inference",
            "cohere": "cohere",
            "deepseek": "deepseek",
            "gemini": "gcp.gemini",
            "gcp.gemini": "gcp.gemini",
            "vertex": "gcp.vertex_ai",
            "vertexai": "gcp.vertex_ai",
            "vertex_ai": "gcp.vertex_ai",
            "gcp.vertex_ai": "gcp.vertex_ai",
            "gcp": "gcp.gen_ai",
            "gcp.gen_ai": "gcp.gen_ai",
            "groq": "groq",
            "watsonx": "ibm.watsonx.ai",
            "ibm.watsonx.ai": "ibm.watsonx.ai",
            "mistral": "mistral_ai",
            "mistral_ai": "mistral_ai",
            "perplexity": "perplexity",
            "xai": "x_ai",
            "x_ai": "x_ai",
            "x.ai": "x_ai",
        }

        return provider_map.get(provider_lower)

    def _ensure_llm_span(
        self,
        *,
        inference_id: str,
        iid: str,
        provider: str | None,
        model: str,
        streaming: bool,
        input_messages: list[dict[str, Any]],
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> trace.Span | None:
        existing = self._llm_spans.get(inference_id)
        if existing:
            return existing

        parent_span = self._invocation_spans.get(iid) if iid else self._session_span
        ctx = trace.set_span_in_context(parent_span) if parent_span else None

        # Build initial attributes per OTel GenAI spec
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.agent.id": self.uid,
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
        }

        # Add server.address (REQUIRED per spec) and server.port (CONDITIONALLY REQUIRED)
        if server_address:
            attributes["server.address"] = server_address
        if server_port:
            attributes["server.port"] = server_port

        span = self.tracer.start_span(
            name=f"chat {model}",
            context=ctx,
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )
        if _CAPTURE_CONTENT:
            self._add_message_events(span, input_messages, "input")

        provider_name = self._map_provider_name(provider, model)
        if provider_name:
            span.set_attribute("gen_ai.provider.name", provider_name)
        span.set_attribute("gen_ai.model.name", model)

        self._llm_spans[inference_id] = span
        return span

    def _ensure_tool_span(
        self,
        tool_id: str,
        iid: str,
        tool_name: str,
    ) -> trace.Span | None:
        if not tool_id:
            return None

        existing = self._tool_spans.get(tool_id)
        if existing:
            return existing

        parent_span = self._invocation_spans.get(iid) if iid else self._session_span
        ctx = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            name=f"execute_tool {tool_name}",
            context=ctx,
            kind=SpanKind.INTERNAL,
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.agent.id": self.uid,
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.type": "extension",
            },
        )

        self._tool_spans[tool_id] = span
        return span

    def _add_message_events(
        self,
        span: trace.Span,
        messages: list[dict[str, Any]],
        direction: str,
    ) -> None:
        """Add chat messages as span events instead of a single JSON attribute.

        Args:
            span: The span to add events to
            messages: List of message dicts with role, content, etc.
            direction: Either "input" or "output" to indicate message direction
        """
        for i, msg in enumerate(messages):
            attributes: dict[str, Any] = {
                "gen_ai.message.index": i,
            }
            if "role" in msg:
                attributes["gen_ai.message.role"] = str(msg["role"])
            if "content" in msg:
                content = msg["content"]
                text_content = self._extract_text_from_content(content)
                attributes["gen_ai.message.content"] = text_content
            if "tool_calls" in msg:
                attributes["gen_ai.message.tool_calls"] = json.dumps(
                    msg["tool_calls"], default=str, separators=(",", ":")
                )
            if "tool_call_id" in msg:
                attributes["gen_ai.message.tool_call_id"] = str(msg["tool_call_id"])
            if "name" in msg:
                attributes["gen_ai.message.name"] = str(msg["name"])

            span.add_event(f"gen_ai.{direction}.message", attributes=attributes)

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from content, handling multi-part content blocks.

        Normalizes content to a plain string format:
        - If content is already a string, returns it directly
        - If content is a list of content blocks (e.g., [{"type":"text","text":"..."}]),
          extracts and concatenates all text parts

        Args:
            content: Message content - either a string or list of content blocks

        Returns:
            Plain text string
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle multi-part content (Anthropic/OpenAI format)
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Standard format: {"type": "text", "text": "..."}
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(str(block["text"]))
                    # Alternative format: just {"text": "..."}
                    elif "text" in block and "type" not in block:
                        text_parts.append(str(block["text"]))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                return "\n".join(text_parts)
        # Fallback: JSON serialize non-standard content
        return json.dumps(content, default=str, separators=(",", ":"))
