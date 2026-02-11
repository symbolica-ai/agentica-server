import asyncio
import json
import random
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Awaitable, Callable, Generator, Literal, cast

import httpx

# Eliminates 200ms lazy import on first chat.completions request
import openai.resources.chat.completions as _  # noqa: F401
from agentica_internal.internal_errors import (
    BadRequestError,
    ConflictError,
    GenerationError,
    InferenceError,
    InsufficientCreditsError,
    InternalServerError,
    MaxTokensError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
    UnauthorizedError,
    UnprocessableEntityError,
)
from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockParam,
    Message,
    MessageCreateParams,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types import (
    ToolParam as AnthropicToolParam,
)
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.message_create_params import MessageCreateParamsBase
from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
    ToolParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agentic.models import ProviderModel
from com.context import InferenceConfig
from com.gen_model import Delta, Generation, Role
from inference.anthropic_cache import (
    apply_cache_control_chat_completions,
    apply_cache_control_messages,
)
from messages import InvocationNotifier, Notifier

# Custom tool for code execution in the Responses API.
# This allows OpenAI models to output code via a tool call rather than markdown blocks.
# The tool accepts free-form text input (the code itself), not JSON-wrapped parameters.
CODE_EXECUTION_TOOL: ToolParam = {
    'type': 'custom',
    'name': 'python',
    'description': 'Execute Python code in the REPL. REPL state is persistent across calls.',
}

# Tool for code execution in the Anthropic Messages API.
# Uses standard Anthropic tool format with input_schema.
ANTHROPIC_CODE_EXECUTION_TOOL: AnthropicToolParam = {
    'name': 'python',
    'description': 'Execute Python code in the REPL. REPL state is persistent across calls. This tool should only be used for code execution, not for reasoning or explaining/commenting your code.',
    'input_schema': {
        'type': 'object',
        'properties': {'code': {'type': 'string'}},
        'required': ['code'],
    },
}

logger = getLogger(__name__)

# JSONL logging for inference requests/responses
INFERENCE_LOG_FILE: Path | None = None  # Path("inference_log.jsonl")

# enables interleaved thinking for [4, 4.6) models https://platform.claude.com/docs/en/build-with-claude/extended-thinking#differences-in-thinking-across-model-versions
ANTHROPIC_EXTRA_HEADERS = {'anthropic-beta': 'interleaved-thinking-2025-05-14'}

if INFERENCE_LOG_FILE is not None:
    logger.warning(
        f"Inference logging enabled to {INFERENCE_LOG_FILE}. This causes large performance overhead."
    )


def _anthropic_usage_to_response_usage(usage: AnthropicUsage) -> ResponseUsage:
    """Convert an Anthropic Usage object to an OpenAI ResponseUsage.

    Maps known fields (input_tokens, output_tokens, cache_read_input_tokens) to
    their ResponseUsage equivalents, and preserves all remaining fields (e.g.
    cache_creation_input_tokens, cache_creation, inference_geo, server_tool_use,
    service_tier, plus any unknown extras) via **kwargs on the ResponseUsage
    constructor so they are not lost.
    """
    # Everything that we can't fit on ResponsesUsage, we put in extras
    extras = usage.model_dump(exclude_none=True)
    extras.pop('input_tokens', None)
    extras.pop('output_tokens', None)
    extras.pop('cache_read_input_tokens', None)

    responses_input_tokens = (
        usage.input_tokens
        + (usage.cache_read_input_tokens or 0)
        + extras.get('cache_creation_input_tokens', 0)
    )

    return ResponseUsage(
        # In responses API input tokens includes all of these
        # so mapping anthropic api to responses api
        input_tokens=responses_input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=responses_input_tokens + usage.output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=usage.cache_read_input_tokens or 0),
        # reasoning tokens are not supported in messages api so we deliberately set
        # it to nonsense values to avoid users thinking their models are not reasoning
        output_tokens_details=OutputTokensDetails(reasoning_tokens=-1),
        **extras,
    )


def _completion_usage_to_response_usage(usage: CompletionUsage) -> ResponseUsage:
    """Convert a Chat Completions CompletionUsage to an OpenAI ResponseUsage.

    Maps known fields and preserves all extra fields (e.g. cost, is_byok,
    cost_details) from model_extra on the usage object and its nested
    prompt_tokens_details / completion_tokens_details.
    """
    cached_tokens = 0
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens is not None:
        cached_tokens = usage.prompt_tokens_details.cached_tokens
    reasoning_tokens = 0
    if (
        usage.completion_tokens_details
        and usage.completion_tokens_details.reasoning_tokens is not None
    ):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
    # Carry over any extra fields from the upstream response (e.g. cost, is_byok, cost_details)
    prompt_extras = (
        usage.prompt_tokens_details.model_extra or {} if usage.prompt_tokens_details else {}
    )
    completion_extras = (
        usage.completion_tokens_details.model_extra or {} if usage.completion_tokens_details else {}
    )
    return ResponseUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens, **prompt_extras),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=reasoning_tokens, **completion_extras
        ),
        **usage.model_extra or {},
    )


def _find_matching_end(text: str, start: str, end: str, pos: int) -> int:
    """Find matching end tag position, handling nesting. Raises ValueError if not found."""
    depth = 1
    start_len, end_len = len(start), len(end)
    while depth > 0:
        next_start = text.find(start, pos)
        next_end = text.index(end, pos)
        # Prioritize start when: start comes first, OR they overlap and start is longer
        # (handles case where end is prefix of start, like ``` vs ```python)
        if next_start != -1 and (
            next_start < next_end or (next_start == next_end and start_len > end_len)
        ):
            depth += 1
            pos = next_start + start_len
        else:
            depth -= 1
            pos = next_end + end_len
    return pos - end_len


def _text_between(text: str, start: str, end: str) -> Generator[str, None, None]:
    start_len = len(start)
    end_len = len(end)
    ptr = 0
    while True:
        try:
            start_pos = text.index(start, ptr)
            end_pos = _find_matching_end(text, start, end, start_pos + start_len)
            yield text[start_pos + start_len : end_pos]
            ptr = end_pos + end_len
        except ValueError:
            break


def _text_not_between(text: str, start: str, end: str) -> Generator[str, None, None]:
    start_len = len(start)
    end_len = len(end)
    ptr = 0
    while True:
        try:
            start_pos = text.index(start, ptr)
        except ValueError:
            yield text[ptr:]  # No more starts, yield rest
            break
        # Check if matching end exists before yielding
        try:
            end_pos = _find_matching_end(text, start, end, start_pos + start_len)
        except ValueError:
            yield text[ptr:]  # Unmatched start, yield everything remaining
            break
        yield text[ptr:start_pos]
        ptr = end_pos + end_len


def _extract_reasoning_summary(response: Response) -> str | None:
    """Extract reasoning summary text from Responses API output items.

    Reasoning summaries appear as output items of type 'reasoning' containing
    'summary' arrays with 'summary_text' items.
    """
    reasoning_parts: list[str] = []
    for item in response.output:
        item_dict = item.model_dump(exclude_none=True)
        if item_dict.get('type') == 'reasoning':
            summaries = item_dict.get('summary', [])
            for s in summaries:
                if s.get('type') == 'summary_text':
                    text = s.get('text', '')
                    if text:
                        reasoning_parts.append(text)
    return ''.join(reasoning_parts) if reasoning_parts else None


def _extract_reasoning_string_from_details(reasoning_details: list[dict[str, Any]]) -> str | None:
    """Extract readable text from reasoning_details blocks.

    Used to populate Generation.reasoning when reasoning_content isn't available
    but structured reasoning_details are present.
    """
    parts: list[str] = []
    for block in reasoning_details:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type', '')
        if block_type == 'reasoning.text':
            text = block.get('text', '')
            if text:
                parts.append(text)
        elif block_type == 'reasoning.summary':
            summary = block.get('summary', '')
            if summary:
                parts.append(summary)
        # reasoning.encrypted blocks have no readable text — skip
    return ''.join(parts) if parts else None


def _extract_code_from_markdown(content: str) -> tuple[str | None, int]:
    """Extract first ```python code block from content.

    Returns:
        Tuple of (first_code_block, extra_block_count)
    """
    code_blocks = list(_text_between(content, '```python', '```'))
    if not code_blocks:
        return None, 0
    return code_blocks[0], len(code_blocks) - 1


def _log_inference(entry_type: str, inference_id: str, data: dict[str, Any]) -> None:
    """Append an inference log entry to the JSONL file."""
    if INFERENCE_LOG_FILE is None:
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": entry_type,
        "inference_id": inference_id,
        "data": data,
    }
    with open(INFERENCE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class InferenceSystem(ABC):
    notifier: Notifier
    fresh_id: Callable[[], str]
    model: ProviderModel

    @property
    @abstractmethod
    def has_messages(self) -> bool:
        """Return True if the conversation has any messages."""
        ...

    @property
    @abstractmethod
    def uses_tool_calls(self) -> bool:
        """Return True if this system has tools configured."""
        ...

    @property
    @abstractmethod
    def _base_url(self) -> httpx.URL:
        """Return the base URL of the underlying HTTP client."""
        ...

    @abstractmethod
    def insert(self, role: Role, content: str): ...

    @abstractmethod
    def _body(
        self, inference_config: InferenceConfig
    ) -> tuple[
        ResponseCreateParamsNonStreaming | CompletionCreateParamsNonStreaming | Any,
        dict[str, Any] | None,
    ]:
        """Build the request body for inference.

        Returns:
            Tuple of (typed_body, extra_body) where extra_body contains fields
            not in the standard API spec (e.g., reasoning configuration).
        """
        ...

    @abstractmethod
    async def _create(
        self, inference_config: InferenceConfig, timeout: int | None = None
    ) -> tuple[dict[str, Any], Generation]: ...

    def __get_server_info(self) -> tuple[str | None, int | None]:
        """Parse endpoint URL to extract server.address and server.port per OTel spec."""
        try:
            parsed = self._base_url
            host = parsed.host
            port = parsed.port
            # If no explicit port, use default based on scheme
            if port is None and parsed.scheme:
                port = 443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else None
            return (host, port)
        except Exception as e:
            logger.warning(f"Failed to parse inference endpoint URL: {e}")
            return (None, None)

    async def invoke(
        self,
        inference_config: InferenceConfig,
        timeout: int | None = None,
        iid: str = "no-iid",
        invocation: InvocationNotifier | None = None,
    ) -> Generation:
        """Inference via HTTP POST."""
        body = self._body(inference_config)
        body_dict = cast(dict[str, Any], body)
        this_id = self.fresh_id()
        try:
            server_address, server_port = self.__get_server_info()
            if invocation:
                invocation.start_inference(
                    inference_id=this_id,
                    request=body_dict,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                # Start the gen_ai.chat span BEFORE the LLM call
                # This captures the actual inference duration
                start_event = invocation.create_chat_event(
                    inference_id=this_id,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                await invocation.log_genai_chat(start_event)
            await self.notifier.on_inference_request(
                inference_id=this_id,
                iid=iid,
                request_str=json.dumps(body_dict),
                timeout=timeout,
            )
            _log_inference("request", this_id, body_dict)
            result, generation = await self._create(
                inference_config=inference_config,
                timeout=timeout,
            )
            _log_inference("response", this_id, result)
            await self.notifier.on_inference_response(
                inference_id=this_id,
                iid=iid,
                response_str=json.dumps(result),
            )
            if invocation:
                server_address, server_port = self.__get_server_info()
                event = invocation.create_chat_event(
                    inference_id=this_id,
                    response=result,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                await invocation.log_genai_chat(event)
            return generation
        except BaseException as e:
            await self.notifier.on_inference_error(
                inference_id=this_id,
                iid=iid,
                err=e,
                message=traceback.format_exc(),
            )
            raise e

    @abstractmethod
    async def invoke_stream(
        self, ctx: InferenceConfig
    ) -> tuple[Awaitable[Generation], AsyncGenerator[Delta, None]]:
        pass

    @abstractmethod
    def insert_function_call(self, name: str, code: str, text: str = "") -> None:
        """Insert a synthetic function call for few-shot examples.

        For Responses API: inserts custom_tool_call item with optional preceding text.
        For Chat Completions API: inserts as assistant message with markdown code block.
        """
        ...

    @abstractmethod
    def insert_execution_result(self, output: str) -> None:
        """Insert a synthetic function call output for few-shot examples.

        For Responses API: inserts function_call_output item.
        For Chat Completions API: inserts as user message.
        """
        ...


# NOTE: should have been called ResponsesConversation
class ResponsesSystem(InferenceSystem):
    def __init__(
        self,
        client: AsyncOpenAI,
        fresh_id: Callable[[], str],
        notifier: Notifier,
        model: ProviderModel,
    ):
        self.client = client
        self.fresh_id = fresh_id
        self.notifier = notifier
        self.model = model
        self.input: list[ResponseInputItemParam] = []
        self.responses: list[Response] = []

    @property
    def _base_url(self) -> httpx.URL:
        return self.client.base_url

    @property
    def _is_direct(self) -> bool:
        """Return True if making requests straight to OpenAI."""
        return "api.openai.com" in self._base_url.host

    @property
    def iid(self) -> str:
        return self.fresh_id()

    @property
    def has_messages(self) -> bool:
        return len(self.input) > 0

    @property
    def uses_tool_calls(self) -> bool:
        return 'openai/gpt-5' in self.model.identifier

    @staticmethod
    def _normalize_role(role: Role) -> Literal['user', 'assistant', 'system', 'developer']:
        """Normalize role string to valid Responses API role literal."""
        match role:
            case 'assistant':
                return 'assistant'
            case 'system':
                return 'system'
            case ('user', _):
                return 'user'

        raise ValueError(f"Invalid role: {role}")

    def insert(self, role: Role, content: str):
        msg: EasyInputMessageParam = {
            'type': 'message',
            'role': self._normalize_role(role),
            'content': content,
        }
        self.input.append(msg)

    def _extract_code_from_response(self, response: Response) -> tuple[str | None, int, bool]:
        """Extract code from tool calls (function or custom) in the response.

        Falls back to markdown extraction if no tool call is found (for non-OpenAI providers).

        Returns:
            Tuple of (code, extra_code_blocks, from_tool_call).
            from_tool_call is True if code came from a tool call, False if from markdown.
        """
        for output_item in response.output:
            # Function tool call (JSON arguments with 'code' field)
            if isinstance(output_item, ResponseFunctionToolCall):
                if output_item.name == 'python':
                    try:
                        args = json.loads(output_item.arguments)
                        if code := args.get('code'):
                            return code, 0, True  # Tool calls are inherently single
                    except json.JSONDecodeError:
                        pass
            # Custom tool call (raw text input)
            elif isinstance(output_item, ResponseCustomToolCall):
                return output_item.input, 0, True  # Tool calls are inherently single

        # Fallback: extract from markdown (for non-OpenAI providers that don't support tools)
        code, extra = _extract_code_from_markdown(response.output_text)
        return code, extra, False

    def insert_execution_result(self, output: str) -> None:
        """Insert code execution result into the conversation.

        Tries to submit as a tool result (function_call_output) if there was a tool call.
        Falls back to inserting as a user message if no tool call is found.
        """
        # Find the most recent function_call in self.input
        call_id: str | None = None
        for item in reversed(self.input):
            # Items are TypedDicts, safe to check 'type' key
            if item.get('type') == 'custom_tool_call':
                call_id = item.get('call_id')
                break

        if call_id:
            # Submit as tool result using typed FunctionCallOutput
            result: FunctionCallOutput = {
                'type': 'function_call_output',
                'call_id': call_id,
                'output': output,
            }
            self.input.append(result)
        else:
            # No tool call found - insert as user message
            self.insert(role=('user', None), content=output)

    def insert_function_call(self, name: str, code: str, text: str = "") -> None:
        """Insert a synthetic function call for few-shot examples.

        Inserts an assistant message with optional text, followed by a custom_tool_call item.
        """
        call_id = f"fewshot_{random.randint(1000, 9999)}"
        # If there's reasoning text, insert it as an assistant message first
        if text.strip():
            msg: EasyInputMessageParam = {
                'type': 'message',
                'role': 'assistant',
                'content': text.strip(),
            }
            self.input.append(msg)

        # Insert the tool call
        tool_call: ResponseInputItemParam = {
            'type': 'custom_tool_call',
            'call_id': call_id,
            'name': name,
            'input': code,
        }
        self.input.append(tool_call)

    def _body(
        self, inference_config: InferenceConfig
    ) -> tuple[ResponseCreateParamsNonStreaming, dict[str, Any] | None]:
        # Strip provider prefix for native OpenAI API (e.g., "openai/gpt-4.1" -> "gpt-4.1")
        model = inference_config.model
        if self._is_direct and '/' in model:
            model = model.split('/', 1)[1]

        if self.model.provider == 'anthropic':
            from inference.anthropic_cache import apply_cache_control_responses

            input_items = apply_cache_control_responses(self.input, inference_config.cache_ttl)
        else:
            input_items = self.input

        body: ResponseCreateParamsNonStreaming = {
            'model': model,
            'input': input_items,
        }

        # Add max tokens if specified
        max_tokens = inference_config.max_tokens()
        if max_tokens is not None:
            body['max_output_tokens'] = max_tokens

        # Add reasoning configuration if specified
        # Responses API natively supports reasoning, so it goes in the typed body
        if inference_config.reasoning_effort is not None:
            body['reasoning'] = {'effort': inference_config.reasoning_effort, 'summary': 'auto'}
            body['include'] = ['reasoning.encrypted_content']

        # Add tools if configured (only for gpt-5 models on native OpenAI API)
        # OpenRouter's Responses API doesn't support custom tools (type: 'custom')
        if self.uses_tool_calls:
            # NOTE: Of all the openai models starting with {gpt-5, gpt-4, o}, only and all of the models starting with gpt-5 support custom tools
            body['tools'] = [CODE_EXECUTION_TOOL]
            # Force the model to use tools rather than responding with markdown
            # body['tool_choice'] = 'required' # NOTE: it is valid to not call a tool when return_type=str
            body['parallel_tool_calls'] = False

        # Responses API doesn't need extra_body - all fields are in the typed spec
        return body, None

    async def _create(
        self, inference_config: InferenceConfig, timeout: int | None = None
    ) -> tuple[dict[str, Any], Generation]:
        body, extra_body = self._body(inference_config)
        try:
            response = await self.client.responses.create(**body, extra_body=extra_body)
        except APIError as e:
            raise _api_error_to_generation_error(e)

        if response.status == 'incomplete':
            if (
                response.incomplete_details
                and response.incomplete_details.reason == 'max_output_tokens'
            ):
                raise MaxTokensError(
                    inference_config.max_tokens()
                    or "The provider's default limit for number of tokens was reached."
                )
            else:
                raise GenerationError(f"Incomplete response: {response.incomplete_details}")
        elif response.status != 'completed':
            raise GenerationError(str(response.error))

        # Extract code from tool calls (function or custom)
        code, extra_code_blocks, code_from_tool = self._extract_code_from_response(response)

        # Usage should always exist (unless some kind of error)
        # https://platform.openai.com/docs/api-reference/responses/object#responses-object-usage
        usage = response.usage
        assert usage is not None, f"Inference request was missing usage: {response!r}"

        # Append raw output items back to self.input for multi-turn context preservation.
        # This is critical for reasoning models where the output items include reasoning
        # with encrypted_content that must be passed back exactly in subsequent requests.
        # We use model_dump() to convert SDK objects to dicts that are JSON-serializable
        # and compatible with ResponseInputItemParam typing.
        for output_item in response.output:
            # Anthropic's API rejects empty assistant messages,
            # despite that Anthropic models seem to **ocassionally** emit them.
            # This code block replaces the empty blocks with [no output generated]
            # NOTE: the empty messages often come when 65536 tokens are output,
            # which indicates a major error in the underlying provider (openrouter)
            # investigation required to see if this is our fault and we're causing overcharging.
            if "anthropic/" in self.model.identifier and isinstance(
                output_item, ResponseOutputMessage
            ):
                for content in output_item.content:
                    if isinstance(content, ResponseOutputText):
                        if not content.text or content.text.isspace():
                            content.text = "[no output generated]"

            self.input.append(
                cast(ResponseInputItemParam, output_item.model_dump(exclude_none=True))
            )

        # Extract reasoning summary from output items (for reasoning models)
        reasoning = _extract_reasoning_summary(response)

        return response.model_dump(exclude_none=True), Generation(
            output_text=response.output_text,
            code=code,
            usage=usage,
            reasoning=reasoning,
            extra_code_blocks=extra_code_blocks,
            code_from_tool=code_from_tool,
        )

    def _body_streaming(
        self, inference_config: InferenceConfig
    ) -> tuple[ResponseCreateParamsNonStreaming, dict[str, Any] | None]:
        # Streaming uses the same body as non-streaming for Responses API
        return self._body(inference_config)

    async def invoke_stream(
        self, ctx: InferenceConfig
    ) -> tuple[Awaitable[Generation], AsyncGenerator[Delta, None]]:
        future: asyncio.Future[Generation] = asyncio.Future()
        queue: asyncio.Queue[Delta | None] = asyncio.Queue()

        async def _stream_task() -> None:
            body, extra_body = self._body_streaming(ctx)
            try:
                # Cast body to dict for stream() - SDK typing limitation
                # (ResponseCreateParamsNonStreaming works at runtime for streaming)
                body_dict = cast(dict[str, Any], body)
                if extra_body is not None:
                    stream_ctx = self.client.responses.stream(**body_dict, extra_body=extra_body)
                else:
                    stream_ctx = self.client.responses.stream(**body_dict)
                async with stream_ctx as stream:
                    async for event in stream:
                        if event.type == 'response.output_text.delta':
                            await queue.put(Delta(content=event.delta, type='output_text'))
                        elif event.type == 'response.reasoning_summary_text.delta':
                            await queue.put(Delta(content=event.delta, type='reasoning'))
                        elif event.type == 'response.completed':
                            response = event.response
                            if response.status == 'incomplete':
                                max_tokens = (
                                    ctx.max_tokens()
                                    or "The provider's default limit for number of tokens was reached."
                                )
                                raise MaxTokensError(max_tokens)
                            elif response.status != 'completed':
                                assert response.error is not None
                                raise GenerationError(str(response.error))

                            code, extra_code_blocks, code_from_tool = (
                                self._extract_code_from_response(response)
                            )
                            usage = response.usage
                            assert usage is not None, (
                                f"Inference request was missing usage: {response!r}"
                            )

                            for output_item in response.output:
                                # Anthropic's API rejects empty assistant messages,
                                # despite that Anthropic models seem to **ocassionally** emit them.
                                # This code block replaces the empty blocks with [no output generated]
                                # NOTE: the empty messages often come when 65536 tokens are output,
                                # which indicates a major error in the underlying provider (openrouter)
                                # investigation required to see if this is our fault and we're causing overcharging.
                                if "anthropic/" in self.model.identifier and isinstance(
                                    output_item, ResponseOutputMessage
                                ):
                                    for content in output_item.content:
                                        if isinstance(content, ResponseOutputText):
                                            if not content.text or content.text.isspace():
                                                content.text = "[no output generated]"

                                self.input.append(
                                    cast(
                                        ResponseInputItemParam,
                                        output_item.model_dump(exclude_none=True),
                                    )
                                )

                            # Extract reasoning summary from output items
                            reasoning = _extract_reasoning_summary(response)

                            # Stream the code block if it came from a tool call.
                            # Tool call code isn't included in output_text.delta
                            # events, so push it explicitly.
                            if code and code_from_tool:
                                await queue.put(
                                    Delta(
                                        content=f"\n<python>\n{code}\n</python>",
                                        type='output_text',
                                    )
                                )

                            future.set_result(
                                Generation(
                                    output_text=response.output_text,
                                    code=code,
                                    usage=usage,
                                    reasoning=reasoning,
                                    extra_code_blocks=extra_code_blocks,
                                    code_from_tool=code_from_tool,
                                )
                            )
            except BaseException as e:
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                await queue.put(None)

        asyncio.create_task(_stream_task())

        async def _delta_generator() -> AsyncGenerator[Delta, None]:
            while True:
                delta = await queue.get()
                if delta is None:
                    break
                yield delta

        return future, _delta_generator()


def _api_error_to_generation_error(err: APIError) -> GenerationError:
    """Map OpenAI API errors to Agentica error types based on HTTP status code."""
    # For APIStatusError subclasses, we have access to the request/response and status code
    if isinstance(err, APIStatusError):
        status_code = err.status_code
        request = err.request
        response = err.response

        # Map HTTP status codes to specific error types
        error_mapping: dict[int, type[InferenceError]] = {
            400: BadRequestError,
            401: UnauthorizedError,
            402: InsufficientCreditsError,
            403: PermissionDeniedError,
            404: NotFoundError,
            409: ConflictError,
            413: RequestTooLargeError,
            422: UnprocessableEntityError,
            429: RateLimitError,
            500: InternalServerError,
            502: ServiceUnavailableError,
            503: ServiceUnavailableError,
            529: OverloadedError,
        }

        error_class = error_mapping.get(status_code, InternalServerError)
        return error_class(request=request, response=response)

    # For non-status errors (connection errors, timeouts, etc.), wrap in generic GenerationError
    return GenerationError(str(err))


def _anthropic_error_to_generation_error(err: AnthropicAPIError) -> GenerationError:
    """Map Anthropic API errors to Agentica error types based on HTTP status code."""
    if isinstance(err, AnthropicAPIStatusError):
        status_code = err.status_code
        request = err.request
        response = err.response

        error_mapping: dict[int, type[InferenceError]] = {
            400: BadRequestError,
            401: UnauthorizedError,
            403: PermissionDeniedError,
            404: NotFoundError,
            413: RequestTooLargeError,
            422: UnprocessableEntityError,
            429: RateLimitError,
            500: InternalServerError,
            502: ServiceUnavailableError,
            503: ServiceUnavailableError,
            529: OverloadedError,
        }

        error_class = error_mapping.get(status_code, InternalServerError)
        return error_class(request=request, response=response)

    return GenerationError(str(err))


# TODO: add inference rounds


class ChatCompletionsSystem(InferenceSystem):
    def __init__(
        self,
        client: AsyncOpenAI,
        fresh_id: Callable[[], str],
        notifier: Notifier,
        model: ProviderModel,
    ):
        self.client = client
        self.fresh_id = fresh_id
        self.notifier = notifier
        self.model = model
        self.messages: list[ChatCompletionMessageParam] = []
        self.completions: list[ChatCompletion] = []

    @property
    def _base_url(self) -> httpx.URL:
        return self.client.base_url

    @property
    def _is_direct(self) -> bool:
        """Return True if making requests straight to OpenAI."""
        return "api.openai.com" in self._base_url.host

    @property
    def has_messages(self) -> bool:
        return len(self.messages) > 0

    @property
    def uses_tool_calls(self) -> bool:
        return False

    def insert(self, role: Role, content: str):
        match role:
            case 'assistant':
                self.messages.append({'role': 'assistant', 'content': content})

            case 'system':
                self.messages.append({'role': 'system', 'content': content})

            case ('user', username):
                msg: ChatCompletionMessageParam = {'role': 'user', 'content': content}
                if username is not None:
                    msg['name'] = username
                self.messages.append(msg)

            case _:
                raise ValueError(f"Invalid role: {role}")

    def _body(
        self, inference_config: InferenceConfig
    ) -> tuple[CompletionCreateParamsNonStreaming, dict[str, Any] | None]:
        # Strip provider prefix (e.g., "openai/gpt-4.1" -> "gpt-4.1")
        model = inference_config.model
        if self._is_direct and '/' in model:
            model = model.split('/', 1)[1]

        if self.model.provider == 'anthropic':
            messages = apply_cache_control_chat_completions(
                self.messages, inference_config.cache_ttl
            )
        else:
            messages = self.messages

        body: CompletionCreateParamsNonStreaming = {
            'model': model,
            'messages': messages,
        }

        # Add max tokens if specified
        max_tokens = inference_config.max_tokens()
        if max_tokens is not None:
            body['max_completion_tokens'] = max_tokens

        # Add reasoning configuration if specified
        # Use flat format: reasoning_effort='high' (works for both OpenAI and OpenRouter)
        if inference_config.reasoning_effort is not None:
            body['reasoning_effort'] = inference_config.reasoning_effort

        return body, None

    def insert_execution_result(self, output: str) -> None:
        """Insert code execution result into the conversation as a user message."""
        self.insert(role=('user', 'execution'), content=output)

    def insert_function_call(
        self,
        name: str,
        code: str,
        text: str = "",  # noqa: ARG002
    ) -> None:
        """Insert a synthetic function call as a markdown code block.

        For Chat Completions API, tool calls are represented as markdown code blocks
        in assistant messages, NOT via function calling. This method should never be
        called for Chat Completions - if you hit this error, there's a bug in the
        code path that should be guarded by `if session.uses_tool_calls:`.
        """
        raise NotImplementedError(
            "Chat Completions API should NEVER use function calling - "
            "code is emitted via markdown blocks. This indicates a bug in the caller."
        )

    async def _create(
        self, inference_config: InferenceConfig, timeout: int | None = None
    ) -> tuple[dict[str, Any], Generation]:
        body, _ = self._body(inference_config)
        try:
            response: ChatCompletion
            response = await self.client.chat.completions.create(**body)
        except APIError as e:
            raise _api_error_to_generation_error(e)

        # Get first choice (we only request n=1)
        choice = response.choices[0]
        message = choice.message

        # Check finish reason for max tokens
        if choice.finish_reason == 'length':
            max_tokens = (
                inference_config.max_tokens()
                or "The provider's default limit for number of tokens was reached."
            )
            raise MaxTokensError(max_tokens)

        # Extract code from content (parse ```python blocks)
        code, extra_code_blocks = _extract_code_from_markdown(message.content or "")

        # Build usage (convert to ResponseUsage for compatibility)
        assert response.usage is not None, "Usage not received from upstream"
        usage = _completion_usage_to_response_usage(response.usage)

        # Append assistant message to conversation history for multi-turn context.
        # Prefer reasoning_details (structured array with encrypted/summarized blocks)
        # over reasoning (plain string) for reasoning model continuity across turns.
        reasoning_content = getattr(message, 'reasoning_content', None)
        reasoning_details = getattr(message, 'reasoning_details', None)
        assistant_msg: dict[str, Any] = {'role': 'assistant', 'content': message.content}
        if reasoning_details:
            assistant_msg['reasoning_details'] = reasoning_details
        elif reasoning_content:
            assistant_msg['reasoning'] = reasoning_content
        self.messages.append(cast(ChatCompletionAssistantMessageParam, assistant_msg))

        # Store the completion for reference
        self.completions.append(response)

        # For Generation.reasoning (used for display/logging), extract readable text
        # from whichever reasoning source is available
        reasoning_string = reasoning_content
        if not reasoning_string and reasoning_details:
            reasoning_string = _extract_reasoning_string_from_details(reasoning_details)

        return response.model_dump(exclude_none=True), Generation(
            output_text=message.content or "",
            code=code,
            usage=usage,
            reasoning=reasoning_string,
            extra_code_blocks=extra_code_blocks,
        )

    def _body_streaming(
        self, inference_config: InferenceConfig
    ) -> tuple[CompletionCreateParamsStreaming, dict[str, Any] | None]:
        """Build streaming request body."""
        model = inference_config.model
        if self._is_direct and '/' in model:
            model = model.split('/', 1)[1]

        if self.model.provider == 'anthropic':
            messages = apply_cache_control_chat_completions(
                self.messages, inference_config.cache_ttl
            )
        else:
            messages = self.messages

        body: CompletionCreateParamsStreaming = {
            'model': model,
            'messages': messages,
            'stream': True,
            'stream_options': {'include_usage': True},
        }

        max_tokens = inference_config.max_tokens()
        if max_tokens is not None:
            body['max_completion_tokens'] = max_tokens

        # Add reasoning configuration if specified
        # Use flat format: reasoning_effort='high' (works for both OpenAI and OpenRouter)
        if inference_config.reasoning_effort is not None:
            body['reasoning_effort'] = inference_config.reasoning_effort

        return body, None

    async def invoke_stream(
        self, ctx: InferenceConfig
    ) -> tuple[Awaitable[Generation], AsyncGenerator[Delta, None]]:
        future: asyncio.Future[Generation] = asyncio.Future()
        queue: asyncio.Queue[Delta | None] = asyncio.Queue()

        async def _stream_task() -> None:
            body, _ = self._body_streaming(ctx)

            collected_content = ""
            collected_reasoning = ""
            usage: ResponseUsage | None = None

            try:
                stream = await self.client.chat.completions.create(**body)
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice and choice.delta:
                        delta = choice.delta

                        # Handle content delta
                        if delta.content:
                            collected_content += delta.content
                            await queue.put(Delta(content=delta.content, type='output_text'))

                        # Handle reasoning delta
                        # NOTE:  not supported for all providers
                        reasoning_content = getattr(delta, 'reasoning_content', None)
                        if reasoning_content:
                            collected_reasoning += reasoning_content
                            await queue.put(Delta(content=reasoning_content, type='reasoning'))

                        # Check for completion
                        if choice.finish_reason:
                            if choice.finish_reason == 'length':
                                max_tokens = (
                                    ctx.max_tokens() or "The provider's default limit reached."
                                )
                                raise MaxTokensError(max_tokens)

                    # Capture usage from final chunk
                    if chunk.usage:
                        usage = _completion_usage_to_response_usage(chunk.usage)

                # Stream complete - build final Generation
                code, extra_code_blocks = _extract_code_from_markdown(collected_content)

                # Append to conversation history for multi-turn context.
                # Streaming deltas only provide reasoning_content as incremental text chunks;
                # structured reasoning_details are not available during streaming.
                # Store as 'reasoning' (plain string) — the non-streaming path handles
                # reasoning_details preservation.
                assistant_msg: dict[str, Any] = {'role': 'assistant', 'content': collected_content}
                if collected_reasoning:
                    assistant_msg['reasoning'] = collected_reasoning
                self.messages.append(cast(ChatCompletionAssistantMessageParam, assistant_msg))

                assert usage is not None, "Stream ended without usage info"
                future.set_result(
                    Generation(
                        output_text=collected_content,
                        code=code,
                        usage=usage,
                        reasoning=collected_reasoning or None,
                        extra_code_blocks=extra_code_blocks,
                    )
                )
            except BaseException as e:
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                await queue.put(None)

        asyncio.create_task(_stream_task())

        async def _delta_generator() -> AsyncGenerator[Delta, None]:
            while True:
                delta = await queue.get()
                if delta is None:
                    break
                yield delta

        return future, _delta_generator()


class MessagesSystem(InferenceSystem):
    def __init__(
        self,
        client: AsyncAnthropic,
        fresh_id: Callable[[], str],
        notifier: Notifier,
        model: ProviderModel,
    ):
        self.client = client
        self.fresh_id = fresh_id
        self.notifier = notifier
        self.model = model
        self.messages: list[MessageParam] = []
        self.system_parts: list[str] = []

    @property
    def _base_url(self) -> httpx.URL:
        return self.client.base_url

    @property
    def _is_direct(self) -> bool:
        """Return True if making requests straight to Anthropic."""
        return "api.anthropic.com" in str(self._base_url)

    @property
    def has_messages(self) -> bool:
        return len(self.messages) > 0

    @property
    def uses_tool_calls(self) -> bool:
        return True

    def insert(self, role: Role, content: str):
        if role == 'system':
            self.system_parts.append(content)
            return

        msg: MessageParam
        match role:
            case 'assistant':
                msg = {'role': 'assistant', 'content': content}
            case ('user', _):
                msg = {'role': 'user', 'content': content}
            case _:
                raise ValueError(f"Invalid role: {role}")
        self.messages.append(msg)

    def _merge_consecutive_user_messages(self, messages: list[MessageParam]) -> list[MessageParam]:
        """Merge consecutive user messages by combining their content blocks.

        This is a safety net for the few-shot → real-task boundary where
        two user messages may be consecutive. In Anthropic's API, a single
        user message can contain multiple content blocks.
        """
        if not messages:
            return messages

        merged: list[MessageParam] = []
        for msg in messages:
            if merged and merged[-1]['role'] == 'user' and msg['role'] == 'user':
                # Merge into the previous user message
                prev_content = merged[-1]['content']
                curr_content = msg['content']

                # Normalize both to list format
                def _to_blocks(
                    content: str | Any,
                ) -> list[ContentBlockParam]:
                    if isinstance(content, str):
                        block: TextBlockParam = {'type': 'text', 'text': content}
                        return [block]
                    else:
                        return list(content)

                prev_blocks = _to_blocks(prev_content)
                curr_blocks = _to_blocks(curr_content)
                merged[-1] = {'role': 'user', 'content': prev_blocks + curr_blocks}
            else:
                merged.append(msg)
        return merged

    def _body(self, inference_config: InferenceConfig) -> tuple[MessageCreateParamsBase, None]:
        # Strip provider prefix for direct Anthropic API
        model = inference_config.model
        if self._is_direct and '/' in model:
            model = model.split('/', 1)[1]

        # Merge consecutive user messages first, then apply prompt caching.
        # Order matters: cache breakpoint indices are computed from len(messages),
        # so merging must happen before caching to avoid breakpoints landing on
        # stale indices that get collapsed into a single message.
        merged_messages = self._merge_consecutive_user_messages(self.messages)

        cached_messages = apply_cache_control_messages(merged_messages, inference_config.cache_ttl)

        # set max tokens according to "max output" row on https://platform.claude.com/docs/en/about-claude/models/overview
        if 'opus-4-6' in self.model.endpoint_identifier:
            max_tokens = 128000
        else:
            max_tokens = 64000

        body: MessageCreateParams = {
            'model': model,
            'messages': cached_messages,
            'max_tokens': max_tokens,
            'tools': [ANTHROPIC_CODE_EXECUTION_TOOL],
        }

        # Add system message if present
        if self.system_parts:
            body['system'] = '\n\n'.join(self.system_parts)

        # Add thinking/reasoning configuration based on model
        if 'opus-4-6' in self.model.endpoint_identifier:
            # Opus 4.6 REQUIRES adaptive thinking
            body['thinking'] = {'type': 'adaptive'}
            if inference_config.reasoning_effort is not None:
                effort = inference_config.reasoning_effort
                if effort == "xhigh":
                    effort = "max"
                if effort in ("minimal", "none"):
                    effort = "low"
                body['output_config'] = {'effort': effort}
        elif inference_config.reasoning_effort is not None:
            # All other Claude models use budget_tokens
            effort_to_pct = {
                'xhigh': 0.95,
                'high': 0.80,
                'medium': 0.50,
                'low': 0.20,
                'minimal': 0.10,
                'none': 0.00,
            }
            pct = effort_to_pct.get(inference_config.reasoning_effort, 0.50)
            budget = max(1024, int(max_tokens * pct))
            # budget_tokens must be strictly less than max_tokens
            if budget >= body['max_tokens']:
                budget = body['max_tokens'] - 1
            body['thinking'] = {'type': 'enabled', 'budget_tokens': budget}

        return body, None

    def _extract_code_from_response(self, message: Message) -> tuple[str | None, int, bool]:
        """Extract code from tool_use blocks in the response.

        Returns:
            Tuple of (code, extra_code_blocks, from_tool_call).
        """
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                if block.name != 'python':
                    raise GenerationError(
                        f"Anthropic Messages API: Agent called a tool other than 'python': {message!r}"
                    )

                code = block.input.get('code')

                if not isinstance(code, str):
                    raise GenerationError(
                        f"Anthropic Messages API emitted invalid tool response: {message!r}"
                    )
                if code:
                    return code, 0, True

        return None, 0, False

    def _process_response(
        self, message: Message, inference_config: InferenceConfig
    ) -> tuple[dict[str, Any], Generation]:
        """Process a final Anthropic Message into conversation history + Generation.

        Shared by both _create and invoke_stream to avoid duplication.
        """
        # Check stop reason for max tokens
        if message.stop_reason == 'max_tokens':
            raise MaxTokensError(
                inference_config.max_tokens()
                or "The provider's default limit for number of tokens was reached."
            )

        # Extract code from tool_use blocks
        code, extra_code_blocks, code_from_tool = self._extract_code_from_response(message)

        # Build usage (convert Anthropic Usage to ResponseUsage for compatibility)
        usage = _anthropic_usage_to_response_usage(message.usage)

        # Preserve full assistant content blocks in conversation history.
        # This is critical for thinking blocks that must be passed back exactly.
        # Convert SDK model objects to dicts for JSON serialization compatibility.
        content_dicts = [block.model_dump(exclude_none=True) for block in message.content]
        self.messages.append(MessageParam(role='assistant', content=content_dicts))

        # Extract output text from text blocks
        output_text = ''
        for block in message.content:
            if isinstance(block, TextBlock):
                output_text += block.text

        # Extract readable thinking for Generation.reasoning
        reasoning_parts: list[str] = []
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking:
                    reasoning_parts.append(block.thinking)
        reasoning = ''.join(reasoning_parts) if reasoning_parts else None

        return message.model_dump(exclude_none=True), Generation(
            output_text=output_text,
            code=code,
            usage=usage,
            reasoning=reasoning,
            extra_code_blocks=extra_code_blocks,
            code_from_tool=code_from_tool,
        )

    async def _create(
        self, inference_config: InferenceConfig, timeout: int | None = None
    ) -> tuple[dict[str, Any], Generation]:
        body, _ = self._body(inference_config)

        try:
            response = await self.client.messages.create(
                **body, extra_headers=ANTHROPIC_EXTRA_HEADERS
            )
        except ValueError as e:
            if 'streaming is required' in e.args[0].lower():
                async with self.client.messages.stream(
                    **body, extra_headers=ANTHROPIC_EXTRA_HEADERS
                ) as stream:
                    response = await stream.get_final_message()
            else:
                raise e
        except AnthropicAPIError as e:
            raise _anthropic_error_to_generation_error(e)

        return self._process_response(response, inference_config)

    def insert_function_call(self, name: str, code: str, text: str = "") -> None:
        """Insert a synthetic function call for few-shot examples.

        Creates an assistant message with optional text block + tool_use block.
        """
        call_id = f"toolu_{random.randint(100000, 999999)}"
        content: list[TextBlockParam | ToolUseBlockParam] = []

        if text.strip():
            content.append(TextBlockParam(type='text', text=text.strip()))

        content.append(
            ToolUseBlockParam(
                type='tool_use',
                id=call_id,
                name=name,
                input={'code': code},
            )
        )

        self.messages.append(MessageParam(role='assistant', content=content))

    def insert_execution_result(self, output: str) -> None:
        """Insert code execution result as a tool_result content block.

        Searches backward for the most recent tool_use block to get its ID.
        If a tool_result already exists for that tool_use_id (e.g. from
        few-shot examples where an assistant turn was skipped in tool-call
        mode, or from multiple insert_execution_result calls for the same
        turn), appends to the existing tool_result instead of creating a
        duplicate — Anthropic requires exactly one tool_result per tool_use.
        """
        # Find the most recent tool_use block
        tool_use_id: str | None = None
        for msg in reversed(self.messages):
            if msg['role'] == 'assistant':
                content = msg['content']
                if isinstance(content, list):
                    for block in reversed(content):
                        # Handle both ToolUseBlockParam (TypedDict) and ToolUseBlock (SDK model)
                        if isinstance(block, ToolUseBlock):
                            tool_use_id = block.id
                            break
                        if isinstance(block, dict) and block['type'] == 'tool_use':
                            tool_use_id = block['id']
                            break
                if tool_use_id:
                    break

        if tool_use_id:
            # Check if a tool_result already exists for this tool_use_id
            for msg in self.messages:
                if msg['role'] == 'user':
                    msg_content = msg.get('content')
                    if isinstance(msg_content, list):
                        for block in msg_content:
                            if (
                                isinstance(block, dict)
                                and block['type'] == 'tool_result'
                                and block['tool_use_id'] == tool_use_id
                                and 'content' in block
                            ):
                                # Append to the existing tool_result
                                existing = block['content']

                                # Join together only the text blocks
                                # since we do not support multi-modality yet
                                if not isinstance(existing, str):
                                    existing = ''.join(b.get('text', '') for b in existing)

                                block['content'] = existing + '\n' + output
                                return

            result_block: ToolResultBlockParam = {
                'type': 'tool_result',
                'tool_use_id': tool_use_id,
                'content': output,
            }
            self.messages.append(MessageParam(role='user', content=[result_block]))
        else:
            # Fallback: insert as plain user message
            self.insert(role=('user', None), content=output)

    async def invoke_stream(
        self, ctx: InferenceConfig
    ) -> tuple[Awaitable[Generation], AsyncGenerator[Delta, None]]:
        """Stream inference using Anthropic's Messages API SSE events."""
        future: asyncio.Future[Generation] = asyncio.Future()
        queue: asyncio.Queue[Delta | None] = asyncio.Queue()

        async def _stream_task() -> None:
            body, _ = self._body(ctx)

            try:
                # Use the SDK's streaming interface
                async with self.client.messages.stream(
                    **body, extra_headers=ANTHROPIC_EXTRA_HEADERS
                ) as stream:
                    # Iterate over streaming events
                    async for event in stream:
                        if event.type == 'content_block_delta':
                            if event.delta.type == 'text_delta':
                                await queue.put(Delta(content=event.delta.text, type='output_text'))
                            elif event.delta.type == 'thinking_delta':
                                await queue.put(
                                    Delta(content=event.delta.thinking, type='reasoning')
                                )
                            # input_json_delta is accumulated silently by the SDK

                    final_message = await stream.get_final_message()

                _, generation = self._process_response(final_message, ctx)

                # Stream the code block to the client if it came from a tool_use block.
                # text_delta/thinking_delta are streamed during the loop above, but
                # input_json_delta (tool_use code) is only accumulated by the SDK —
                # we need to push it explicitly so the client sees the generated code.
                if generation.code and generation.code_from_tool:
                    await queue.put(
                        Delta(
                            content=f"\n<python>\n{generation.code}\n</python>",
                            type='output_text',
                        )
                    )

                future.set_result(generation)
            except AnthropicAPIError as e:
                if not future.done():
                    future.set_exception(_anthropic_error_to_generation_error(e))
                raise
            except BaseException as e:
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                await queue.put(None)

        asyncio.create_task(_stream_task())

        async def _delta_generator() -> AsyncGenerator[Delta, None]:
            while True:
                delta = await queue.get()
                if delta is None:
                    break
                yield delta

        return future, _delta_generator()


class UnauthorizedInferenceEndpoint(httpx.HTTPStatusError):
    """Raised when the inference endpoint returns a 401 Unauthorized response."""

    pass
