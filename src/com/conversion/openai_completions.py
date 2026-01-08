import time
from typing import TYPE_CHECKING, Any

from com.constraints import Constraint
from com.context import Context
from com.deltas import GeneratedDelta

from .abstract import Converter

__all__ = [
    'OpenAIChatCompletionConverter',
]

if TYPE_CHECKING:
    from ..apis.openai_chat_completions import OpenAIChatCompletionInput


def is_empty_content(content: Any) -> bool:
    if isinstance(content, str):
        return content is None or content == ''
    elif isinstance(content, dict):
        return content.get('type') == 'text' and (
            content.get('text') is None or content.get('text') == ''
        )
    elif isinstance(content, list):
        return all(is_empty_content(item) for item in content)
    return False


# === Converter ===


class OpenAIChatCompletionConverter(Converter['OpenAIChatCompletionInput']):
    async def _from(
        self, context: Context, constraints: list[Constraint], validate: bool = True
    ) -> 'OpenAIChatCompletionInput':
        """
        Convert from a Context object to an OpenAIChatCompletionInput object for generation.

        Args:
            context: The Context object to convert from.
            constraints: The list of constraints to use during generation.

        Returns:
            OpenAIChatCompletionInput: The OpenAIChatCompletionInput object.
        """
        from ..apis.openai_chat_completions import OpenAIChatCompletionInput
        from .openai_chat_completions import convert_from_constraints, convert_from_deltas

        # See https://platform.openai.com/docs/guides/prompt-caching#best-practices for more details.
        # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#best-practices-for-effective-caching  for more details.

        # Adding static content at the beginning for Anthropic caching
        kwargs: dict[str, Any] = {
            'model': context.gen.model,
        }
        if context.gen.streaming:
            kwargs['stream'] = True
            kwargs['stream_options'] = {'include_usage': True}

        if context.gen.endpoint.user_id is not None:
            kwargs['user'] = context.gen.endpoint.user_id

        if context.gen.reasoning_effort is not None:
            # as per https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
            kwargs['reasoning'] = {'effort': context.gen.reasoning_effort}

        constraint_kwargs = dict()
        constraint_kwargs = await convert_from_constraints(
            context,
            constraints,
            validate=validate,
        )
        model_provider = "anthropic" if context.gen.model.startswith('anthropic') else "openai"
        messages = convert_from_deltas(context.gen.deltas, model_provider)
        kwargs['messages'] = [m for m in messages if not is_empty_content(m.get('content'))]

        # Adding caching for OpenAI models
        if model_provider == "openai":
            condition = (
                lambda cache_key: cache_key.num_requests <= 15
                and time.time() - cache_key.first_request_time <= 60
            )
            kwargs.update(constraint_kwargs)
            kwargs['prompt_cache_key'] = context.gen.use_cache_key(condition)

        # Set up num blocks and num breakpoints
        num_blocks, num_breakpoints = 0, 4

        # if tools := constraint_kwargs.get('tools', []):
        #     if constraint_kwargs.get('tool_choice', 'auto') == 'auto' and len(tools) == len(
        #         context.json.executables
        #     ):
        #         # If tools are present and they are equal to the number of executables, assume they are static content
        #         if 'tool_choice' in constraint_kwargs:
        #             kwargs['tool_choice'] = constraint_kwargs.pop('tool_choice')
        #         else:
        #             kwargs['tool_choice'] = 'auto'
        #         kwargs['tools'] = constraint_kwargs.pop('tools')
        #         # Add everything else
        #         kwargs.update(constraint_kwargs)
        # else:
        kwargs.update(constraint_kwargs)

        # Now add breakpoints in text content blocks in messages starting from the end
        for m in reversed(kwargs["messages"]):
            if num_breakpoints == 0:
                break
            if m["content"] is not None:
                for b in reversed(m["content"]):
                    if num_breakpoints == 0:
                        break
                    if isinstance(b, dict) and "type" in b and b["type"] == "text":
                        if num_blocks % 20 == 0:
                            b["cache_control"] = {"type": "ephemeral"}
                            num_breakpoints -= 1
                        num_blocks += 1

        return OpenAIChatCompletionInput(**kwargs)

    def _to(
        self, data: dict[str, Any], constraints: list[Constraint], *, streaming: bool = False
    ) -> GeneratedDelta:
        """
        Convert to a GeneratedDelta object post-generation from the JSON OpenAI ChatCompletion response.

        Args:
            data: The JSON OpenAI ChatCompletion response to be converted.
            constraints: The list of constraints that were used during generation.

        Returns:
            GeneratedDelta: The GeneratedDelta object.
        """
        from .openai_chat_completions import convert_to_delta

        # TODO: nuance about ['choices'][0] here?
        delta = convert_to_delta(data['choices'][0], constraints, streaming=streaming)
        delta.usage = data.get('usage', None)
        return delta
