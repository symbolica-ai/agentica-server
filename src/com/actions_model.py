import asyncio
import json
import random
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentica_internal.internal_errors import APIConnectionError, GenerationError, RateLimitError
from agentica_internal.internal_errors.generation import MaxRoundsError

from com.abstract import Action
from com.context import *
from inference.endpoint import Generation

__all__ = [
    "ModelInference",
]

if TYPE_CHECKING:
    pass


@dataclass
class ModelInference(Action[Generation]):
    """
    Represents requesting the model to generate a delta in the given context.
    """

    # TODO: rename to infer
    async def try_inference(self, ctx: Context) -> Generation | None:
        streaming: bool = ctx.inference_config.streaming
        delay: float = ctx.inference_config.error_handling.rate_limit_delay
        exponential_base: float = ctx.inference_config.error_handling.rate_limit_exponential_base
        jitter: bool = ctx.inference_config.error_handling.rate_limit_jitter

        try:
            if streaming:
                _response, deltas = await ctx.system.invoke_stream(ctx.inference_config)
                async for delta in deltas:
                    # Only log chunks that have content to stream
                    if delta.content:
                        # TODO: @samuel add proper type for `chunk`
                        chunk = {
                            'id': uuid.uuid4(),  # TODO: remove ID from both server and client
                            'role': delta.role,
                            'content': delta.content,
                            'type': delta.type,  # NOTE: unused by the client for now
                        }
                        await ctx.log('stream_chunk', chunk)
                response = await _response
            else:
                response = await ctx.system.invoke(ctx.inference_config)

            # Log the agent response delta for client echo stream (non-streaming case)
            # For streaming, chunks are already logged above via 'stream_chunk'
            if not streaming:
                # Log reasoning first (appears at start of agent message block)
                if response.reasoning:
                    await ctx.log(
                        'delta',
                        {
                            'role': 'agent',
                            'content': response.reasoning,
                            'type': 'reasoning',
                        },
                    )
                # Then log the regular output
                # Only append code block if it came from a tool call (code_from_tool=True)
                # When code_from_tool=False, the code was extracted from output_text which
                # already contains the markdown code block - appending it again causes duplication
                agent_content = response.output_text
                if response.code and response.code_from_tool:
                    agent_content += f"\n<python>\n{response.code}\n</python>"
                await ctx.log(
                    'delta',
                    {
                        'role': 'agent',
                        'content': agent_content,
                    },
                )

            # Log usage (appears at end of agent message block)
            await ctx.log(
                'delta',
                {
                    'role': 'agent',
                    'content': json.dumps(response.usage.model_dump()),
                    'type': 'usage',
                },
            )

            ctx.inference_config.spend_tokens(response.usage.output_tokens)

            if ctx.invocation:
                if response.usage is not None:
                    await ctx.invocation.log_usage(response.usage)
                # Note: server info is set during start_inference in InferenceEndpoint
                # Build content for telemetry - only append code separately if it came from tool call
                telemetry_content = response.output_text
                if response.code and response.code_from_tool:
                    telemetry_content += f"\n<python>\n{response.code}\n</python>"
                elif response.code:
                    # For markdown extraction, code is already in output_text
                    # Include it in telemetry content for completeness (already there)
                    pass
                output_message = {
                    'role': 'assistant',
                    'content': telemetry_content,
                }
                if response.reasoning is not None:
                    output_message['reasoning_content'] = response.reasoning

                await ctx.invocation.log_genai_chat(
                    ctx.invocation.create_chat_event(
                        output_messages=[output_message],
                        usage=response.usage,
                        streaming=ctx.inference_config.streaming,
                    )
                )

            return response
        except (RateLimitError, APIConnectionError):
            # Transient errors - retry with exponential backoff
            delay *= exponential_base * (1 + jitter * random.random())
            await asyncio.sleep(delay)
            return None
        except GenerationError as e:
            await ctx.inference_config.send_gen_err(e)
            raise e

    async def perform(self, ctx: Context) -> Generation:
        if max_rounds := ctx.inference_config.max_rounds:
            if ctx.inference_config.inference_rounds_count >= max_rounds:
                raise MaxRoundsError(max_rounds)

        max_retries: int = ctx.inference_config.error_handling.max_retries

        for _ in range(max_retries):
            response = await self.try_inference(ctx)
            if response is not None:
                # successful inference
                ctx.inference_config.inference_rounds_count += 1
                return response
        else:
            raise ValueError(
                f"Max retries reached for generating with inference config: {ctx.inference_config}"
            )
