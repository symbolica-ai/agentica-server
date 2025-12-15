import asyncio
import random
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentica_internal.internal_errors.generation import MaxRoundsError

from com.abstract import Action
from com.constraints import *
from com.context import *
from com.deltas import *
from com.exceptions import *
from messages import GenAIUsage

__all__ = [
    "ModelInference",
    "InsertDelta",
]

if TYPE_CHECKING:
    from messages import GenAIUsage


@dataclass
class ModelInference(Action[GeneratedDelta]):
    """
    Represents requesting the model to generate a delta in the given context.
    """

    constraints: list[str] = field(default_factory=list)
    stop_tokens: list[StopTokensType] = field(default_factory=list)
    max_tokens: int | None = None
    max_retries: int | None = None

    def process_constraints(self, ctx: Context) -> list[Constraint]:
        cons: list[Constraint] = []
        stop_token_cons = [StopTokenConstraint._from(c) for c in self.stop_tokens]
        cons.extend(stop_token_cons)
        max_completion_tokens = self.max_tokens or ctx.gen.max_tokens()
        if max_completion_tokens is not None:
            cons.append(MaxTokensConstraint._from(max_completion_tokens))
        return cons

    async def _infer_streaming(
        self, ctx: Context, cons: list[Constraint]
    ) -> AsyncGenerator[GeneratedDelta, None]:
        import httpx

        try:
            api_input = await ctx.converter._from(ctx, cons)
            async for api_output in ctx.gen.endpoint.invoke_stream(
                api_input,  # type: ignore
                timeout=ctx.gen.error_handling.read_timeout,
                iid=ctx.gen.iid,
                invocation=ctx.invocation,
            ):
                yield ctx.converter._to(api_output, cons, streaming=True)
        except httpx.HTTPError as e:
            raise generation_error_from_http(ctx, e) from e

    def infer_streaming(
        self, ctx: Context, cons: list[Constraint]
    ) -> tuple[Awaitable[GeneratedDelta], AsyncIterator[GeneratedDelta]]:
        """
        Build a glued-together generation delta from streamed smaller generation deltas.

        NOTE: If you decide to await the glued-together delta, you *MUST* have started consuming the stream first.
        """
        future_delta: asyncio.Future[GeneratedDelta] = asyncio.Future()
        stream: AsyncIterator[GeneratedDelta] = self._infer_streaming(ctx, cons)

        async def stream_wrapper() -> AsyncGenerator[GeneratedDelta, None]:
            from agentica_internal.core.json import json_merge

            """Does all the gluing together of deltas."""
            glued: GeneratedDelta | None = None
            async for delta in stream:
                if glued is None:
                    # We keep around certain things from the very first delta, namely:
                    # - `id` (we just need a unique one)
                    # - `name` (/role, this should never change)
                    glued = delta
                else:
                    # Logic for gluing together deltas
                    if content := delta.content:
                        if glued.content is None:
                            glued.content = content
                        else:
                            glued.content += content
                    if usage := delta.usage:
                        if glued.usage is None:
                            glued.usage = usage
                        else:
                            # Sum together usages recursively in two dicts.
                            # TODO: Standardize a Usage type instead of dict[str, Any]
                            glued.usage = json_merge(glued.usage, usage)
                    if reasoning_content := delta.reasoning_content:
                        if glued.reasoning_content is None:
                            glued.reasoning_content = reasoning_content
                        else:
                            glued.reasoning_content += reasoning_content
                    if not isinstance(delta.end, EndGenEOS):
                        # Only update the end if it's not the end of stream
                        glued.end = delta.end
                    if annotations := delta.annotations:
                        # TODO: Standardize an Annotations type instead of dict[str, Any]
                        if glued.annotations is None:
                            glued.annotations = annotations
                        else:
                            glued.annotations = json_merge(glued.annotations, annotations)
                    if audio := delta.audio:
                        # TODO: Standardize an Audio type instead of dict[str, Any]
                        # TODO: How am I supposed to glue audio?
                        if glued.audio is None:
                            glued.audio = audio
                        else:
                            glued.audio = json_merge(glued.audio, audio)
                    if (refusal := delta.refusal) and glued.refusal is None:
                        glued.refusal = refusal
                yield delta

            assert glued is not None, "Empty stream!"
            # Final glued delta is ready.
            future_delta.set_result(glued)

        return future_delta, stream_wrapper()

    async def infer(self, ctx: Context, cons: list[Constraint]) -> GeneratedDelta:
        import httpx

        try:
            api_input = await ctx.converter._from(ctx, cons)
            api_output = await ctx.gen.endpoint.invoke(
                api_input,  # type: ignore
                timeout=ctx.gen.error_handling.read_timeout,
                iid=ctx.gen.iid,
                invocation=ctx.invocation,
            )
            response = ctx.converter._to(api_output, cons, streaming=False)
            return response
        except httpx.HTTPError as e:
            raise generation_error_from_http(ctx, e) from e

    def usage(self, ctx: Context, delta: GeneratedDelta) -> GenAIUsage | None:
        """Get the usage for a generated delta and subtract it from the completion tokens."""
        gen_usage = _usage_from_generated_delta(delta)
        if gen_usage is None:
            return None
        if completion_tokens := gen_usage.completion_tokens:
            ctx.gen.spend_tokens(completion_tokens)
        return gen_usage

    async def try_inference(self, ctx: Context, cons: list[Constraint]) -> GeneratedDelta | None:
        streaming: bool = ctx.gen.streaming
        delay: float = ctx.gen.error_handling.rate_limit_delay
        exponential_base: float = ctx.gen.error_handling.rate_limit_exponential_base
        jitter: bool = ctx.gen.error_handling.rate_limit_jitter

        try:
            if streaming:
                _response, _stream = self.infer_streaming(ctx, cons)
                async for delta in _stream:
                    chunk = {
                        'id': delta.id,
                        'role': delta.name.name,
                        'content': delta.content,
                        'reasoning_content': delta.reasoning_content,
                        'usage': delta.usage,
                    }
                    await ctx.log('stream_chunk', chunk)
            else:
                _response = self.infer(ctx, cons)

            response = await _response
            usage = self.usage(ctx, response)

            if ctx.invocation:
                if usage is not None:
                    await ctx.invocation.log_usage(usage)
                output_messages = _output_messages_from_generated_delta(response)
                # Note: server info is set during start_inference in InferenceEndpoint
                event = ctx.invocation.create_chat_event(
                    output_messages=output_messages,
                    usage=usage,
                    streaming=ctx.gen.streaming,
                )
                await ctx.invocation.log_genai_chat(event)

            if isinstance(response.end, EndGenMaxTokens):
                raise MaxTokensError(response.end.constraint.max_tokens)
            if isinstance(response.end, EndGenStopToken) and response.end.filtered:
                raise ContentFilteringError()
            return response
        except RateLimitError as e:
            delay *= exponential_base * (1 + jitter * random.random())
            await asyncio.sleep(delay)
            return None
        except GenerationError as e:
            await ctx.gen.send_gen_err(e)
            raise e

    async def perform(self, ctx: Context) -> GeneratedDelta:
        if max_rounds := ctx.gen.max_rounds:
            if ctx.gen.inference_rounds_count >= max_rounds:
                raise MaxRoundsError(max_rounds)

        cons: list[Constraint] = self.process_constraints(ctx)
        max_retries: int = self.max_retries or ctx.gen.error_handling.max_retries

        for _ in range(max_retries):
            response = await self.try_inference(ctx, cons)
            if response is not None:
                # successful inference
                ctx.gen.inference_rounds_count += 1
                return response
        else:
            raise ValueError(f"Max retries reached for generating with constraints: {cons}")


@dataclass
class InsertDelta(Action[Context]):
    """
    Represents inserting a generated delta into the history context.
    """

    response: GeneratedDelta

    async def perform(self, ctx: Context) -> Context:
        await ctx.gen.push_delta(self.response)
        return ctx


def _usage_from_generated_delta(delta: GeneratedDelta) -> 'GenAIUsage | None':
    from messages import GenAIUsage

    if isinstance(delta.usage, dict):
        return GenAIUsage.from_dict(delta.usage)
    return None


def _output_messages_from_generated_delta(delta: GeneratedDelta) -> list[dict[str, Any]]:
    message: dict[str, Any] = {"role": delta.name.name}
    if delta.content is not None:
        message["content"] = delta.content
    if delta.reasoning_content is not None:
        message["reasoning_content"] = delta.reasoning_content
    return [message]
