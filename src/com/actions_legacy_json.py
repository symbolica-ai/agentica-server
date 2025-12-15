import asyncio
import enum
import json
import random
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Literal

from com.abstract import Action
from com.constraints import *
from com.context import *
from com.context_legacy_json import *
from com.deltas import *
from com.exceptions import *
from messages import GenAIUsage

__all__ = [
    "GetExecutableNames",
    "GetExecutableDescriptions",
    "GetExecutableSchemas",
    "AddExecutable",
    "ExecuteEnd",
    "EndExecuted",
    "Generate",
    "ProcessResponse",
    "GenState",
    "GetState",
    "SetState",
    "EnterCodeState",
    "GetCode",
    "ClearCode",
    "AppendCode",
]


class GenState(enum.Enum):
    # Default, free-form text generation
    TEXT = 'text'  # Plain text generation
    # Code-generation states
    CODE = 'code'  # Code generation
    COMPOUND = 'compound'  # Continuation of code generation
    RESULT = 'result'  # Result
    # Intermediate states
    EVALUATE = 'evaluate'  # Evaluating code
    # End states
    FINISHED = 'finished'  # Success
    ERROR = 'error'  # Error

    def is_code_state(self) -> bool:
        return self in [GenState.CODE, GenState.COMPOUND, GenState.RESULT]

    def is_end_state(self) -> bool:
        return self in [GenState.FINISHED, GenState.ERROR]

    def is_compound(self) -> bool:
        return self == GenState.COMPOUND


@dataclass
class GetState(Action[GenState]):
    """
    Represents retrieving the state (`State`) of the context.
    """

    async def perform(self, ctx: Context) -> GenState:
        return ctx.state


@dataclass
class SetState(Action[None]):
    """
    Represents setting the state (`State`) of the context.
    """

    state: GenState

    async def perform(self, ctx: Context) -> None:
        ctx.state = self.state


@dataclass
class EnterCodeState(Action[None]):
    """
    Represents entering (or remaining in) a code state in the context.
    """

    async def perform(self, ctx: Context) -> None:
        if not ctx.state.is_code_state():
            ctx.state = GenState.CODE
        # Otherwise, leave us in the current type of code state.


@dataclass
class GetExecutableNames(Action[list[str]]):
    constraint_type: Literal["object", "callable"] = "callable"
    whitelist: list[str] | None = None
    blacklist: list[str] | None = None
    """
    Represents retrieving the schemas from the execution context.
    """

    async def perform(self, ctx: Context) -> list[str]:
        if self.constraint_type == "object":
            return [
                s.name
                for s in ctx.sandbox.object_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            ]
        elif self.constraint_type == "callable":
            return [
                s.name
                for s in ctx.sandbox.callable_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            ]
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


# ------------------------------------------------------------------------------


@dataclass
class GetExecutableDescriptions(Action[dict[str, str | None]]):
    constraint_type: Literal["object", "callable"] = "callable"
    whitelist: list[str] | None = None
    blacklist: list[str] | None = None
    """
    Represents retrieving the schemas from the execution context.
    """

    async def perform(self, ctx: Context) -> dict[str, str | None]:
        if self.constraint_type == "object":
            return {
                s.name: s.description
                for s in ctx.json.object_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            }
        elif self.constraint_type == "callable":
            return {
                s.name: s.description
                for s in ctx.json.callable_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            }
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


# ------------------------------------------------------------------------------


@dataclass
class GetExecutableSchemas(Action[dict[str, dict[str, Any]]]):
    constraint_type: Literal["object", "callable"] = "callable"
    whitelist: list[str] | None = None
    blacklist: list[str] | None = None
    """
    Represents retrieving the schemas from the execution context.
    """

    async def perform(self, ctx: Context) -> dict[str, dict[str, Any]]:
        if self.constraint_type == "object":
            return {
                s.name: await s.schema(ctx.sandbox)
                for s in ctx.json.object_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            }
        elif self.constraint_type == "callable":
            return {
                s.name: await s.schema(ctx.sandbox)
                for s in ctx.json.callable_executables.values()
                if (self.whitelist is None or s.name in self.whitelist)
                and (self.blacklist is None or s.name not in self.blacklist)
            }
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


# ------------------------------------------------------------------------------


@dataclass
class AddExecutable(Action[Context]):
    """
    Represents adding an executable to the execution context. It must already be in the sandbox.
    """

    name: str
    type: Literal["object", "callable"]

    async def perform(self, ctx: Context) -> None:
        if self.type == "object":
            ctx.json.executables[self.name] = Executable(self.name, "object")
        elif self.type == "callable":
            ctx.json.executables[self.name] = Executable(self.name, "callable")
        else:
            raise ValueError(f"Invalid type: {self.type}")


# ------------------------------------------------------------------------------


@dataclass
class ExecuteEnd(Action[list[Any] | None]):
    """
    Represents executing `Callable`s at the end of a generation, in case it's end reason is a `EndGenCallableTypes`.
    """

    end: EndGen

    async def perform(self, ctx: Context) -> list[Any] | None:
        if isinstance(self.end, EndGenCallableTypes):
            results: list[Any] = []
            for constraint, id, content in zip(
                self.end.constraints, self.end.ids, self.end.content
            ):
                executable = constraint.executable
                error_type: str | None = None
                error_message: str | None = None
                result: Any = None

                try:
                    result_obj = constraint.executable.name + "_result_" + id
                    result = await executable.exec(content, ctx.sandbox, result_obj)
                    results.append(result)
                except ValidationError as e:
                    error_type = "ValidationError"
                    error_message = str(e)
                    if not constraint.allow_partial:
                        raise e
                except ExecutionError as e:
                    error_type = "ExecutionError"
                    error_message = str(e)
                    raise e
                finally:
                    # Log tool execution to OpenTelemetry
                    if ctx.invocation:
                        from messages import GenAIToolEvent

                        # Convert content to dict if it's a string
                        tool_input = (
                            content if isinstance(content, dict) else {"arguments": content}
                        )

                        tool_event = GenAIToolEvent(
                            iid=ctx.gen.iid,
                            tool_id=id,
                            tool_name=executable.name,
                            input=tool_input,
                            output=result,
                            error_type=error_type,
                            error_message=error_message,
                        )
                        await ctx.invocation.log_genai_tool(tool_event)
            return results
        return None


# ------------------------------------------------------------------------------


class EndExecuted(Action[bool]):
    """
    Represents checking if the generation ended due to an executable.
    """

    async def perform(self, ctx: Context) -> bool:
        return isinstance(ctx.gen.deltas[-1].end, EndGenCallableTypes)


# ------------------------------------------------------------------------------


@dataclass
class Generate(Action[GeneratedDelta]):
    """
    Represents requesting the repl to generate a delta in the given context.
    """

    constraints: list[str] = field(default_factory=list)
    constraint_type: Literal["object", "callable"] = "callable"
    stop_tokens: list[StopTokensType] = field(default_factory=list)
    max_tokens: int | None = None
    max_retries: int | None = None

    def process_constraints(self, ctx: Context) -> list[Constraint]:
        if ctx.gen.type != "json" and ctx.gen.guided:
            raise ValueError(
                f"Lark support for generating callable arguments not implemented yet {ctx.gen.type=} {ctx.gen.guided=}"
            )

        stop_token_cons = [StopTokenConstraint._from(c) for c in self.stop_tokens]
        max_completion_tokens = (
            self.max_tokens if self.max_tokens is not None else ctx.gen.max_completion_tokens
        )
        max_tokens_cons = [MaxTokensConstraint._from(max_completion_tokens)]

        if self.constraint_type == "callable":
            if not self.constraints:
                self.constraints = [name for name in ctx.json.callable_executables.keys()]
            elif not all(c in ctx.json.callable_executables.keys() for c in self.constraints):
                raise ValueError(
                    f"Some callables were not found in the execution context: {self.constraints}"
                )

            cons = [
                CallableConstraint(
                    ctx.json.callable_executables[c],
                    ctx.gen.type,
                    ctx.gen.guided,
                    name=c,
                )
                for c in self.constraints
            ]
        else:
            if not all(c in ctx.json.object_executables.keys() for c in self.constraints):
                raise ValueError(
                    f"Some callables were not found in the execution context: {self.constraints}"
                )
            cons = [
                ObjectConstraint(
                    [ctx.json.object_executables[c] for c in self.constraints],
                    ctx.gen.type,
                    ctx.gen.guided,
                    ".".join(self.constraints),
                )  # type: ignore
            ]

        return stop_token_cons + cons + max_tokens_cons  # type: ignore

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
                        # TODO: audio
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

    async def try_inference(self, ctx: Context, cons: list[Constraint]) -> GeneratedDelta | None:
        streaming: bool = ctx.gen.streaming
        delay: float = ctx.gen.error_handling.rate_limit_delay
        exponential_base: float = ctx.gen.error_handling.rate_limit_exponential_base
        jitter: bool = ctx.gen.error_handling.rate_limit_jitter
        try:
            if streaming:
                _response, _stream = self.infer_streaming(ctx, cons)
                async for delta in _stream:
                    # TODO @samuel: extend the things we log, including tool calls, &c.
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

            if ctx.gen.streaming and ctx.invocation:
                usage = _usage_from_generated_delta(response)
                output_messages = _output_messages_from_generated_delta(response)
                # Note: server info is set during start_inference in InferenceEndpoint
                event = ctx.invocation.create_chat_event(
                    output_messages=output_messages,
                    usage=usage,
                    streaming=True,
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
                raise MaxTokensError(max_rounds)

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


# ------------------------------------------------------------------------------


@dataclass
class ProcessResponse(Action[Context]):
    """
    Represents inserting a generated delta into the history context.
    """

    response: GeneratedDelta
    callable_results: list[Any] | None = None

    async def perform(self, ctx: Context) -> Context:
        if self.callable_results is not None:
            if isinstance(self.response.end, EndGenCallableTypes):
                self.response.end.results = self.callable_results.copy()
            else:
                raise ValueError(
                    f"Callable results provided but generation did not stop for the execution of Callables: {self.response.end}"
                )
        await ctx.gen.push_delta(self.response)
        return ctx


def _usage_from_generated_delta(delta: GeneratedDelta) -> GenAIUsage | None:
    if isinstance(delta.usage, dict):
        return GenAIUsage.from_dict(delta.usage)
    return None


def _output_messages_from_generated_delta(delta: GeneratedDelta) -> list[dict[str, Any]]:
    message: dict[str, Any] = {"role": delta.name.name}
    if delta.content is not None:
        message["content"] = delta.content
    if delta.reasoning_content is not None:
        message["reasoning_content"] = delta.reasoning_content
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
        if tool_calls:
            message["tool_calls"] = tool_calls
    return [message]


@dataclass
class GetCode(Action[str]):
    """
    Represents retrieving the accumulated code buffer of the context.
    """

    async def perform(self, ctx: Context) -> str:
        return dedent(ctx.sandbox.code_buffer)


# ------------------------------------------------------------------------------


@dataclass
class ClearCode(Action[None]):
    """
    Represents clearing the code buffer held in the context.
    """

    async def perform(self, ctx: Context) -> None:
        ctx.sandbox.code_buffer = ""
        return None


# ------------------------------------------------------------------------------


@dataclass
class AppendCode(Action[None]):
    """
    Represents appending code to the code buffer held in the context.
    """

    code: str

    async def perform(self, ctx: Context) -> None:
        ctx.sandbox.code_buffer += self.code
        return None


# ------------------------------------------------------------------------------


@dataclass
class ExecuteCode(Action[None]):
    """
    Represents appending code to the code buffer held in the context.
    """

    code: str

    async def perform(self, ctx: Context) -> 'Repl':
        await ctx.sandbox.execute_code_buffer()
        return None
