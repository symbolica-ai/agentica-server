from dataclasses import dataclass
from typing import Any

from agentica_internal.session_manager_messages import AllServerMessage
from agentica_internal.session_manager_messages.session_manager_messages import (
    InteractionCodeBlock,
    InteractionExecuteResult,
)

from com.abstract import Action
from com.context import Context
from com.gen_model import Role

__all__ = [
    "SendLog",
    "LogCodeBlock",
    "LogExecuteResult",
    "Capture",
    "Insert",
    "Retrieve",
    "SDKIsPython",
    "InsertExecutionResult",
    "InsertFunctionCall",
]

# ------------------------------------------------------------------------------


@dataclass
class SendLog(Action[None]):
    """Send a session manager log message."""

    message: AllServerMessage

    async def perform(self, ctx: Context) -> None:
        if ctx.invocation is None:
            return
        await ctx.invocation.log_message(self.message)


type UUID = str


@dataclass
class LogCodeBlock(Action[UUID]):
    """Log a code block."""

    code: str

    async def perform(self, ctx: Context) -> UUID:
        interaction = InteractionCodeBlock(code=self.code)
        if inv := ctx.invocation:
            await inv.log_interaction(interaction)
        return interaction.exec_id


@dataclass
class LogExecuteResult(Action[None]):
    """Log a execute result."""

    result: str
    exec_id: UUID

    async def perform(self, ctx: Context) -> None:
        interaction = InteractionExecuteResult(result=self.result, exec_id=self.exec_id)
        if inv := ctx.invocation:
            await inv.log_interaction(interaction)


# ------------------------------------------------------------------------------


@dataclass
class Insert(Action[None]):
    """
    Represents inserting raw text into the history context under a given role.
    """

    content: str
    role: 'Role'

    async def perform(self, ctx: Context) -> None:
        ctx.system.insert(self.role, self.content)
        # Log the delta for the client's echo stream (restored from GenModel.push_delta)
        info: dict[str, Any] = {
            'role': self.role[0] if isinstance(self.role, tuple) else self.role,
            'content': self.content,
        }
        if isinstance(self.role, tuple) and self.role[1] is not None:
            info['username'] = self.role[1]
        await ctx.log('delta', info)


# ------------------------------------------------------------------------------


@dataclass
class SDKIsPython(Action[bool]):
    async def perform(self, ctx: Context) -> bool:
        return "python" in ctx.protocol


# ------------------------------------------------------------------------------


@dataclass
class Capture[A](Action[A]):
    """
    Represents capturing a value in the history context under a given variable name.
    """

    variable: str
    x: A

    async def perform(self, ctx: Context) -> A:
        ctx.captures[self.variable] = self.x
        return self.x


# ------------------------------------------------------------------------------


@dataclass
class Retrieve(Action[Any]):
    """
    Represents retrieving a value from the history context under a given variable name.
    """

    variable: str

    async def perform(self, ctx: Context) -> Any:
        return ctx.captures[self.variable]


# ------------------------------------------------------------------------------


@dataclass
class InsertExecutionResult(Action[None]):
    """Insert code execution result into the conversation.

    Delegates to the InferenceSystem which handles the appropriate format:
    - ResponsesSystem: tries tool result first, falls back to user message
    - ChatCompletionsSystem: always inserts as user message
    """

    output: str

    async def perform(self, ctx: Context) -> None:
        ctx.system.insert_execution_result(self.output)
        # Log the delta for the client's echo stream (execution output includes stdout)
        info: dict[str, Any] = {
            'role': 'user',
            'content': self.output,
            'username': 'execution',
        }
        await ctx.log('delta', info)


# ------------------------------------------------------------------------------


@dataclass
class InsertFunctionCall(Action[None]):
    """Insert a synthetic function call into the conversation for few-shot examples.

    This is used when uses_tool_calls=True to format few-shot examples as tool calls
    rather than markdown code blocks.
    """

    name: str
    code: str
    text: str = ""

    async def perform(self, ctx: Context) -> None:
        ctx.system.insert_function_call(self.name, self.code, self.text)
