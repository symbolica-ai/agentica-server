from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentica_internal.session_manager_messages import AllServerMessage
from agentica_internal.session_manager_messages.session_manager_messages import (
    InteractionCodeBlock,
    InteractionExecuteResult,
)

from com.abstract import Action
from com.context import Context

if TYPE_CHECKING:
    # bring down load and type checker time: these are only used as annotations

    from com.deltas import GenRole


__all__ = [
    "SendLog",
    "LogCodeBlock",
    "LogExecuteResult",
    "Capture",
    "Insert",
    "Retrieve",
    "SDKIsPython",
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
    name: 'GenRole'

    async def perform(self, ctx: Context) -> None:
        from uuid import uuid4

        from com.deltas import Delta

        await ctx.gen.push_delta(Delta(id=str(uuid4()), name=self.name, content=self.content))


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
