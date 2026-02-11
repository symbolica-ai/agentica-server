from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from com.abstract import Action
from com.context import Context

__all__ = [
    "ReplRunCode",
    "ReplGetSessionInfo",
    "ReplGetSessionInfoAttr",
    "ReplGetEvaluationInfo",
    "ReplGetEvaluationInfoAttr",
    "ReplCallMethod",
    "SandboxCallMethod",
    "ReplRaiseOrReturnVar",
]


if TYPE_CHECKING:
    from agentica_internal.repl.info import ReplEvaluationInfo, ReplSessionInfo


@dataclass
class ReplRunCode(Action['ReplEvaluationInfo']):
    """
    Represents running code via the AgentRepl inside the execution environment.
    Will provide the current `iid` as an option, which ensures that any `raise` or `return` will
    emit the correct FutureResponseMsg on the warp channel.
    """

    code: str
    opts: dict[str, bool | int | str | float | None]

    async def perform(self, ctx: Context) -> Any:
        return await ctx.sandbox.repl_run_code(self.code, iid=ctx.inference_config.iid, **self.opts)


# ------------------------------------------------------------------------------


@dataclass
class ReplGetSessionInfo(Action['ReplSessionInfo']):
    """
    Represents getting the ReplSessionInfo dataclass object cached on the Sandbox.
    """

    async def perform(self, ctx: Context) -> 'ReplSessionInfo':
        return ctx.sandbox.session_info


# ------------------------------------------------------------------------------


@dataclass
class ReplGetEvaluationInfo(Action['ReplEvaluationInfo']):
    """
    Represents getting the ReplEvaluationInfo dataclass object cached on the Sandbox.
    """

    async def perform(self, ctx: Context) -> 'ReplEvaluationInfo':
        return ctx.sandbox.eval_info


# ------------------------------------------------------------------------------


@dataclass
class ReplGetSessionInfoAttr[T](Action[T]):
    """
    Represents getting a property of the ReplSessionInfo dataclass object cached on the Sandbox.
    """

    attr: str

    async def perform(self, ctx: Context) -> T:
        return getattr(ctx.sandbox.session_info, self.attr)


# ------------------------------------------------------------------------------


@dataclass
class ReplGetEvaluationInfoAttr[T](Action[T]):
    """
    Represents getting a property of the ReplSessionInfo dataclass object cached on the Sandbox.
    """

    attr: str

    async def perform(self, ctx: Context) -> T:
        return getattr(ctx.sandbox.eval_info, self.attr)


# ------------------------------------------------------------------------------


@dataclass
class ReplCallMethod(Action[Any]):
    """
    Represents running an arbitrary method of the AgentRepl inside the execution environment.

    The returned result will must be JSON-serializable, and will be deserialized back into Python values.
    """

    method: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    async def perform(self, ctx: Context) -> Any:
        args, kwargs = self.args, self.kwargs
        # TODO: uncomment this when JSON-mode is re-enabled
        # args = tuple(ctx.inference_config.iid if a is IID_TOKEN else a for a in self.args)
        return await ctx.sandbox.repl_call_method(self.method, *args, **kwargs)


@dataclass
class SandboxCallMethod(Action[Any]):
    """
    Represents running an arbitrary method of the sandbox inside the execution environment.
    """

    method: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    async def perform(self, ctx: Context) -> Any:
        args, kwargs = self.args, self.kwargs
        sandbox_method = getattr(ctx.sandbox, self.method)
        return await sandbox_method(self.method, *args, **kwargs)


# legacy: only used by json_mode
@dataclass
class ReplRaiseOrReturnVar(Action[None]):
    """
    Represents returning or raising the content of a local variable. Legacy.
    """

    var_name: str
    is_raise: bool

    async def perform(self, ctx: Context) -> Any:
        if self.is_raise:
            return await ctx.sandbox.repl_raise_var(ctx.inference_config.iid, self.var_name)
        else:
            return await ctx.sandbox.repl_return_var(ctx.inference_config.iid, self.var_name)
