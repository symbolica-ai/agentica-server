from collections.abc import Callable, Coroutine
from types import FunctionType
from typing import ParamSpec, TypeVar

from agentica_internal.repl.all import ReplEvaluationInfo, ReplSessionInfo, Scope

# these are loaded so that we can get type checking in repl_call_method and sandbox_call_method
from sandbox.guest.agent_repl import AgentRepl

from .abstract import Do, HistoryMonad, Pure
from .actions_repl import *

__all__ = [
    "repl_del_var",
    "repl_has_var",
    "repl_return_var",
    "repl_raise_var",
    "repl_run_code",
    "repl_call_method",
    "repl_session_info",
    "is_agentic_function",
    "is_returning_text",
    "ReplSessionInfo",
    "ReplEvaluationInfo",
]


def repl_del_var(name: str) -> HistoryMonad[None]:
    """Clears the given variable from the sandbox's locals."""
    return repl_call_method(AgentRepl.del_var, Scope.USER, name)


def repl_has_var(name: str) -> HistoryMonad[bool]:
    """Checks if the given variable is in the sandbox's locals."""
    return repl_call_method(AgentRepl.has_var, Scope.USER, name)


# TODO: remove
def repl_return_var(name: str) -> HistoryMonad[None]:
    """Materialize a sandbox value on the SDK."""
    return Do(ReplRaiseOrReturnVar(var_name=name, is_raise=False), Pure)


# TODO: remove
def repl_raise_var(name: str) -> HistoryMonad[None]:
    """Materialize a sandbox exception on the SDK."""
    return Do(ReplRaiseOrReturnVar(var_name=name, is_raise=True), Pure)


def repl_run_code(
    source: str, **options: bool | int | str | None
) -> HistoryMonad[ReplEvaluationInfo]:
    """Runs the given code in the sandbox, and returns a ReplEvaluationInfo.

    It takes simple options, these are provided to Sandbox.run_code. In the event that the evaluation
    returned a value or raised an exception syntactically, and the option `iid` is set to a string,
    the Sandbox will additionally encode the value/exception and return it to the SDK via a
    `FutureResultMsg` with `future_id = iid`.

    Any other options are forwarded to `AgentRepl.run_code(code, **options)`.
    """

    return Do(ReplRunCode(source, options), Pure)


def repl_session_info() -> HistoryMonad[ReplSessionInfo]:
    return Do(ReplGetSessionInfo(), Pure)


def repl_evaluation_info() -> HistoryMonad[ReplEvaluationInfo]:
    return Do(ReplGetEvaluationInfo(), Pure)


def is_agentic_function() -> HistoryMonad[bool]:
    return Do(ReplGetSessionInfoAttr('is_function'), Pure)


def is_returning_text() -> HistoryMonad[bool]:
    return Do(ReplGetSessionInfoAttr('is_returning_text'), Pure)


I = ParamSpec('I')
O = TypeVar('O')


def repl_call_method(
    method: Callable[[I], O], *args: I.args, **kwargs: I.kwargs
) -> HistoryMonad[O]:
    """Calls a method of the AgentRepl. inside the execution environment."""
    assert isinstance(method, FunctionType), "first arg must be a method of AgentRepl"
    assert method.__qualname__.startswith(('Repl.', 'AgentRepl.')), (
        "first arg must be a method of AgentRepl"
    )
    return Do(ReplCallMethod(method.__name__, args, kwargs), Pure)


def sandbox_call_method(
    method: Callable[[I], Coroutine[None, None, O]], *args, **kwargs
) -> HistoryMonad[O]:
    """Calls a method of the Sandbox. inside the execution environment."""
    assert isinstance(method, FunctionType), "first arg must be a method of Sandbox"
    assert method.__qualname__.startswith('Sandbox.'), "first arg must be a method of Sandbox"
    return Do(SandboxCallMethod(method.__name__, args, kwargs), Pure)
