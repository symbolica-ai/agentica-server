from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from com.abstract import Do, HistoryMonad, Pure
from com.actions_basic import *
from com.actions_model import ModelInference
from com.roles import *
from inference.endpoint import Generation, Role

__all__ = [
    "pure",
    "bypass",
    "combine",
    "lift",
    "insert_string",
    "capture",
    "retrieve",
    "notM",
    "ifM",
    "when",
    "guard",
    "whileM",
    "untilM",
    "is_python_sdk",
    "send_log",
    "log_code_block",
    "log_execute_result",
    "insert_execution_result",
    "insert_function_call",
    "model_inference",
]


if TYPE_CHECKING:
    from agentica_internal.session_manager_messages.session_manager_messages import (
        AllServerMessage,
    )


# Base monads

pure = Pure


# Utility monads


def bypass[A](
    f: HistoryMonad[Any] | Callable[[A], HistoryMonad[Any]],
) -> Callable[[A], HistoryMonad[A]]:
    """
    Compose (or bind the result of the monad) `f`, discarding the wrapped value
    and instead propagating the value of the left binding.

    For instance,
    ```python
    pure(3) >> bypass(f) >> lambda x: pure(x * 2)  # here `x` will be `3` regardless of `f`
    ```
    will result in `f` being composed (or applied with `3` and composed in if `f` is a function)
    and the result of `f` being discarded, instead preserving `3`, having the above end up wrapping the value `6`.
    """
    if callable(f):
        return lambda x: f(x) >> pure(x)
    else:
        return lambda x: f >> pure(x)


# ------------------------------------------------------------------------------


def combine[A, B](m1: HistoryMonad[A], m2: HistoryMonad[B]) -> HistoryMonad[tuple[A, B]]:
    """
    Compose two monads in the sequence they are given,
    resulting in a tuple of the values they produced, respectively.
    """
    return m1.bind(lambda x1: m2.bind(lambda x2: pure((x1, x2))))


# ------------------------------------------------------------------------------


# Lifting (a -> b) to (a -> m b)
# This is only done because (pure . f) is not convenient to write in Python.
def lift[A, B](f: Callable[[A], B]) -> Callable[[A], HistoryMonad[B]]:
    """
    Convert a function of `a -> b` to a function of `a -> HistoryMonad[b]`
    by wrapping the result in `pure`.
    """

    return lambda a: Pure(f(a))


# Insertion monad


def insert_string(content: str, role: 'Role') -> HistoryMonad[None]:
    """Verbatim insert some text into the prompt history."""
    return Do(Insert(content, role), Pure)


# Variable monads


def capture[A](name: str, m: HistoryMonad[A]) -> HistoryMonad[A]:
    """Capture the wrapped value of `m` under a variable name in the monad context."""
    return m >> (lambda x: Do(Capture(name, x), Pure))


def retrieve(name: str) -> HistoryMonad[Any]:
    """Retrieve a captured value of a variable from the monad context wrapped in a HistoryMonad."""
    return Do(Retrieve(name), Pure)


# Control flow monads


def notM(m: HistoryMonad[bool]) -> HistoryMonad[bool]:
    """Negates the wrapped boolean in a HistoryMonad."""
    return m >> (lambda b: pure(not b))


def ifM[A](cond: HistoryMonad[bool], m: HistoryMonad[A], n: HistoryMonad[A]) -> HistoryMonad[A]:
    """Given the result of composing `cond`, consequently compose in the `m` or `n` monad."""
    return cond >> (lambda c: m if c else n)


def when(cond: bool, m: HistoryMonad[None]) -> HistoryMonad[None]:
    """Given `cond`, compose in the `m` monad, otherwise compose in the effectless `pure(None)` monad."""
    return m if cond else pure(None)


def guard(
    cond: bool, consequent: HistoryMonad[None], alternative: HistoryMonad[None]
) -> HistoryMonad[None]:
    """Given `cond`, compose in the `consequent` or `alternative` monad."""
    return consequent if cond else alternative


def whileM(cond: HistoryMonad[bool]) -> HistoryMonad[None]:
    """While the monad evaluates to true, repeat composing the monad."""
    # TODO: this is recursive, should be done with a loop.
    return cond >> (lambda c: when(c, whileM(cond)))


def untilM(cond: HistoryMonad[bool]) -> HistoryMonad[None]:
    """Until the monad evaluates to true, repeat composing the monad."""

    return whileM(notM(cond))


# misc monads


def is_python_sdk() -> HistoryMonad[bool]:
    return Do(SDKIsPython(), Pure)


def send_log(event: 'AllServerMessage') -> HistoryMonad[None]:
    """Send a structured log message through the invocation context."""
    return Do(SendLog(event), Pure)


type UUID = str


def log_code_block(code: str) -> HistoryMonad[UUID]:
    """Log a code block being written."""
    return Do(LogCodeBlock(code), Pure)


def log_execute_result(result: str, exec_id: UUID) -> HistoryMonad[None]:
    """Log a code block being executed and its result."""
    return Do(LogExecuteResult(result, exec_id), Pure)


def insert_execution_result(output: str) -> HistoryMonad[None]:
    """Insert code execution result into the conversation.

    The InferenceSystem handles the appropriate format:
    - ResponsesSystem: tries tool result first, falls back to user message
    - ChatCompletionsSystem: always inserts as user message
    """
    return Do(InsertExecutionResult(output), Pure)


def insert_function_call(name: str, code: str, text: str = "") -> HistoryMonad[None]:
    """Insert a synthetic function call into the conversation for few-shot examples.

    This is used when uses_tool_calls=True to format few-shot examples as tool calls
    rather than markdown code blocks.

    Args:
        name: Tool name (e.g., "python")
        code: The code to execute
        text: Optional reasoning/explanation text to include in the assistant message
    """
    return Do(InsertFunctionCall(name, code, text), Pure)


# Generation monad


def model_inference() -> HistoryMonad[Generation]:
    """
    Query the inference system w.r.t. the current inference config (on the Context)
    to retrieve a `Generation` containing output text and optional code.
    """
    return Do(ModelInference(), Pure)
