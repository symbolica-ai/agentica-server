from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    from com.context import Context

__all__ = [
    'Action',
    'HistoryMonad',
    'Pure',
    'Do',
]


# === History Monad ===


class HistoryMonad[A](ABC):
    """
    A value of type `HistoryMonad[A]` represents a computation which, when performed,
    make some side effect on the chat-state (e.g. history) resulting in some consequent value `A`.

    For example, `gen: HistoryMonad[str]` is a monad which *represents* calling inference on model,
    producing a `str` of the generated text;
    `insert: str -> HistoryMonad[None]` is a *function* which represents inserting some given text into the history.

    Using `.bind()`, we can compose monads: `gen.bind(insert)` means extract the `str` produced by `gen`,
    then pass it into `insert`. Therefore, `gen.bind(insert)` is a `HistoryMonad[None]` monad which represents
    the combined side effect of generating text and inserting it into the history.

    > A monad is just a monoid object in the monoidal monoidoid of endofunctors, what's the problem?
    """

    Do: ClassVar[type['Do']]
    Pure: ClassVar[type['Pure']]

    @staticmethod
    def pure[B](x: B) -> 'HistoryMonad[B]':
        """Wrap a plain value into the history monad, with no side effects."""
        return Pure(x)

    @staticmethod
    def immediate[B](f: Callable[[], 'HistoryMonad[B]']) -> 'HistoryMonad[B]':
        """Builds a monad as its needed, depending on the execution of `f`."""
        return Do(Noop(), lambda _: f())

    def bind[B](self, f: Callable[[A], 'HistoryMonad[B]']) -> 'HistoryMonad[B]':
        """
        Sequentially compose two actions,
        passing any value produced by the first as an argument to the second.

        ```python
        pure(3 + 2).bind(lambda x: pure(x * 2)) == pure(10)
        ```
        """

        if isinstance(self, Pure):
            return f(self.value)

        if isinstance(self, Do):
            return Do(self.action, lambda b: self.continuation(b).bind(f))

        raise NotImplementedError()

    # `>>` is a monadic bind `>>=` when the rhs is a lambda. otherwise `>>`.
    # (>>) :: m a -> m b -> m b
    # (>>=) :: m a -> (a -> m b) -> m b
    def __rshift__[B](
        self, other: 'HistoryMonad[B]' | Callable[[A], 'HistoryMonad[B]']
    ) -> 'HistoryMonad[B]':
        """
        Binds a monad, passing the result of the previous monad,
        or binds while ignoring the value if a lambda was not passed.

        ```
        # Generate text, insert it, then insert a newline
        gen() >> insert_string >> insert_string('\n')
        ```
        """

        if isinstance(other, HistoryMonad):
            return self.bind(lambda _: other)
        return self.bind(other)


@dataclass
class Pure[A](HistoryMonad[A]):
    """
    Lifts a value into the history monad, with no side effects.
    """

    value: A

    def __init__(self, value: A = None):
        self.value = value


@dataclass
class Do[A, B](HistoryMonad[A]):
    """
    Represents a monad action encapsulating a side effect
    along with how to pass the result of that action to the next monad.
    """

    action: 'Action[B]'
    continuation: Callable[[B], HistoryMonad[A]]


HistoryMonad.Pure = Pure
HistoryMonad.Do = Do


# === Actions ===


class Action[A](ABC):
    """
    This is how an action producing a value of type `A` is represented inside
    a monad `HistoryMonad[A]`.
    """

    @abstractmethod
    async def perform(self, ctx: 'Context') -> A:
        """
        Dictates how an action should produce a side effect on the chat `Context`,
        returning a value of type `A` of the action.
        """
        raise NotImplementedError()


class Noop(Action[None]):
    """An action that does nothing."""

    def __repr__(self) -> str:
        return 'Noop()'

    async def perform(self, ctx: 'Context') -> None:
        """Do nothing."""
        return None
