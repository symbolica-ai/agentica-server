from functools import partial
from typing import Any, Callable, ClassVar, Protocol

__all__ = [
    "apply",
    "bimap",
    "compose",
    "const",
    "dup",
    "fst",
    "identity",
    "partial",
    "snd",
]


def fst[A, B](x: tuple[A, B]) -> A:
    """Access the first element of a tuple."""
    return x[0]


def snd[A, B](x: tuple[A, B]) -> B:
    """Access the second element of a tuple."""
    return x[1]


def dup[A](x: A) -> tuple[A, A]:
    """Duplicate a value to into a pair of the same value `x -> (x, x)`."""
    return x, x


def bimap[A, B, C, D](
    f: Callable[[A], B],
    g: Callable[[C], D],
) -> Callable[[tuple[A, C]], tuple[B, D]]:
    """Apply a pair of functions to a pair of values."""
    return lambda x: (f(x[0]), g(x[1]))


def apply[*A, B](f: Callable[[*A], B], args: tuple[*A]) -> B:
    """Apply a function given its arguments list."""
    return f(*args)


def compose[*A, B, C](f: Callable[[B], C], g: Callable[[*A], B]) -> Callable[[*A], C]:
    """Compose two functions, `f` and `g` to the function `lambda x: f(g(x))`."""
    return lambda *args: f(g(*args))


def identity[A](x: A) -> A:
    """Returns the input unchanged."""
    return x


def const[A](x: A) -> Callable[..., A]:
    """Creates a function that produces this constant value."""
    return lambda *_, **__: x


class Monad[A](Protocol):
    """Protocol defining which methods a monad should implement."""

    Do: ClassVar[type['Do']]
    Pure: ClassVar[type['Pure']]

    @staticmethod
    def pure[B](x: B) -> 'Monad[B]': ...
    def bind[B](self, f: Callable[[A], 'Monad[B]']) -> 'Monad[B]': ...
    def __rshift__[B](self, other: 'Monad[B]' | Callable[[A], 'Monad[B]']) -> 'Monad[B]': ...


class Action[A](Protocol):
    """Protocol for a monad action."""

    def perform(self, ctx: Any) -> A: ...


class Do[A, B](Monad[A]):
    """Protocol for a monad representing a chained action."""

    action: Action[B]
    continuation: Callable[[B], Monad[A]]


class Pure[A](Monad[A]):
    """Protocol for a monad wrapping a value."""

    value: A


# (m a, m b) -> m (a, b)
# do { x1 <- m1; x2 <- m2; return (x1, x2) }
def combine[A, B](m1: Monad[A], m2: Monad[B]) -> Monad[tuple[A, B]]:
    """
    Compose two monads in the sequence they are given,
    resulting in a tuple of the values they produced, respectively.
    """

    pure = m1.__class__.pure
    return m1.bind(lambda x1: m2.bind(lambda x2: pure((x1, x2))))
