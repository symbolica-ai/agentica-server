from dataclasses import dataclass

__all__ = [
    "Constraint",
    "MaxTokensConstraint",
    "StopTokenConstraint",
    "MaxTokensType",
    "StopTokensType",
    "ExecutableType",
    "ConstraintTypes",
]


type MaxTokensType = int
type StopTokensType = str
type ExecutableType = str
type ConstraintTypes = MaxTokensType | StopTokensType


@dataclass
class Constraint:
    """A constraint for generation."""


@dataclass
class MaxTokensConstraint(Constraint):
    """A constraint for the maximum number of tokens used."""

    max_tokens: MaxTokensType

    @classmethod
    def _from(cls, constraints: MaxTokensType) -> 'MaxTokensConstraint':
        """
        Convert any instances of `int` or `Iterable[int]` to a `MaxTokensConstraint`.
        """
        if isinstance(constraints, int):
            return MaxTokensConstraint(max_tokens=constraints)
        else:
            raise ValueError(f"Max tokens constraint must be an int, got {type(constraints)}")


@dataclass
class StopTokenConstraint(Constraint):
    """A constraint for stop tokens."""

    token: StopTokensType

    @classmethod
    def _from(cls, constraints: StopTokensType) -> 'StopTokenConstraint':
        """
        Convert any instances of `str` or `Iterable[str]` to a `StopTokenConstraint`.
        """
        if isinstance(constraints, str):
            return cls(token=constraints)
        else:
            raise ValueError(f"Stop tokens constraint must be a str, got {type(constraints)}")
