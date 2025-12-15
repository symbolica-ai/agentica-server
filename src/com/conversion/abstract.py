import abc
from typing import Any, ClassVar

from com.constraints import Constraint
from com.context import Context
from com.deltas import GeneratedDelta

__all__ = ["Converter"]


class Converter[T](abc.ABC):
    """
    Abstract base class for converters.
    """

    # fmt: off
    # ruff: noqa
    RateLimitCode:                  ClassVar[str | None] = "429"
    BadRequestCode:                 ClassVar[str | None] = "400"
    UnauthorizedCode:               ClassVar[str | None] = "401"
    InsufficientCreditsCode:        ClassVar[str | None] = "402"
    PermissionDeniedCode:           ClassVar[str | None] = "403"
    NotFoundCode:                   ClassVar[str | None] = "404"
    ConflictCode:                   ClassVar[str | None] = "409"
    RequestTooLargeCode:            ClassVar[str | None] = "413"
    UnprocessableEntityCode:        ClassVar[str | None] = "422"
    ServiceUnavailableCode:         ClassVar[str | None] = "503"
    OverloadedCode:                 ClassVar[str | None] = "529"
    DeadlineExceededCode:           ClassVar[str | None] = "504"
    # Missing codes:
    ModerationCode:                 ClassVar[str | None] = None
    TimeoutCode:                    ClassVar[str | None] = None
    ModelDownOrInvalidResponseCode: ClassVar[str | None] = None
    NoModelProviderAvailableCode:   ClassVar[str | None] = None
    # fmt: on

    @abc.abstractmethod
    async def _from(self, context: Context, constraints: list[Constraint]) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def _to(self, data: dict[str, Any], constraints: list[Constraint]) -> GeneratedDelta:
        raise NotImplementedError
