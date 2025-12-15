from typing import TYPE_CHECKING

from agentica_internal.internal_errors import *

from com.context import Context

__all__ = [
    'APIConnectionError',
    'APITimeoutError',
    'BadRequestError',
    'ConflictError',
    'ContentFilteringError',
    'DeadlineExceededError',
    'ExecutableError',
    'ExecutionError',
    'GenerationError',
    'InferenceError',
    'InternalServerError',
    'MaxTokensError',
    'NotFoundError',
    'OverloadedError',
    'PermissionDeniedError',
    'RateLimitError',
    'RequestTooLargeError',
    'ServiceUnavailableError',
    'UnauthorizedError',
    'UnprocessableEntityError',
    'ValidationError',
    'generation_error_from_http',
]

if TYPE_CHECKING:
    from httpx import HTTPError

    from com.context_json import Executable


# === Base Exceptions ===


class ExecutableError(Exception):
    """Base class for exceptions during handling executable objects."""

    executable: 'Executable | None' = None

    def __init__(self, message: str, executable: 'Executable | None' = None):
        super().__init__(message)
        self.executable = executable


# === Executable Exceptions ===


class ValidationError(ExecutableError):
    """Exception when validating executable arguments."""


class ExecutionError(ExecutableError):
    """Exception when executing executable code."""


def generation_error_from_http(ctx: Context, e: 'HTTPError') -> InferenceError:
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        assert hasattr(e, "response")
        status = str(e.response.status_code)
        # Prefer provider-specific codes first to avoid collisions with generic ones
        if (
            ctx.converter.NoModelProviderAvailableCode
            and status == ctx.converter.NoModelProviderAvailableCode
        ):
            return OverloadedError(
                request=e.request,
                response=e.response,
                prefix="No model provider available",
            )
        if (
            ctx.converter.ModelDownOrInvalidResponseCode
            and status == ctx.converter.ModelDownOrInvalidResponseCode
        ):
            return ServiceUnavailableError(
                request=e.request,
                response=e.response,
                prefix="Chosen model is down or invalid response",
            )
        if ctx.converter.TimeoutCode and status == ctx.converter.TimeoutCode:
            return DeadlineExceededError(request=e.request, response=e.response)
        if (
            ctx.converter.InsufficientCreditsCode
            and status == ctx.converter.InsufficientCreditsCode
        ):
            return InsufficientCreditsError(
                request=e.request,
                response=e.response,
                prefix="Insufficient credits",
            )
        if ctx.converter.ModerationCode and status == ctx.converter.ModerationCode:
            return PermissionDeniedError(
                request=e.request,
                response=e.response,
                prefix="Chosen model requires moderation and input has been flagged",
            )
        # Generic mappings
        if status == ctx.converter.RateLimitCode:
            return RateLimitError(request=e.request, response=e.response)
        if status == ctx.converter.BadRequestCode:
            return BadRequestError(request=e.request, response=e.response)
        if status == ctx.converter.UnauthorizedCode:
            return UnauthorizedError(request=e.request, response=e.response)
        if status == ctx.converter.PermissionDeniedCode:
            return PermissionDeniedError(request=e.request, response=e.response)
        if status == ctx.converter.NotFoundCode:
            return NotFoundError(request=e.request, response=e.response)
        if status == ctx.converter.ConflictCode:
            return ConflictError(request=e.request, response=e.response)
        if status == ctx.converter.UnprocessableEntityCode:
            return UnprocessableEntityError(request=e.request, response=e.response)
        if status == ctx.converter.RequestTooLargeCode:
            return RequestTooLargeError(request=e.request, response=e.response)
        if status == ctx.converter.OverloadedCode:
            return OverloadedError(request=e.request, response=e.response)
        if status == ctx.converter.ServiceUnavailableCode:
            return ServiceUnavailableError(request=e.request, response=e.response)
        return InternalServerError(request=e.request, response=e.response)
    elif isinstance(e, httpx.TimeoutException):
        return APITimeoutError(request=e.request, message="Timeout error.")
    else:
        return APIConnectionError(request=e.request, message="Connection error.")
