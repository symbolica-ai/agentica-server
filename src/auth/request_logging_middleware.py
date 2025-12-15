"""Request logging middleware for Litestar."""

import logging
import time

from litestar.middleware import AbstractMiddleware
from litestar.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(AbstractMiddleware):
    """Middleware that logs all HTTP requests with method, path, status code, and duration."""

    def __init__(self, app: ASGIApp):
        """Initialize the middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the request and log it.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Store start time and request info
        start_time = time.perf_counter()
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "UNKNOWN")

        # Track response status code
        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        # Process the request
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log after request completes
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"{method} {path} - {status_code or 'UNKNOWN'} - {duration_ms:.2f}ms")
