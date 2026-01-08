"""Shared httpx client for connection pooling.

Creating a new httpx.AsyncClient for every request is expensive because:
1. SSL/TLS context creation is a blocking CPU operation
2. Each request requires a new TCP connection + TLS handshake
3. No connection reuse means higher latency

This module provides a shared client that should be used across the application.
The client is initialized at startup and closed on shutdown.
"""

import httpx

# Module-level client, initialized at startup
_client: httpx.AsyncClient | None = None


def init_client(
    max_connections: int | None = None,
    max_keepalive_connections: int | None = 30,
    keepalive_expiry: float | None = 30.0,
) -> None:
    """Initialize the shared HTTP client. Call at application startup."""
    global _client
    if _client is not None:
        return
    # Configure connection pool limits
    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive_connections,
        max_connections=max_connections,
        keepalive_expiry=keepalive_expiry,
    )
    _client = httpx.AsyncClient(
        limits=limits,
        http2=True,  # Enable HTTP/2 for connection multiplexing
    )


def get_client() -> httpx.AsyncClient:
    """Get the shared httpx client.

    If init_client() wasn't called, creates client lazily.
    """
    global _client
    if _client is None:
        init_client()
    assert _client is not None
    return _client


async def close_client() -> None:
    """Close the shared client. Call on application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
