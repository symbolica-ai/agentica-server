"""Stress tests for HTTP client connection pooling.

Tests the http_client module under various concurrent load scenarios to verify:
- Connection limits are respected
- Connection reuse works properly
- Behavior under high concurrency
"""

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from os import getenv

import httpx
import pytest
import pytest_asyncio
import uvicorn
from litestar import Litestar, route

from application import http_client

logging.getLogger("httpx").setLevel(logging.WARNING)

# Shared state for the test server (reset per server instance)
_server_state: dict = {}

# printing is disabled locally so running all tests doesn't produce
# a bunch of hard-to-read noise
if getenv("LOCAL_TESTING"):

    def print(*args, **kwargs):
        pass


def create_test_app(response_delay: float = 0.0) -> Litestar:
    """Create a Litestar app for testing."""

    @route("/api/test", http_method=["GET", "POST"])
    async def handle_request() -> dict:
        _server_state["request_count"] = _server_state.get("request_count", 0) + 1
        _server_state["concurrent"] = _server_state.get("concurrent", 0) + 1
        _server_state["max_concurrent"] = max(
            _server_state.get("max_concurrent", 0), _server_state["concurrent"]
        )

        if response_delay > 0:
            await asyncio.sleep(response_delay)

        _server_state["concurrent"] -= 1
        return {
            "status": "ok",
            "request_id": _server_state["request_count"],
        }

    @route("/stats", http_method=["GET"])
    async def get_stats() -> dict:
        return {
            "total_requests": _server_state.get("request_count", 0),
            "max_concurrent": _server_state.get("max_concurrent", 0),
            "current_concurrent": _server_state.get("concurrent", 0),
        }

    return Litestar(route_handlers=[handle_request, get_stats])


@contextmanager
def mock_http_server(
    host: str = "127.0.0.1",
    port: int = 9876,
    response_delay: float = 0.0,
) -> Iterator[str]:
    """Create a simple HTTP server for testing using litestar/uvicorn.

    Args:
        host: Host to bind to
        port: Port to bind to
        response_delay: Artificial delay before responding (seconds)

    Yields:
        Base URL of the server
    """
    global _server_state
    _server_state = {}

    app = create_test_app(response_delay)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run server in a background thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):  # 5 second timeout
        time.sleep(0.1)
        if server.started:
            break
    else:
        raise RuntimeError("Server failed to start")

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


@pytest_asyncio.fixture
async def reset_http_client():
    """Fixture to ensure clean http_client state between tests."""
    # Close any existing client
    await http_client.close_client()
    yield
    # Clean up after test
    await http_client.close_client()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_basic_connection_pooling(reset_http_client):
    """Test that basic connection pooling works with default settings."""
    with mock_http_server(port=9871) as base_url:
        http_client.init_client()
        client = http_client.get_client()

        # Make several sequential requests
        for _ in range(5):
            response = await client.get(f"{base_url}/api/test")
            assert response.is_success, f"Expected success, got {response.status_code}"
            data = response.json()
            assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_concurrent_requests_within_limit(reset_http_client):
    """Test many concurrent requests within connection limits."""
    max_connections = 20
    num_requests = 50

    with mock_http_server(port=9872, response_delay=0.05) as base_url:
        http_client.init_client(
            max_connections=max_connections,
            max_keepalive_connections=10,
        )
        client = http_client.get_client()

        async def make_request(req_id: int) -> dict:
            response = await client.post(
                f"{base_url}/api/test",
                json={"request_id": req_id},
            )
            return response.json()

        # Fire all requests concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        elapsed = time.perf_counter() - start_time

        # All requests should succeed
        assert len(results) == num_requests
        assert all(r["status"] == "ok" for r in results)

        # Get server stats
        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        # Max concurrent should not exceed our connection limit
        assert stats["max_concurrent"] <= max_connections, (
            f"Max concurrent {stats['max_concurrent']} exceeded limit {max_connections}"
        )

        print(f"\n  {num_requests} requests completed in {elapsed:.3f}s")
        print(f"  Max concurrent requests observed: {stats['max_concurrent']}")


@pytest.mark.asyncio
async def test_connection_limit_enforced(reset_http_client):
    """Test that connection limits are actually enforced under load."""
    max_connections = 5
    num_requests = 30
    response_delay = 0.1  # 100ms delay to force connection queuing

    with mock_http_server(port=9873, response_delay=response_delay) as base_url:
        http_client.init_client(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        client = http_client.get_client()

        async def make_request(req_id: int) -> dict:
            response = await client.post(
                f"{base_url}/api/test",
                json={"request_id": req_id},
            )
            return response.json()

        # Fire all requests concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        elapsed = time.perf_counter() - start_time

        assert len(results) == num_requests

        # Get server stats
        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        # Connection limit should be enforced
        assert stats["max_concurrent"] <= max_connections, (
            f"Connection limit violated: {stats['max_concurrent']} > {max_connections}"
        )

        # With 30 requests, 5 connections, 100ms each -> minimum ~600ms
        # (6 batches of 5 requests)
        min_expected_time = (num_requests / max_connections) * response_delay * 0.8
        assert elapsed >= min_expected_time, (
            f"Requests completed too fast ({elapsed:.3f}s), connection pooling may not be working"
        )

        print(f"\n  {num_requests} requests with {max_connections} max connections")
        print(f"  Completed in {elapsed:.3f}s (min expected: {min_expected_time:.3f}s)")
        print(f"  Max concurrent: {stats['max_concurrent']}")


@pytest.mark.asyncio
async def test_high_concurrency_stress(reset_http_client):
    """Stress test with high number of concurrent requests."""
    max_connections = 100
    num_requests = 500

    with mock_http_server(port=9874, response_delay=0.01) as base_url:
        http_client.init_client(
            max_connections=max_connections,
            max_keepalive_connections=30,
        )
        client = http_client.get_client()

        async def make_request(req_id: int) -> tuple[int, bool]:
            try:
                response = await client.post(
                    f"{base_url}/api/test",
                    json={"request_id": req_id},
                    timeout=30.0,
                )
                return (req_id, response.is_success)
            except httpx.HTTPError as e:
                print(f"Request {req_id} failed: {e}")
                return (req_id, False)

        start_time = time.perf_counter()
        results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        elapsed = time.perf_counter() - start_time

        successful = sum(1 for _, success in results if success)
        failed = num_requests - successful

        # All requests should succeed
        assert successful == num_requests, f"{failed} requests failed out of {num_requests}"

        # Get server stats
        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        print("\n  High concurrency stress test:")
        print(f"  {num_requests} requests, {max_connections} max connections")
        print(f"  Completed in {elapsed:.3f}s")
        print(f"  Max concurrent: {stats['max_concurrent']}")
        print(f"  Throughput: {num_requests / elapsed:.1f} req/s")


@pytest.mark.asyncio
async def test_keepalive_connection_reuse(reset_http_client):
    """Test that keepalive connections are properly reused."""
    with mock_http_server(port=9875) as base_url:
        http_client.init_client(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=30.0,
        )
        client = http_client.get_client()

        # Make requests in batches to test connection reuse
        batch_size = 10
        batch_count = 3
        for _ in range(batch_count):
            requests = [client.get(f"{base_url}/api/test") for _ in range(batch_size)]
            responses = await asyncio.gather(*requests)
            assert all(r.is_success for r in responses)

            # Small delay between batches
            await asyncio.sleep(0.1)

        # All requests should have succeeded using pooled connections
        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()
        assert stats["total_requests"] == batch_size * batch_count


@pytest.mark.asyncio
async def test_lazy_client_initialization(reset_http_client):
    """Test that client is lazily initialized if init_client wasn't called."""
    with mock_http_server(port=9876) as base_url:
        # Don't call init_client - it should be created lazily
        client = http_client.get_client()
        assert client is not None

        response = await client.get(f"{base_url}/api/test")
        assert response.is_success, f"Expected success, got {response.status_code}"


@pytest.mark.asyncio
async def test_client_close_and_reinit(reset_http_client):
    """Test that client can be closed and reinitialized."""
    with mock_http_server(port=9877) as base_url:
        # Initialize and use
        http_client.init_client(max_connections=5)
        client1 = http_client.get_client()
        response = await client1.get(f"{base_url}/api/test")
        assert response.is_success, f"Expected success, got {response.status_code}"

        # Close
        await http_client.close_client()

        # Reinitialize with different settings
        http_client.init_client(max_connections=10)
        client2 = http_client.get_client()

        # Should work with new client
        response = await client2.get(f"{base_url}/api/test")
        assert response.is_success, f"Expected success, got {response.status_code}"


@pytest.mark.asyncio
async def test_unlimited_connections(reset_http_client):
    """Test behavior with no connection limit (max_connections=None)."""

    if not getenv("LOCAL_TESTING"):
        num_requests = 504  # higher number hit fd limit
    else:
        num_requests = 100  # for local dev the limit is lower

    with mock_http_server(port=9878, response_delay=0.02) as base_url:
        http_client.init_client(
            max_connections=None,  # No limit
            max_keepalive_connections=20,
        )
        client = http_client.get_client()

        async def make_request() -> bool:
            response = await client.get(f"{base_url}/api/test")
            return response.is_success

        start_time = time.perf_counter()
        results = await asyncio.gather(*[make_request() for _ in range(num_requests)])
        elapsed = time.perf_counter() - start_time

        assert all(results), "Some requests failed"

        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        print("\n  Unlimited connections test:")
        print(f"  {num_requests} requests completed in {elapsed:.3f}s")
        print(f"  Max concurrent observed: {stats['max_concurrent']}")


@pytest.mark.asyncio
async def test_very_low_connection_limit(reset_http_client):
    """Test with very restrictive connection limit (1 connection)."""
    max_connections = 1
    num_requests = 10
    response_delay = 0.05

    with mock_http_server(port=9879, response_delay=response_delay) as base_url:
        http_client.init_client(
            max_connections=max_connections,
            max_keepalive_connections=1,
        )
        client = http_client.get_client()

        async def make_request(req_id: int) -> dict:
            response = await client.post(
                f"{base_url}/api/test",
                json={"request_id": req_id},
            )
            return response.json()

        start_time = time.perf_counter()
        results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        elapsed = time.perf_counter() - start_time

        assert len(results) == num_requests

        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        # With 1 connection, requests must be serialized
        assert stats["max_concurrent"] == 1, (
            f"Expected max_concurrent=1, got {stats['max_concurrent']}"
        )

        # Should take at least num_requests * response_delay
        min_time = num_requests * response_delay * 0.8
        assert elapsed >= min_time, "Requests completed too fast for single connection"

        print("\n  Single connection test:")
        print(f"  {num_requests} requests serialized in {elapsed:.3f}s")


@pytest.mark.asyncio
async def test_burst_traffic_pattern(reset_http_client):
    """Test handling burst traffic patterns."""
    max_connections = 20
    burst_size = 50
    num_bursts = 3

    with mock_http_server(port=9880, response_delay=0.02) as base_url:
        http_client.init_client(
            max_connections=max_connections,
            max_keepalive_connections=15,
        )
        client = http_client.get_client()

        total_requests = 0
        total_time = 0.0

        for burst in range(num_bursts):
            requests = [
                client.post(f"{base_url}/api/test", json={"burst": burst, "req": i})
                for i in range(burst_size)
            ]

            start = time.perf_counter()
            responses = await asyncio.gather(*requests)
            burst_time = time.perf_counter() - start

            assert all(r.is_success for r in responses)
            total_requests += len(responses)
            total_time += burst_time

            # Brief pause between bursts
            await asyncio.sleep(0.1)

        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()

        print("\n  Burst traffic test:")
        print(f"  {num_bursts} bursts of {burst_size} requests")
        print(f"  Total: {total_requests} requests in {total_time:.3f}s")
        print(f"  Max concurrent: {stats['max_concurrent']}")

        assert stats["max_concurrent"] <= max_connections
