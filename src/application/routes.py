import asyncio
import json
import logging
import os
import sys
import time
import traceback
from base64 import b64decode
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import msgspec.json
from agentica_internal.multiplex_protocol.multiplex_protocol import (
    MultiplexClientInstanceMessage,
    MultiplexClientMessage,
    MultiplexMessage,
    MultiplexServerInstanceMessage,
    MultiplexServerMessage,
)
from agentica_internal.session_manager_messages import CreateAgentRequest
from litestar import Request, Response, WebSocket, delete, get, post, websocket
from litestar.exceptions import HTTPException, WebSocketDisconnect
from litestar.handlers import BaseRouteHandler
from litestar.response import Stream
from opentelemetry.propagate import extract

from agentic.models import ValidationError
from agentic.protocol import MagicProtocol
from agentic.version_policy import (
    VersionStatus,
    check_sdk_version,
    format_unsupported_message,
    format_upgrade_message,
)
from application.http_client import get_client
from application.metrics import (
    active_agents,
    agent_creation_duration_seconds,
    agent_creations_total,
    agent_invocation_duration_seconds,
    agent_invocations_total,
    get_metrics,
    websocket_connections,
)
from application.responses import AnsiHTMLResponse, LogFileResponse, ZipResponse
from server_session_manager import ServerSessionManager

logger = logging.getLogger(__name__)


_json_encoder = msgspec.json.Encoder()


# Real-time trace streaming infrastructure
class TraceStreamManager:
    """Manages real-time trace streaming subscriptions with bounded queues."""

    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self._all_subscribers: list[asyncio.Queue] = []
        self._trace_subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def broadcast_spans(self, spans_data: dict[str, Any]) -> None:
        """Broadcast spans to all relevant subscribers."""
        async with self._lock:
            # Extract trace IDs from the spans
            trace_ids = set()
            for batch in spans_data.get("resourceSpans", []):
                for scope_span in batch.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        if trace_id := span.get("traceId"):
                            trace_ids.add(trace_id)

            # Broadcast to "all traces" subscribers
            dead_queues = []
            for queue in self._all_subscribers:
                try:
                    queue.put_nowait(spans_data)
                except asyncio.QueueFull:
                    # Drop oldest item and try again
                    try:
                        queue.get_nowait()
                        queue.put_nowait(spans_data)
                    except:
                        dead_queues.append(queue)
                except:
                    dead_queues.append(queue)

            # Clean up dead queues
            for queue in dead_queues:
                self._all_subscribers.remove(queue)

            # Broadcast to trace-specific subscribers
            for trace_id in trace_ids:
                if trace_id in self._trace_subscribers:
                    dead_queues = []
                    for queue in self._trace_subscribers[trace_id]:
                        try:
                            queue.put_nowait(spans_data)
                        except asyncio.QueueFull:
                            try:
                                queue.get_nowait()
                                queue.put_nowait(spans_data)
                            except:
                                dead_queues.append(queue)
                        except:
                            dead_queues.append(queue)

                    for queue in dead_queues:
                        self._trace_subscribers[trace_id].remove(queue)

    async def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to all traces."""
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        async with self._lock:
            self._all_subscribers.append(queue)
        return queue

    async def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from all traces."""
        async with self._lock:
            if queue in self._all_subscribers:
                self._all_subscribers.remove(queue)

    async def subscribe_trace(self, trace_id: str) -> asyncio.Queue:
        """Subscribe to a specific trace."""
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        async with self._lock:
            if trace_id not in self._trace_subscribers:
                self._trace_subscribers[trace_id] = []
            self._trace_subscribers[trace_id].append(queue)
        return queue

    async def unsubscribe_trace(self, trace_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from a specific trace."""
        async with self._lock:
            if trace_id in self._trace_subscribers:
                if queue in self._trace_subscribers[trace_id]:
                    self._trace_subscribers[trace_id].remove(queue)
                if not self._trace_subscribers[trace_id]:
                    del self._trace_subscribers[trace_id]


# Global trace stream manager (max 100 items per queue)
trace_stream_manager = TraceStreamManager(max_queue_size=100)


@get("/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok"}


@post("/otel/ingest/v1/traces")
async def otel_ingest_spans(request: Request) -> dict[str, str]:
    """
    Receive spans from OTel Collector for real-time streaming.

    This endpoint receives OTLP trace data (protobuf format, gzip compressed) from
    the collector and broadcasts it to all active SSE subscribers.
    """
    try:
        import gzip

        # Get raw body
        body = await request.body()

        # Decompress if gzipped (collector is configured with compression: gzip)
        content_encoding = request.headers.get("content-encoding", "")
        if "gzip" in content_encoding:
            body = gzip.decompress(body)

        # Parse JSON (collector is configured with encoding: json)
        data = json.loads(body)

        logger.debug(
            f"Received trace data from collector: {len(data.get('resourceSpans', []))} resource spans"
        )

        # Broadcast to SSE subscribers
        await trace_stream_manager.broadcast_spans(data)
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error processing spans: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    metrics_data, content_type = get_metrics()
    return Response(
        content=metrics_data,
        media_type=content_type,
    )


@post("/session/register")
async def session_register(request: Request) -> dict[str, str]:
    """
    Register a new client session.
    Clients should call this before creating agents.
    """
    session_manager: ServerSessionManager = request.app.state.session_manager

    # Extract client session ID from headers
    cid = request.headers.get("X-Client-Session-ID")
    if not cid:
        raise HTTPException(status_code=400, detail="Missing X-Client-Session-ID header")

    session_manager.register_session(cid)
    return {"status": "ok", "message": f"Session {cid} registered"}


def _make_type_filter(type_tag: str | None):
    """Create a filter function that matches message type by tag name."""
    if type_tag is None:
        return None
    return lambda msg: getattr(type(msg).__struct_config__, 'tag', None) == type_tag


@get("/sandbox_logs")
async def sandbox_logs(
    request: Request,
    fmt: Literal['zip', 'html', 'log'] = 'html',
    max_files: int = 16,
    max_chunks: int = 0,
    find: str = '',
    uid: str = '',
) -> Response:
    session_manager: ServerSessionManager = request.app.state.session_manager
    if not max_chunks:
        max_chunks = 1024 if (find or uid) else 32
    if fmt == 'zip':
        root, paths = session_manager.sandbox_log_paths(max_files=max_files, find=find, uid=uid)
        return ZipResponse('sandbox_logs_{ts}.zip', root, paths)
    elif fmt == 'html':
        merged = session_manager.sandbox_logs_merged(
            max_files=max_files, max_chunks=max_chunks, find=find, uid=uid
        )
        return AnsiHTMLResponse(merged)
    elif fmt == 'log':
        merged = session_manager.sandbox_logs_merged(
            max_files=max_files, max_chunks=max_chunks, find=find, uid=uid
        )
        return LogFileResponse('sandbox_logs_merged_{ts}.log', merged)
    else:
        return Response('bad fmt')


@get("/logs")
async def logs(request: Request, type: str | None = None) -> list[dict[str, Any]]:
    session_manager: ServerSessionManager = request.app.state.session_manager
    return list(session_manager.get_json_all_logs(_make_type_filter(type)))


@get("/logs/{uid:str}")
async def logs_by_uid(request: Request, uid: str, type: str | None = None) -> list[dict[str, Any]]:
    session_manager: ServerSessionManager = request.app.state.session_manager
    return list(session_manager.get_json_logs_by_uid(uid, _make_type_filter(type)))


@get("/logs/{uid:str}/{iid:str}")
async def logs_by_uid_and_iid(
    request: Request, uid: str, iid: str, type: str | None = None
) -> list[dict[str, Any]]:
    session_manager: ServerSessionManager = request.app.state.session_manager
    return list(session_manager.get_json_logs(uid, iid, _make_type_filter(type)))


# The following two streams are infinite, its up to the client
# (or us with a timeout) to close the connection when finished.


@get("/echo/{uid:str}/{iid:str}")
async def echo(request: Request, uid: str, iid: str) -> Stream:
    session_manager: ServerSessionManager = request.app.state.session_manager

    stream = session_manager.listen_json(uid, iid)

    async def ndjson() -> AsyncGenerator[bytes, None]:
        async for log in stream:
            yield _json_encoder.encode(log) + b'\n'

    return Stream(
        content=ndjson(),
        media_type="text/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@get("/echo/{uid:str}")
async def echo_all(request: Request, uid: str) -> Stream:
    session_manager: ServerSessionManager = request.app.state.session_manager
    stream = session_manager.listen_all_json(uid)

    async def ndjson() -> AsyncGenerator[bytes, None]:
        async for log in stream:
            yield _json_encoder.encode(log) + b'\n'

    return Stream(
        content=ndjson(),
        media_type="text/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@get("/echo")
async def echo_global(request: Request, uid: str) -> Stream:
    session_manager: ServerSessionManager = request.app.state.session_manager
    stream = session_manager.listen_global_json()

    async def ndjson() -> AsyncGenerator[bytes, None]:
        async for log in stream:
            yield _json_encoder.encode(log) + b'\n'

    return Stream(
        content=ndjson(),
        media_type="text/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@post("/agent/create")
async def agent_create(request: Request, data: CreateAgentRequest) -> str | Response[str]:
    """
    Posts to this route will return 400 if the warp payload is invalid.
    Otherwise the server claims that a agent has been created and is ready to be
    invoked via websocket. We additionally handle protocol version policy checks here.
    """
    start_time = time.time()
    status = "success"
    model = data.model

    protocol = MagicProtocol.parse(data.protocol)
    version_status = check_sdk_version(protocol.sdk, protocol.version)

    if version_status == VersionStatus.UNSUPPORTED:
        raise HTTPException(
            status_code=426,
            detail=format_unsupported_message(protocol.sdk, protocol.version),
        )

    session_manager: ServerSessionManager = request.app.state.session_manager
    try:
        if isinstance(data.warp_globals_payload, str):
            data.warp_globals_payload = b64decode(data.warp_globals_payload, validate=True)
        elif not isinstance(data.warp_globals_payload, (bytes, bytearray)):
            raise ValueError(
                f"Unexpected warp_payload type: {type(data.warp_globals_payload).__name__}"
            )
    except Exception as e:
        status = "error"
        agent_creations_total.labels(model=model, status=status).inc()
        logger.error(
            f"Agent creation failed: invalid warp payload", extra={"model": model, "error": str(e)}
        )
        raise HTTPException(status_code=400, detail=f"Invalid warp payload: {e}")

    # Extract client session ID from headers
    cid = request.headers.get("X-Client-Session-ID")
    if not cid:
        raise HTTPException(status_code=400, detail="Missing X-Client-Session-ID header")

    # Get session manager ID from environment
    session_manager_id = os.getenv("ORGANIZATION_ID", "LOCAL_SESSION_MANAGER")

    try:
        uid = await session_manager.create_agent(data, cid, session_manager_id)
        active_agents.inc()

        logger.info(
            f"Agent creation request successful",
            extra={
                "agent_uid": uid,
                "model": model,
                "streaming": data.streaming,
            },
        )
    except ValidationError as e:
        status = "error"
        agent_creations_total.labels(model=model, status=status).inc()
        stack_trace = str(e) + "\n" + traceback.format_exc()
        logger.error(
            f"Agent creation failed: validation error", extra={"model": model, "error": str(e)}
        )
        raise HTTPException(status_code=e.http_status_code, detail=stack_trace)
    except Exception as e:
        status = "error"
        agent_creations_total.labels(model=model, status=status).inc()
        stack_trace = str(e) + "\n" + traceback.format_exc()
        short = repr(e)
        logger.error(
            f"Agent creation failed: internal error {short}",
            extra={"model": model, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=stack_trace)
    finally:
        duration = time.time() - start_time
        agent_creations_total.labels(model=model, status=status).inc()
        agent_creation_duration_seconds.labels(model=model).observe(duration)

    # Build JSON response with uid and session_manager_id
    response_data = {
        "uid": uid,
        "session_manager_id": session_manager_id,
    }

    if version_status == VersionStatus.DEPRECATED:
        return Response(
            content=json.dumps(response_data),
            status_code=201,
            media_type="application/json",
            headers={
                "X-SDK-Warning": "deprecated",
                "X-SDK-Upgrade-Message": format_upgrade_message(protocol.sdk, protocol.version),
            },
        )

    return Response(
        content=json.dumps(response_data),
        status_code=201,
        media_type="application/json",
    )


@websocket(path="/socket")
async def setup_socket_and_loop(socket: WebSocket) -> None:
    """
    WebSocket endpoint to setup the socket for multiplexed agent invocations.

    This endpoint accepts a WebSocket connection and delegates to the session manager's
    loop_on_socket, which starts an event loop and runs a Multiplexer to handle multiple agent
    invocations over the same socket.

    The handler will not close the websocket from this end unless there is an error.
    A handler will never raise an exception. In the worst possible case it will log
    exceptions as health issues and close the websocket. In this way the server
    remains functional and the clients can always interpret an unusual websocket closure
    as an unknown error.

    This handler also extracts distributed trace context from WebSocket headers
    to enable end-to-end tracing from SDK through the session manager.
    """
    start_time = time.time()
    status = "success"

    session_manager: ServerSessionManager = socket.app.state.session_manager

    # Track WebSocket connection
    websocket_connections.inc()

    # Extract trace context from WebSocket headers for distributed tracing
    headers_dict = dict(socket.headers)
    parent_context = extract(headers_dict)

    cid = headers_dict.get("x-client-session-id")
    if not cid:
        raise HTTPException(status_code=400, detail="Missing X-Client-Session-ID header")

    try:
        await session_manager.loop_on_socket(socket, cid=cid, parent_context=parent_context)
    except WebSocketDisconnect:
        pass
    except Exception:
        status = "error"
        raise
    finally:
        # Track metrics
        duration = time.time() - start_time
        agent_invocations_total.labels(status=status).inc()
        agent_invocation_duration_seconds.observe(duration)
        websocket_connections.dec()


@delete("/agent/destroy/{uid:str}")
async def agent_destroy(request: Request, uid: str) -> None:
    session_manager: ServerSessionManager = request.app.state.session_manager
    try:
        await session_manager.destroy_agent(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dataclass
class MultiplexTypes:
    MultiplexMessage: MultiplexMessage
    MultiplexClientMessage: MultiplexClientMessage
    MultiplexClientInstanceMessage: MultiplexClientInstanceMessage
    MultiplexServerMessage: MultiplexServerMessage
    MultiplexServerInstanceMessage: MultiplexServerInstanceMessage


@get("/schema/multiplex-messages")
async def multiplex_message_schema_docs() -> MultiplexTypes:
    """
    This endpoint exists solely to force inclusion of MultiplexMessage schemas in OpenAPI.
    It's not meant to be called - it's just for documentation purposes.
    The WebSocket endpoint at /socket uses these message types.
    """
    # This will never actually be called, but forces schema generation
    raise NotImplementedError("This endpoint is for schema documentation only")


# Tempo API proxy endpoints for trace queries
# Default to localhost for local dev, can override with TEMPO_URL env var
TEMPO_URL = os.getenv("TEMPO_URL", "http://localhost:3200")

# Max traces to fetch with full details (prevents timeout)
MAX_DETAILED_TRACES = 250


async def search_traces(
    start: int | None = None,
    end: int | None = None,
    limit: int = 100,
    service: str | None = None,
) -> dict[str, Any]:
    """Internal function to search traces from Tempo."""
    # Default to last hour if not specified (Unix seconds, not nanoseconds)
    now_sec = int(time.time())
    one_hour_sec = 3600

    if start is None:
        start = now_sec - one_hour_sec
    if end is None:
        end = now_sec

    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "limit": limit,
    }

    # Add service filter if provided
    if service:
        params["tags"] = f'service.name="{service}"'

    client = get_client()
    response = await client.get(
        f"{TEMPO_URL}/api/search",
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


@get("/traces")
async def traces(
    start: int | None = None,
    end: int | None = None,
    limit: int = 100,
    service: str | None = None,
) -> dict[str, Any]:
    """
    Search traces from Tempo.

    Query Parameters:
        start: Start time in Unix seconds (default: 1 hour ago)
        end: End time in Unix seconds (default: now)
        limit: Maximum number of traces to return (default: 100)
        service: Filter by service name (optional)

    Example:
        GET /traces?start=1699000000&end=1699100000&limit=50
        GET /traces  # Returns traces from last hour
    """
    try:
        return await search_traces(start=start, end=end, limit=limit, service=service)
    except httpx.HTTPError as e:
        logger.error(f"Failed to query Tempo: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to query trace backend: {str(e)}")


@get("/traces/{trace_id:str}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """
    Get a specific trace by ID from Tempo.

    Args:
        trace_id: The trace ID (32-character hex string)

    Example:
        GET /traces/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
    """
    try:
        client = get_client()
        response = await client.get(f"{TEMPO_URL}/api/traces/{trace_id}", timeout=30.0)
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        raise HTTPException(status_code=502, detail=f"Failed to fetch trace: {str(e)}")
    except httpx.HTTPError as e:
        logger.error(f"Failed to query Tempo: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to query trace backend: {str(e)}")


@get("/traces/detailed")
async def get_detailed_traces(
    start: int | None = None,
    end: int | None = None,
    limit: int = 10,
    service: str | None = None,
) -> dict[str, Any]:
    """
    Search traces and fetch full details for each (including all spans and events).

    This is a convenience endpoint that combines search + fetch.
    Note: This can be slow if limit is large, as it fetches each trace individually.

    Query Parameters:
        start: Start time in Unix seconds (default: 1 hour ago)
        end: End time in Unix seconds (default: now)
        limit: Maximum number of traces to return (default: 10, max: {MAX_DETAILED_TRACES})
        service: Filter by service name (optional)

    Example:
        GET /traces/detailed?limit=5&service=session-manager
    """
    try:
        # Enforce max limit to prevent timeout
        if limit > MAX_DETAILED_TRACES:
            raise HTTPException(
                status_code=400,
                detail=f"Limit cannot exceed {MAX_DETAILED_TRACES}. Use /traces endpoint for larger queries.",
            )

        # First, search for traces
        search_result = await search_traces(start=start, end=end, limit=limit, service=service)

        if not search_result.get("traces"):
            return {"traces": [], "count": 0}

        # Fetch full details for each trace
        client = get_client()
        detailed_traces = []
        for trace_summary in search_result["traces"]:
            trace_id = trace_summary.get("traceID")
            if not trace_id:
                continue

            try:
                response = await client.get(f"{TEMPO_URL}/api/traces/{trace_id}", timeout=60.0)
                response.raise_for_status()
                trace_data = response.json()

                # Add search metadata to the full trace
                trace_data["metadata"] = trace_summary
                detailed_traces.append(trace_data)
            except Exception as e:
                logger.warning(f"Failed to fetch trace {trace_id}: {e}")
                continue

        return {
            "traces": detailed_traces,
            "count": len(detailed_traces),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch detailed traces: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch detailed traces: {str(e)}")


@get("/traces/stream")
async def stream_traces(service: str | None = None) -> Stream:
    """
    Stream spans as they arrive from OTel Collector in real-time (Server-Sent Events).

    This is TRUE streaming - spans are pushed from the collector as they're generated.
    No polling, immediate delivery.

    Query Parameters:
        service: Filter by service name (optional)

    Example:
        GET /traces/stream?service=session-manager

    Response Format (SSE):
        event: spans
        data: {"resourceSpans": [...], "timestamp": 1234567890}
    """

    async def trace_stream_generator() -> AsyncGenerator[str, None]:
        queue = await trace_stream_manager.subscribe_all()

        try:
            # Send initial connection message
            yield f"event: connected\ndata: {json.dumps({'message': 'streaming started'})}\n\n"

            while True:
                # Wait for spans from the collector (blocking, no polling!)
                spans_data = await queue.get()

                # Filter by service if specified
                if service:
                    filtered = {"resourceSpans": []}
                    for batch in spans_data.get("resourceSpans", []):
                        resource_attrs = {
                            attr.get("key"): attr.get("value", {}).get("stringValue")
                            for attr in batch.get("resource", {}).get("attributes", [])
                        }
                        if resource_attrs.get("service.name") == service:
                            filtered["resourceSpans"].append(batch)

                    if not filtered["resourceSpans"]:
                        continue  # Skip if no matching service
                    spans_data = filtered

                # Send spans as SSE event
                event_data = {
                    "resourceSpans": spans_data.get("resourceSpans", []),
                    "timestamp": int(time.time()),
                }
                yield f"event: spans\ndata: {json.dumps(event_data)}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            await trace_stream_manager.unsubscribe_all(queue)

    return Stream(
        trace_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@get("/traces/stream/{trace_id:str}")
async def stream_trace_updates(trace_id: str) -> Stream:
    """
    Stream spans for a specific trace as they arrive in real-time (Server-Sent Events).

    This is TRUE streaming - spans are pushed from the collector as they're generated.
    Only spans matching the given trace_id are sent.

    Example:
        GET /traces/stream/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6

    Response Format (SSE):
        event: span
        data: {"spanId": "...", "name": "gen_ai.chat", "traceId": "...", ...}

        event: connected
        data: {"message": "streaming started for trace {trace_id}"}
    """

    async def trace_update_generator() -> AsyncGenerator[str, None]:
        queue = await trace_stream_manager.subscribe_trace(trace_id)

        try:
            # Send initial connection message
            yield f"event: connected\ndata: {json.dumps({'trace_id': trace_id, 'message': 'streaming started'})}\n\n"

            while True:
                # Wait for spans from the collector (blocking, no polling!)
                spans_data = await queue.get()

                # Extract and send spans matching this trace_id
                for batch in spans_data.get("resourceSpans", []):
                    for scope_span in batch.get("scopeSpans", []):
                        for span in scope_span.get("spans", []):
                            if span.get("traceId") == trace_id:
                                # Send individual span as SSE event
                                yield f"event: span\ndata: {json.dumps(span)}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            await trace_stream_manager.unsubscribe_trace(trace_id, queue)

    return Stream(
        trace_update_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def get_routes() -> list[BaseRouteHandler]:
    current_module = sys.modules[__name__]
    return [obj for obj in current_module.__dict__.values() if isinstance(obj, BaseRouteHandler)]
