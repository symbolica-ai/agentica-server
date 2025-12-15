"""Prometheus metrics for Session Manager."""

import time
from functools import wraps

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Agent creation metrics
agent_creations_total = Counter(
    'session_manager_agent_creations_total', 'Total number of agent creations', ['model', 'status']
)

agent_creation_duration_seconds = Histogram(
    'session_manager_agent_creation_duration_seconds',
    'Time spent creating agents',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Agent invocation metrics
agent_invocations_total = Counter(
    'session_manager_agent_invocations_total', 'Total number of agent invocations', ['status']
)

agent_invocation_duration_seconds = Histogram(
    'session_manager_agent_invocation_duration_seconds',
    'Time spent on agent invocations',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Active agents gauge
active_agents = Gauge('session_manager_active_agents', 'Number of currently active agents')

# WebSocket connections
websocket_connections = Gauge(
    'session_manager_websocket_connections', 'Number of active WebSocket connections'
)

# HTTP request metrics
http_requests_total = Counter(
    'session_manager_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'session_manager_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


def track_agent_creation(model: str):
    """Decorator to track agent creation metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                active_agents.inc()
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                agent_creations_total.labels(model=model, status=status).inc()
                agent_creation_duration_seconds.labels(model=model).observe(duration)

        return wrapper

    return decorator


def track_agent_invocation():
    """Decorator to track agent invocation metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                agent_invocations_total.labels(status=status).inc()
                agent_invocation_duration_seconds.observe(duration)

        return wrapper

    return decorator


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics in the exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
