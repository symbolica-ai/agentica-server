from __future__ import annotations

import os
from typing import Any

from litestar import Litestar, delete, get, post
from litestar.connection import Request
from litestar.logging import LoggingConfig
from litestar.response import Response
from litestar.static_files import StaticFilesConfig

from .store import STORE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Known types based on session_manager_messages.py and extras
KNOWN_TYPES: set[str] = {
    # Session Manager messages
    "sm_create_agentic_function",
    "sm_create_agent",
    "sm_destroy_agentic_function",
    "sm_invocation_enter",
    "sm_invocation_exit",
    "sm_invocation_error",
    "sm_health",
    "sm_inference_error",
    "sm_inference_request",
    "sm_inference_response",
    "sm_monad",
    # Generic
    "action",
    "end",
    "return",
    "exec",
    "exec_result",
    "delta",
    "stream_chunk",
}


@get("/")
async def index() -> Response:
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return Response(html, media_type="text/html")


@post("/logs")
async def ingest_logs(request: Request) -> dict[str, str]:
    data = await request.json()
    # Accept either a single message or a list of messages
    if isinstance(data, list):
        for msg in data:
            if isinstance(msg, dict):
                await STORE.add(msg)
    elif isinstance(data, dict):
        await STORE.add(data)
    return {"status": "ok"}


@delete("/logs")
async def clear_logs() -> None:
    await STORE.clear()
    # Return 204 No Content (default for DELETE) with no body
    return None


@get("/version")
async def get_version() -> dict[str, int]:
    return {"version": await STORE.version()}


@get("/types")
async def get_types() -> list[str]:
    discovered = set(await STORE.types())
    merged = sorted(discovered.union(KNOWN_TYPES))
    return merged


@get("/uids")
async def get_uids() -> list[str]:
    return await STORE.uids()


@get("/iids")
async def get_iids(uid: str) -> list[str]:
    return await STORE.iids(uid)


@get("/messages")
async def get_messages(
    type: str | None = None,
    uid: str | None = None,
    iid: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return await STORE.query(
        msg_type=type,
        uid=uid,
        iid=iid,
        include=include,
        exclude=exclude,
        limit=limit,
    )


logging_config = LoggingConfig(
    root={"level": "DEBUG", "handlers": ["console"]},
    loggers={
        __name__: {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "uvicorn": {"level": "DEBUG"},
        "uvicorn.error": {"level": "DEBUG"},
        "uvicorn.access": {"level": "DEBUG"},
        "litestar": {"level": "DEBUG"},
    },
    formatters={"standard": {"format": "%(asctime)s - %(levelname)s - %(name)s: %(message)s"}},
)


def create_app() -> Litestar:
    return Litestar(
        route_handlers=[
            index,
            ingest_logs,
            clear_logs,
            get_version,
            get_types,
            get_uids,
            get_iids,
            get_messages,
        ],
        static_files_config=[StaticFilesConfig(directories=[STATIC_DIR], path="/static")],
        logging_config=logging_config,
        debug=True,
    )
