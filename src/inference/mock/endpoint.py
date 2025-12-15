import atexit
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from textwrap import dedent
from typing import Any, Callable

import uvicorn
from litestar import Litestar, Request, Response, post
from litestar.exceptions import HTTPException

from .queue import Queue

logger = getLogger(__name__)


@dataclass
class Reply:
    """Represents an agent's reply"""

    message: str
    tool_name_arguments: tuple[str, str, str] | None = None
    error_code: int | None = None

    def __post_init__(self):
        self.message = dedent(self.message).strip()

    def to_completion(self) -> dict[str, Any]:
        if self.tool_name_arguments is not None:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.message,
                            "tool_calls": [
                                {
                                    "id": self.tool_name_arguments[2],
                                    "type": "function",
                                    "function": {
                                        "name": self.tool_name_arguments[0],
                                        "arguments": self.tool_name_arguments[1],
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
                "usage": {
                    "prompt_tokens": len(self.message),
                    "completion_tokens": len(self.message),
                    "total_tokens": len(self.message),
                },
            }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.message},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": len(self.message),
                "completion_tokens": len(self.message),
                "total_tokens": len(self.message),
            },
        }


class CompletionsEndpoint:
    responses: dict[str, Queue[Reply]]
    port: int
    endpoint_url: str

    _app: Litestar
    _config: uvicorn.Config
    _server: uvicorn.Server
    _thread: threading.Thread | None

    _hooks: list[Callable[[Request], None]]

    def __init__(self, port: int = 8000):
        self.responses = defaultdict(Queue)
        self.port = port
        self.endpoint_url = f"http://localhost:{port}/v1/chat/completions"

        self._app = self._init_server()
        self._config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(self._config)
        self._thread = None
        self._hooks = []
        atexit.register(self.shutdown)

    def hook(self, fn: Callable[[Request], None]):
        """Set a hook to run as each request comes in."""
        self._hooks.append(fn)

    def _init_server(self) -> Litestar:
        @post("/v1/chat/completions")
        async def create_chat_completion(request: Request) -> Response:
            queue = self.responses[request.url.path]
            # allow hooks to run beforehand
            for hook in self._hooks:
                hook(request)
            # for now, we don't really care about the history or config in the request
            payload = await request.json()
            logger.debug(f"Received request: {payload}")
            response = queue.get()  # blocks until a response is available
            logger.debug(f"Sending response: {response}")
            if response.error_code is not None:
                raise HTTPException(
                    status_code=response.error_code, detail=response.to_completion()
                )
            return Response(status_code=200, content=response.to_completion())

        return Litestar(
            route_handlers=[
                create_chat_completion,
            ],
            debug=True,
        )

    def respond(self, path: str, reply: Reply):
        self.responses[path].put(reply, block=False)

    def shutdown(self, path: str | None = None):
        if path is None:
            # shutdown all queues
            for queue in self.responses.values():
                queue.shutdown()
        else:
            queue = self.responses[path]
            queue.shutdown()

    def reset(self):
        for queue in self.responses.values():
            queue.queue.clear()

    def start_threaded(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def start_sync(self):
        self._server.run()

    def stop(self, timeout: float = 5.0):
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=timeout)
