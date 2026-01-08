from asyncio import CancelledError, Queue, Task, create_task
from collections.abc import Awaitable, Callable
from logging import getLogger

from agentica_internal.multiplex_protocol import MultiplexServerMessage, multiplex_to_json

logger = getLogger(__name__)


class WebSocketSender:
    def __init__(self, *, send_bytes: Callable[[bytes], Awaitable[None]]):
        self._send_bytes = send_bytes
        self._queue: "Queue[MultiplexServerMessage]" = Queue()
        self._writer_task: Task[None] | None = None

    async def _writer(self) -> None:
        while True:
            try:
                msg = await self._queue.get()
                logger.debug("Sending: %s", msg)
                await self._send_bytes(multiplex_to_json(msg))
            except CancelledError:
                break

    async def start(self) -> None:
        if self._writer_task is not None:
            return
        self._writer_task = create_task(self._writer(), name="WS.TransportWriter")

    async def stop(self) -> None:
        task = self._writer_task
        if task is None:
            return
        try:
            task.cancel()
        except:
            pass
        self._writer_task = None

    async def enqueue(self, msg: MultiplexServerMessage) -> None:
        await self._queue.put(msg)

    def __del__(self) -> None:
        try:
            if self._writer_task is not None:
                self._writer_task.cancel()
        except:
            pass
