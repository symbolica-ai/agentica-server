# Adapted from https://raw.githubusercontent.com/bytecodealliance/componentize-py/refs/heads/main/bundled/poll_loop.py

import asyncio
import os
import socket
import subprocess
import time
from collections.abc import Callable
from contextvars import Context
from heapq import heappop, heappush
from typing import IO, Any

from agentica_internal.core.futures import new_hookable_future

try:
    os.listdir()
    IN_WASM = False
except:
    IN_WASM = True
    os.environ["AGENTICA_LOG_TAGS"] = "REPL"


class PollLoop(asyncio.AbstractEventLoop):
    def __init__(self) -> None:
        self.running: bool = False
        self.handles: list[asyncio.Handle] = []
        self.scheduled: list[tuple[float, asyncio.TimerHandle]] = []  # Min-heap of (when, handle)
        self.exception: BaseException | None = None
        self._debug: bool = False
        self._task_factory: Callable[..., asyncio.Future[Any]] | None = None
        self._exception_handler: (
            Callable[[asyncio.AbstractEventLoop, dict[str, Any]], Any] | None
        ) = None

    def get_debug(self) -> bool:
        return self._debug

    def run_until_complete(self, future):
        self.running = True
        asyncio.events._set_running_loop(self)
        try:
            if not (
                asyncio.isfuture(future)
                or asyncio.iscoroutine(future)
                or hasattr(future, "__await__")
            ):
                raise TypeError("An asyncio.Future, a coroutine or an awaitable is required")
            future = asyncio.ensure_future(future)
            while self.running and not future.done():
                # Process immediate handles
                handles = self.handles
                self.handles = []
                for handle in handles:
                    if not handle._cancelled:
                        handle._run()

                # Process due scheduled callbacks
                now = self.time()
                while self.scheduled and self.scheduled[0][0] <= now:
                    when, handle = heappop(self.scheduled)
                    if not handle._cancelled:
                        handle._run()

                if self.exception is not None:
                    raise self.exception

                # If there are scheduled callbacks but nothing immediate,
                # sleep until the next scheduled callback to avoid busy-waiting
                if not self.handles and self.scheduled and not future.done():
                    next_when = self.scheduled[0][0]
                    sleep_time = max(0, next_when - self.time())
                    if sleep_time > 0:
                        # Sleep for a short time to avoid busy-waiting
                        # Cap at 0.01 seconds (10ms) to remain responsive
                        time.sleep(min(sleep_time, 0.01))

            return future.result()
        finally:
            self.stop()

    def is_running(self) -> bool:
        return self.running

    def is_closed(self) -> bool:
        return not self.running

    def stop(self) -> None:
        self.running = False
        try:
            if asyncio.events._get_running_loop() is self:
                asyncio.events._set_running_loop(None)
        except Exception:
            pass

    def close(self) -> None:
        self.running = False
        try:
            if asyncio.events._get_running_loop() is self:
                asyncio.events._set_running_loop(None)
        except Exception:
            pass

    async def shutdown_asyncgens(self) -> None:  # type: ignore
        return None

    def call_exception_handler(self, context: dict[str, Any]) -> None:
        if self._exception_handler is not None:
            self._exception_handler(self, context)
        else:
            self.default_exception_handler(context)
        # Store exception for propagation in run_until_complete
        self.exception = context.get("exception", None)

    def call_soon(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Context | None = None,
    ) -> asyncio.Handle:
        handle = asyncio.Handle(callback, args, self, context)
        self.handles.append(handle)
        return handle

    def create_task(
        self,
        coro: Any,
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task[Any]:
        if self._task_factory is not None:
            task = self._task_factory(self, coro, context=context)
            # Task factory returns Future, but we know it should be a Task
            return task  # type: ignore[return-value]
        return asyncio.Task(coro, loop=self, name=name, context=context)

    def create_future(self) -> asyncio.Future[Any]:
        return new_hookable_future(self)

    # The remaining methods should be irrelevant for our purposes and thus unimplemented

    def run_in_executor(self, executor, func, *args):
        del executor, func, args
        raise NotImplementedError

    def run_forever(self) -> None:
        """Run the event loop until stop() is called."""
        self.running = True
        asyncio.events._set_running_loop(self)
        try:
            while self.running:
                # Process immediate handles
                handles = self.handles
                self.handles = []
                for handle in handles:
                    if not handle._cancelled:
                        handle._run()

                # Process due scheduled callbacks
                now = self.time()
                while self.scheduled and self.scheduled[0][0] <= now:
                    when, handle = heappop(self.scheduled)
                    if not handle._cancelled:
                        handle._run()

                if self.exception is not None:
                    raise self.exception

                # If there are scheduled callbacks but nothing immediate,
                # sleep until the next scheduled callback to avoid busy-waiting
                if not self.handles and self.scheduled:
                    next_when = self.scheduled[0][0]
                    sleep_time = max(0, next_when - self.time())
                    if sleep_time > 0:
                        # Sleep for a short time to avoid busy-waiting
                        # Cap at 0.01 seconds (10ms) to remain responsive
                        time.sleep(min(sleep_time, 0.01))
                elif not self.handles:
                    # No work at all, sleep briefly to avoid 100% CPU
                    time.sleep(0.001)
        finally:
            asyncio.events._set_running_loop(None)

    async def shutdown_default_executor(self, executor=None):
        del executor
        return None

    def _timer_handle_cancelled(self, handle: asyncio.TimerHandle) -> None:
        # When a timer handle is cancelled, we don't need to remove it from the heap
        # The handle's _cancelled flag will be checked when it's popped
        # This is more efficient than searching and removing from the heap
        pass

    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Context | None = None,
    ) -> asyncio.TimerHandle:
        when = self.time() + delay
        return self.call_at(when, callback, *args, context=context)

    def call_at(
        self,
        when: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Context | None = None,
    ) -> asyncio.TimerHandle:
        handle = asyncio.TimerHandle(when, callback, args, self, context)
        heappush(self.scheduled, (when, handle))
        return handle

    def time(self) -> float:
        return time.monotonic()

    def call_soon_threadsafe(self, callback, *args, context=None):
        del callback, args, context
        raise NotImplementedError

    def set_default_executor(self, executor):
        del executor
        raise NotImplementedError

    async def getaddrinfo(self, host, port, *, family=0, type=0, proto=0, flags=0):
        del host, port, family, type, proto, flags
        raise NotImplementedError

    async def getnameinfo(self, sockaddr, flags=0):
        del sockaddr, flags
        raise NotImplementedError

    async def create_connection(
        self,
        protocol_factory,
        host=None,
        port=None,
        *,
        ssl=None,
        family=0,
        proto=0,
        flags=0,
        sock=None,
        local_addr=None,
        server_hostname=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
        happy_eyeballs_delay=None,
        interleave=None,
    ):
        del (
            protocol_factory,
            host,
            port,
            ssl,
            family,
            proto,
            flags,
            sock,
            local_addr,
            server_hostname,
            ssl_handshake_timeout,
            ssl_shutdown_timeout,
            happy_eyeballs_delay,
            interleave,
        )
        raise NotImplementedError

    async def create_server(
        self,
        protocol_factory,
        host=None,
        port=None,
        *,
        keep_alive: bool | None = None,
        family: int = socket.AF_UNSPEC,
        flags: int = socket.AI_PASSIVE,
        sock=None,
        backlog=100,
        ssl=None,
        reuse_address=None,
        reuse_port=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
        start_serving=True,
    ):
        del (
            protocol_factory,
            host,
            port,
            keep_alive,
            family,
            flags,
            sock,
            backlog,
            ssl,
            reuse_address,
            reuse_port,
            ssl_handshake_timeout,
            ssl_shutdown_timeout,
            start_serving,
        )
        raise NotImplementedError

    async def sendfile(self, transport, file, offset=0, count=None, *, fallback=True):
        del transport, file, offset, count, fallback
        raise NotImplementedError

    async def start_tls(
        self,
        transport,
        protocol,
        sslcontext,
        *,
        server_side=False,
        server_hostname=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
    ):
        del (
            transport,
            protocol,
            sslcontext,
            server_side,
            server_hostname,
            ssl_handshake_timeout,
            ssl_shutdown_timeout,
        )
        raise NotImplementedError

    async def create_unix_connection(
        self,
        protocol_factory,
        path=None,
        *,
        ssl=None,
        sock=None,
        server_hostname=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
    ):
        del (
            protocol_factory,
            path,
            ssl,
            sock,
            server_hostname,
            ssl_handshake_timeout,
            ssl_shutdown_timeout,
        )
        raise NotImplementedError

    async def create_unix_server(
        self,
        protocol_factory,
        path=None,
        *,
        sock=None,
        backlog=100,
        ssl=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
        start_serving=True,
    ):
        del (
            protocol_factory,
            path,
            sock,
            backlog,
            ssl,
            ssl_handshake_timeout,
            ssl_shutdown_timeout,
            start_serving,
        )
        raise NotImplementedError

    async def connect_accepted_socket(
        self,
        protocol_factory,
        sock,
        *,
        ssl=None,
        ssl_handshake_timeout=None,
        ssl_shutdown_timeout=None,
    ):
        del protocol_factory, sock, ssl, ssl_handshake_timeout, ssl_shutdown_timeout
        raise NotImplementedError

    async def create_datagram_endpoint(
        self,
        protocol_factory,
        local_addr=None,
        remote_addr=None,
        *,
        family=0,
        proto=0,
        flags=0,
        reuse_address=None,
        reuse_port=None,
        allow_broadcast=None,
        sock=None,
    ):
        del (
            protocol_factory,
            local_addr,
            remote_addr,
            family,
            proto,
            flags,
            reuse_address,
            reuse_port,
            allow_broadcast,
            sock,
        )
        raise NotImplementedError

    async def connect_read_pipe(self, protocol_factory, pipe):
        del protocol_factory, pipe
        raise NotImplementedError

    async def connect_write_pipe(self, protocol_factory, pipe):
        del protocol_factory, pipe
        raise NotImplementedError

    async def subprocess_shell(
        self,
        protocol_factory,
        cmd,
        *,
        stdin: int | IO[Any] | None = subprocess.PIPE,
        stdout: int | IO[Any] | None = subprocess.PIPE,
        stderr: int | IO[Any] | None = subprocess.PIPE,
        **kwargs,
    ):
        del protocol_factory, cmd, stdin, stdout, stderr, kwargs
        raise NotImplementedError

    async def subprocess_exec(
        self,
        protocol_factory,
        *args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    ):
        del protocol_factory, args, stdin, stdout, stderr, kwargs
        raise NotImplementedError

    def add_reader(self, fd, callback, *args):
        del fd, callback, args
        raise NotImplementedError

    def remove_reader(self, fd):
        del fd
        raise NotImplementedError

    def add_writer(self, fd, callback, *args):
        del fd, callback, args
        raise NotImplementedError

    def remove_writer(self, fd):
        del fd
        raise NotImplementedError

    async def sock_recv(self, sock, nbytes):
        del sock, nbytes
        raise NotImplementedError

    async def sock_recv_into(self, sock, buf):
        del sock, buf
        raise NotImplementedError

    async def sock_recvfrom(self, sock, bufsize):
        del sock, bufsize
        raise NotImplementedError

    async def sock_recvfrom_into(self, sock, buf, nbytes=0):
        del sock, buf, nbytes
        raise NotImplementedError

    async def sock_sendall(self, sock, data):
        del sock, data
        raise NotImplementedError

    async def sock_sendto(self, sock, data, address):
        del sock, data, address
        raise NotImplementedError

    async def sock_connect(self, sock, address):
        del sock, address
        raise NotImplementedError

    async def sock_accept(self, sock):
        del sock
        raise NotImplementedError

    async def sock_sendfile(self, sock, file, offset=0, count=None, *, fallback=None):
        del sock, file, offset, count, fallback
        raise NotImplementedError

    def add_signal_handler(self, sig, callback, *args):
        del sig, callback, args
        raise NotImplementedError

    def remove_signal_handler(self, sig):
        del sig
        raise NotImplementedError

    def set_task_factory(self, factory: Callable[..., asyncio.Future[Any]] | None) -> None:
        self._task_factory = factory

    def get_task_factory(self) -> Callable[..., asyncio.Future[Any]] | None:
        return self._task_factory

    def get_exception_handler(
        self,
    ) -> Callable[[asyncio.AbstractEventLoop, dict[str, Any]], Any] | None:
        return self._exception_handler

    def set_exception_handler(
        self, handler: Callable[[asyncio.AbstractEventLoop, dict[str, Any]], Any] | None
    ) -> None:
        self._exception_handler = handler

    def default_exception_handler(self, context: dict[str, Any]) -> None:
        """Default exception handler that prints to stderr."""
        import sys
        import traceback

        message = context.get('message')
        if not message:
            message = 'Unhandled exception in event loop'

        exception = context.get('exception')
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        else:
            exc_info = None

        print(message, file=sys.stderr)
        if exc_info:
            traceback.print_exception(*exc_info, file=sys.stderr)

        # Print other context information
        for key, value in context.items():
            if key not in ('message', 'exception'):
                print(f'{key}: {value}', file=sys.stderr)

    def set_debug(self, enabled: bool) -> None:
        self._debug = enabled


class PollLoopPolicy(asyncio.AbstractEventLoopPolicy):
    """Event loop policy that creates and manages a single-threaded PollLoop.

    This ensures asyncio.run() and asyncio.Runner create PollLoop instances
    instead of the default selector loop (which is not permitted in WASI).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = self.new_event_loop()
        return self._loop

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        # since WASM has no threads, there can only be one loop
        if self._loop is None:
            self._loop = PollLoop()
        else:
            raise RuntimeError("there is already an event loop")
        return self._loop

    # Note: get_child_watcher() and set_child_watcher() are intentionally not
    # implemented. They were deprecated in Python 3.12 and removed in 3.14.
    # Child processes are not supported in WASI environments anyway.


# Install the policy at import time only when running under componentized WASI
if IN_WASM:
    asyncio.set_event_loop_policy(PollLoopPolicy())
