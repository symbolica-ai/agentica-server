import asyncio
import concurrent.futures
import importlib.util
import sys
import threading
import types
from asyncio import AbstractEventLoopPolicy
from pathlib import Path
from typing import Protocol

from agentica_internal.core import print as P
from agentica_internal.core.log import LogBase, set_log_tags
from agentica_internal.warpc.exceptions import WarpShutdown
from agentica_internal.warpc.worlds.interface import *

__all__ = [
    'PyWasmRunner',
]


class ExecEnv(Protocol):
    def __init__(self): ...
    def init_exec_env(self, id_name: str, log_tags: str | None) -> None: ...
    def get_event_loop(self) -> asyncio.AbstractEventLoop: ...
    def run_msg_loop(self): ...

    loop: asyncio.AbstractEventLoop


class PyWasmRunner(LogBase, AbstractEventLoopPolicy):
    id_name: str

    host_loop: asyncio.AbstractEventLoop
    guest_loop: asyncio.AbstractEventLoop
    guest_log_tags: str | None

    host_recv_bytes: AsyncRecvBytes
    host_send_bytes: AsyncSendBytes
    host_recv_ready: SyncRecvReady
    host_write_log: SyncWriteLog

    exec_env: ExecEnv
    guest_task: asyncio.Task[None] | asyncio.Future[None] | None
    guest_thread: threading.Thread | None
    system_policy: AbstractEventLoopPolicy

    _executor: concurrent.futures.ThreadPoolExecutor
    _ready: threading.Event

    logging: bool
    name: str
    needs_close: bool

    def __init__(
        self,
        *,
        id_name: str,
        recv_bytes: AsyncRecvBytes,
        send_bytes: AsyncSendBytes,
        recv_ready: SyncRecvReady,
        write_log: SyncWriteLog,
        log_tags: str | None = None,
        runner_logging: bool = False,
        wasm_path: str | None = None,  # kept for signature parity; unused here
        wasm_compiled_cache: str | None = None,  # kept for signature parity; unused here
        wasm_inherit_io: bool = True,  # kept for signature parity; unused here
    ) -> None:
        self.needs_close = False
        self.id_name = id_name
        self.guest_log_tags = log_tags
        super().__init__(logging=runner_logging, id_name=id_name)

        # for vulture
        del wasm_path
        del wasm_compiled_cache
        del wasm_inherit_io

        with self.log_as('init') as ctx:
            self.guest_task = None
            self.guest_thread = None

            self.host_recv_bytes = recv_bytes
            self.host_send_bytes = send_bytes
            self.host_recv_ready = recv_ready
            self.host_write_log = write_log

            # Event to signal when guest_task is fully initialized
            # Prevents race condition where worker thread checks guest_task before it's set
            self._ready = threading.Event()

            # Create dedicated thread pool for this sandbox (1 worker)
            # This prevents exhaustion of the default asyncio thread pool
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f'{self.name}.worker'
            )

            self.log(f'self = 0x{id(self):x}')

    ###############################################################################

    # this makes PyWasmRunner into a EventLoopPolicy

    def get_event_loop(self):
        if threading.current_thread() is self.guest_thread:
            return self.guest_loop
        return self.system_policy.get_event_loop()

    def set_event_loop(self, loop):
        return self.system_policy.set_event_loop(loop)

    def new_event_loop(self):
        if threading.current_thread() is self.guest_thread:
            return self.guest_loop
        return self.system_policy.new_event_loop()

    def get_child_watcher(self):
        return self.system_policy.get_child_watcher()

    def set_child_watcher(self, watcher):
        return self.system_policy.set_child_watcher(watcher)

    ###############################################################################

    def short_str(self):
        return self.name

    ###############################################################################

    def __from_thread(self, async_fn, *args):
        # Wait for initialization to complete (prevents race condition)
        # The main thread signals _ready after setting guest_task
        if not self._ready.is_set():
            # ctx.log('ready flag not set; waiting 5 seconds')
            if not self._ready.wait(timeout=5.0):
                # ctx.warn('timeout waiting for runner initialization')
                self.log_forced('raising WarpShutdown due to readiness')
                raise WarpShutdown()
            else:
                pass
                # ctx.log('wait complete; ready flag set')

        loop = self.host_loop

        if loop is None or not loop.is_running() or not self.guest_task:
            # if loop is None:
            #     self.log_forced('no host loop; raising WarpShutdown')
            # elif not loop.is_running():
            #     self.log_forced('host loop', loop, 'is not running; will raise WarpShutdown')
            # elif not self.guest_task:
            #     self.log_forced('guest task is', self.guest_task, '; will raise WarpShutdown')
            # ctx.log('missing host loop or guest task')
            raise WarpShutdown('missing host loop or guest task')
        coro_obj = None
        coro_fut = None
        try:
            coro_obj = async_fn(*args)
            coro_fut = asyncio.run_coroutine_threadsafe(coro_obj, self.host_loop)
            return coro_fut.result()
        except concurrent.futures.CancelledError:
            pass
            # ctx.info('PyWasmRunner has shutdown; will raise WarpShutdown')
        except BaseException as exc:
            # ctx.warn('got exception:', exc, '; will cleanup and raise WarpShutdown')
            # Ensure the coroutine is closed if we failed to schedule it
            try:
                if coro_obj is not None:
                    # ctx.warn('closing coroutine object', coro_obj)
                    coro_obj.close()
                if coro_fut is not None and not coro_fut.done():
                    # ctx.warn('cancelling future', coro_fut)
                    coro_fut.cancel()
            except BaseException as nested_exc:
                pass
                # ctx.warn('got exception while cleaning up:', nested_exc)
        # self.log_forced('raising WarpShutdown due to coroutine failure')
        raise WarpShutdown()

    def guest_recv_bytes(self) -> bytes:
        try:
            return self.__from_thread(self.host_recv_bytes)
        except BaseException:
            return QUIT

    def guest_send_ready(self) -> bool:
        try:
            return self.__from_thread(self.host_recv_ready) is True
        except BaseException:
            return False

    def guest_write_log(self, text: str) -> None:
        self.host_loop.call_soon_threadsafe(self.host_write_log, text)

    def guest_send_bytes(self, data: bytes) -> None:
        # TODO: just use call_soon_threadsafe, surely? we don't care
        # about the return value here
        self.__from_thread(self.host_send_bytes, data)

    def guest_run_msg_loop(self):
        """Must be called in an executor."""

        # sets the log tags for this particular context, which is localized
        # to this thread, so will not affect the main thread's log tags
        reset = set_log_tags(self.guest_log_tags)

        self.guest_thread = thread = threading.current_thread()
        self.system_policy = asyncio.get_event_loop_policy()

        try:
            thread.name = f'{self.name}.loop'
            with self.log_as('guest_run_msg_loop') as ctx:
                self.exec_env = exec_env = new_exec_env(
                    self.guest_recv_bytes,
                    self.guest_send_bytes,
                    self.guest_send_ready,
                    self.guest_write_log,
                )
                exec_env.init_exec_env(self.id_name, None)
                self.guest_loop = guest_loop = exec_env.get_event_loop()
                asyncio.set_event_loop(self.guest_loop)
                asyncio.set_event_loop_policy(self)
                ctx.log('guest loop set to', guest_loop)
                ctx.log('calling exec_env.guest_run_msg_loop')
                exec_env.run_msg_loop()
                ctx.log('guest_run_msg_loop returned')
        finally:
            reset()
            self.guest_thread = None
            asyncio.set_event_loop_policy(self.system_policy)

    def host_run_msg_loop(self) -> asyncio.Task:  # type: ignore[return]
        self.needs_close = True
        with self.log_as('host_run_msg_loop') as ctx:
            assert self.guest_task is None, "guest task already present"
            self.host_loop = host_loop = asyncio.get_running_loop()
            ctx.log('host loop set to', host_loop)

            # Clear the ready event before starting thread
            self._ready.clear()

            # Use dedicated executor instead of default asyncio.to_thread()
            # This prevents thread pool exhaustion when running many tests
            # Note: thread starts immediately, creating a race with initialization
            future = self._executor.submit(self.guest_run_msg_loop)
            wrapped = asyncio.wrap_future(future)
            self.guest_task = wrapped

            # Signal that initialization is complete
            # Worker thread waits for this before proceeding
            self._ready.set()

            assert self.guest_task is not None, "could not create guest task"
            return self.guest_task  # type: ignore[return-value]

    async def run_msg_loop(self) -> None:
        with self.log_as('guest_run_msg_loop') as ctx:
            try:
                task = self.host_run_msg_loop()
                ctx.log('task = ', task)
                ctx.log('result = ', await task)
                ctx.log('task = ', task)
            finally:
                self.guest_task = None

    def close(self) -> None:
        if not self.needs_close:
            return
        self.needs_close = False
        asyncio.set_event_loop_policy(self.system_policy)
        with self.log_as('close') as ctx:
            # Clear ready flag to signal shutdown
            ctx.log('clearing ready flag')
            self._ready.clear()

            if not self.guest_task:
                ctx.warn('no guest task')
            elif self.guest_task.done():
                ctx.warn('guest task already complete')
            else:
                ctx.warn('cancelling pending guest task')
                self.guest_task.cancel()
                self.guest_task = None

            del self.host_loop
            del self.guest_loop

            # Shutdown dedicated executor to clean up thread
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False, cancel_futures=True)


EXECENV_DIR = Path(__file__).parent.parent / "guest"
EXECENV_FILE = EXECENV_DIR / "execenv.py"

# Lock to prevent race conditions when multiple sandboxes load execenv concurrently.
# Without this, concurrent sandboxes could overwrite each other's wit_world in sys.modules.
_MODULE_LOCK = threading.Lock()


class ExecEnvHostModule(types.ModuleType):
    def __init__(
        self,
        recv_bytes: SyncRecvBytes,
        send_bytes: SyncSendBytes,
        recv_ready: SyncRecvReady,
        write_log: SyncWriteLog,
    ):
        super().__init__('wit_world')
        self.recv_bytes = recv_bytes
        self.send_bytes = send_bytes
        self.recv_ready = recv_ready
        self.write_log = write_log

        class BaseWitWorld: ...

        self.WitWorld = BaseWitWorld


class ExecEnvGuestModule(types.ModuleType):
    WitWorld: type

    def __new__(cls, host_module: ExecEnvHostModule):
        modules = sys.modules
        # Serialize module loading to prevent race conditions where concurrent
        # sandboxes overwrite each other's wit_world in sys.modules.
        with _MODULE_LOCK:
            try:
                module_spec = importlib.util.spec_from_file_location(
                    'sandbox.guest',
                    EXECENV_FILE,
                    submodule_search_locations=[EXECENV_DIR.as_posix()],
                )
                guest_module = importlib.util.module_from_spec(module_spec)
                guest_module.__package__ = 'sandbox.guest'
                modules['wit_world'] = host_module
                module_spec.loader.exec_module(guest_module)
                assert hasattr(guest_module, 'WitWorld')
                return guest_module
            except Exception as ex:
                P.tprint('\n'.join(sys.modules))
                P.tprint('exception during loading of execenv')
                P.print_exception(ex)
                raise


def new_exec_env(
    recv_bytes: SyncRecvBytes,
    send_bytes: SyncSendBytes,
    recv_ready: SyncRecvReady,
    write_log: SyncWriteLog,
) -> ExecEnv:
    host_module = ExecEnvHostModule(recv_bytes, send_bytes, recv_ready, write_log)
    guest_module = ExecEnvGuestModule(host_module)
    wit_world = guest_module.WitWorld()
    return wit_world


class WASMRunnerError(BaseException):
    pass
