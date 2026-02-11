import signal
import threading
from asyncio import AbstractEventLoop, Queue, get_running_loop
from os import environ, getpid, kill
from sys import __stderr__ as STDERR
from typing import Self

from _pytest.outcomes import Skipped
from agentica_internal.core.log import LogBase
from agentica_internal.core.print import ddiv, hdiv, print_asyncio_stacks, print_exception, tprint
from agentica_internal.warpc.worlds.sdk_world import SDKWorld

from sandbox import Sandbox, SandboxMode

__all__ = ['SandboxTestContext', 'Sandbox', 'SDKWorld']


class SandboxTestContext(LogBase):
    used: bool
    done: bool
    logging: bool
    name: str
    sb_mode: SandboxMode

    _sandbox: Sandbox
    _sdk_world: SDKWorld
    _globals: dict[str, object]
    _locals: dict[str, object]
    _timeout: int
    _loop: AbstractEventLoop
    _name: str

    _sb_to_sdk: Queue
    _sdk_to_sb: Queue

    def __init__(
        self,
        name: str,
        *args,
        sm_logging: bool = False,
        sm_timeout: int = 0,
        sb_mode: SandboxMode = 'from_env',
        **kwargs,
    ):
        global num_runs

        super().__init__()
        assert type(name) is str, f"first arg {name!r} must be name of test"

        if environ.get('SM_TEST_LOGGING', None) == '1':
            sm_logging = True

        self.used = False
        self.done = False
        self.log_name = name
        self.logging = sm_logging
        self.sb_mode = sb_mode

        self._name = f'{type(self).__name__}({self.name!r})'
        self.log('new session')

        self._globals = {}
        self._locals = {}

        self._timeout = sm_timeout
        self._loop = get_running_loop()

        self.add_globals(*args, **kwargs)

    def add_globals(self, *args, **kwargs) -> Self:
        add_args_kwargs(self._name, self._globals, args, kwargs)
        return self

    def add_locals(self, *args, **kwargs) -> Self:
        add_args_kwargs(self._name, self._locals, args, kwargs)
        return self

    def __short_str__(self):
        return str(self)

    def __setup__(self):
        logging = self.logging
        sdk_world = SDKWorld(logging=logging, name=self.name)
        sdk_world.event_loop = get_running_loop()

        self._sb_to_sdk = sb_to_sdk = Queue()
        self._sdk_to_sb = sdk_to_sb = Queue()

        sandbox = Sandbox(
            sdk_send_bytes=sb_to_sdk.put,
            sdk_recv_bytes=sdk_to_sb.get,
            logging=logging,
            mode=self.sb_mode,
        )

        self._sandbox = sandbox
        self._sdk_world = sdk_world

    async def __populate__(self):
        globals_payload = self._sdk_world.to_payload(self._globals)
        locals_payload = self._sdk_world.to_payload(self._locals)
        try:
            await self._sandbox.repl_init(globals_data=globals_payload, locals_data=locals_payload)
        except BaseException as exc:
            gnames = ','.join(map(repr, self._globals.keys()))
            lnames = ','.join(map(repr, self._locals.keys()))
            STDERR.write(
                f"{self!r} encountered exception while setting up globals={gnames}, locals={lnames}\n"
            )
            print_exception(exc)
            STDERR.flush()
            self.__timed_out__()

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def __short_str__(self) -> str:
        return self._name

    def __timed_out__(self, *args):
        if self.done:
            return
        time = self._timeout
        STDERR.write('\n' * 3)
        STDERR.write('█' * 80)
        STDERR.write(f'\n{repr(self)} timed out after {time} seconds\n')
        STDERR.flush()
        print_asyncio_stacks()
        STDERR.write(f'killing entire process')
        STDERR.flush()
        kill(getpid(), signal.SIGABRT)

    async def __aenter__(self) -> Sandbox:
        thread = threading.current_thread()
        self._thread_name = thread.name
        name = self.name
        if len(name) > 20:
            name = '⋯.' + name.split('.')[-1]
        if len(name) > 20:
            name = '⋯_' + name.split('_')[-1]
        thread.name = name
        with self.log_as('async_enter') as ctx:
            assert not self.used, "Do not use a SandboxWorldPair twice"

            self.used = True

            if timeout := self._timeout:
                ctx.log('creating watchdog timer')
                signal.signal(signal.SIGALRM, self.__timed_out__)
                signal.alarm(timeout)

            ctx.log('creating Sandbox and SDKWorld')
            self.__setup__()

            ctx.log('starting inbox loop')
            self._sdk_world.start_msg_loop(self._sdk_to_sb.put, self._sb_to_sdk.get)

            ctx.log('populating sandbox with globals:', self._globals)
            ctx.log('populating sandbox with locals:', self._locals)
            await self.__populate__()

            hdiv() if self.logging else None
            return self._sandbox

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        thread = threading.current_thread()
        thread.name = self._thread_name
        self.done = True
        hdiv() if self.logging else None
        if exc_type and exc_type not in (TimeoutError, AssertionError, Skipped):
            ddiv()
            tprint(f"{self!r} encountered exception {exc_type.__name__}:")
            print_exception(exc_val)
        with self.log_as('async_exit') as ctx:
            await self.__shutdown__()

    async def __shutdown__(self):
        await self._sandbox.aclose()
        self._sdk_world.close()
        del self._sandbox
        del self._sdk_world


def add_args_kwargs(sm_name: str, dct: dict, args: tuple, kwargs: dict[str, object]):
    dct.update(kwargs)
    for arg in args:
        name = getattr(arg, '__name__', None)
        if not isinstance(name, str):
            raise TypeError(f"{sm_name} given a non-named global or local: {arg!r}")
        dct[name] = arg
