import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from os import getenv
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from sandbox import Sandbox


@pytest.fixture()
def make_dummy_sandbox():
    def _make(
        logging=False,
    ) -> AbstractAsyncContextManager['Sandbox']:
        from sandbox import Sandbox

        async def _dummy_send_bytes(payload: bytes) -> None:
            pass

        async def _dummy_recv_bytes() -> bytes:
            while True:
                await asyncio.sleep(0)

        @asynccontextmanager
        async def _context():
            sandbox = Sandbox(
                sdk_send_bytes=_dummy_send_bytes,
                sdk_recv_bytes=_dummy_recv_bytes,
                logging=logging,
                mode="from_env",
            )
            try:
                yield sandbox
            finally:
                await sandbox.aclose()

        return _context()

    return _make


# by default, run every test under both modes: local and wasm

if getenv('SKIP_WASM') == '1':
    runners = ['local']
elif getenv('SKIP_LOCAL') == '1':
    runners = ['wasm']
else:
    runners = ['local', 'wasm']


@pytest.fixture(params=runners, ids=runners, autouse=True)
def runner_mode(request, monkeypatch):
    mode = request.param
    if mode == 'local':
        if getenv('SKIP_LOCAL') == '1':
            pytest.skip("Skipping local runner because SKIP_LOCAL=1")
        else:
            monkeypatch.setenv("AGENTICA_NO_SANDBOX", '1')

    elif mode == 'wasm':
        if getenv('SKIP_WASM') == '1':
            pytest.skip("Skipping wasm runner because SKIP_WASM=1")
        else:
            monkeypatch.delenv("AGENTICA_NO_SANDBOX", raising=False)
            monkeypatch.setenv("WASMTIME_BACKTRACE_DETAILS", '1')

    from agentica_internal.core.print import print_asyncio_stacks_in_n_seconds

    print_asyncio_stacks_in_n_seconds(30)

    return mode


@pytest_asyncio.fixture
async def delayed_stacktrace():
    from agentica_internal.core.print import print_asyncio_stacks_in_n_seconds

    print_asyncio_stacks_in_n_seconds(3)
    yield


@pytest_asyncio.fixture
async def short_logging():
    from agentica_internal.core.log import ScopedLogging

    with ScopedLogging('SHORT', is_global=True):
        yield


@pytest_asyncio.fixture
async def virt_logging():
    from agentica_internal.core.log import ScopedLogging

    with ScopedLogging('VIRT', is_global=True):
        yield


@pytest_asyncio.fixture()
def all_logging(is_local_runner):
    from agentica_internal.core.log import ScopedLogging

    with ScopedLogging('ALL', is_global=True):
        yield


@pytest_asyncio.fixture()
def agent_logging(is_local_runner):
    from agentica_internal.core.log import ScopedLogging

    with ScopedLogging('AGENT', is_global=True):
        yield


@pytest.fixture()
def is_local_runner(runner_mode):
    return runner_mode == 'local'


@pytest.fixture(autouse=True)
def auto_timeout(request):
    timeout_seconds = int(getenv("PYTEST_TIMEOUT", "30"))
    if timeout_seconds > 0 and getenv("DISABLE_TEST_TIMEOUT") != '1':
        request.node.add_marker(pytest.mark.timeout(timeout_seconds))
