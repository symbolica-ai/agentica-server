import asyncio

import pytest

from sm_testing import *


@pytest.mark.asyncio
async def test_sandbox_nonblock():
    finish_order = []

    async def ping():
        for i in range(5):
            # print(f'ping {i}', flush=True)
            finish_order.append(f'ping {i}')
            await asyncio.sleep(0.5)

    async with SandboxTestContext('sm') as sb:

        async def repl_task():
            # print('starting repl_task', flush=True)
            await sb.repl_run_code('import time; time.sleep(10)')
            # print('repl_task finished', flush=True)
            finish_order.append('repl')

        start = asyncio.get_event_loop().time()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ping())
            tg.create_task(repl_task())
        elapsed = asyncio.get_event_loop().time() - start

        # If concurrent: ping finishes first (~5s), then repl (~10s), total ~10s
        # If blocking: repl finishes first (~10s), then ping (~15s), total ~15s
        assert finish_order == ['ping 0', 'ping 1', 'ping 2', 'ping 3', 'ping 4', 'repl'], (
            f"Expected ping to finish first, got {finish_order}"
        )
        assert elapsed < 12, f"Tasks appear to have run sequentially (took {elapsed:.1f}s)"
