import asyncio
from os import environ

environ['WASMTIME_BACKTRACE_DETAILS'] = '1'
environ['AGENTICA_NO_SANDBOX'] = '0'

from agentica_internal.warpc.worlds.asyncio_world import AsyncIOWorld

world = AsyncIOWorld()

count = 0
# payloads = list(world.setup_repl_payloads({'my_global': 5}, {'my_local': 6}))
# payloads.append(world.repl_execute_payload('print(my_global + my_local)', 'eval'))

from sandbox import (
    Sandbox,
)


async def warp_send_bytes(payload: bytes) -> None:
    print('EE warp sent: ', payload)
    pass


async def warp_recv_bytes() -> bytes:
    # return payloads.pop(0) if payloads else b'\0'
    return b'\0'


sb = Sandbox(
    sdk_send_bytes=warp_send_bytes,
    sdk_recv_bytes=warp_recv_bytes,
)


async def main():
    print('starting event loop')
    print(await sb.repl_exec('print(5555)'))
    print(await sb.repl_exec('print(5555)'))
    print(await sb.repl_exec('x = 5'))
    print(await sb.repl_get_var_schema('x'))
    print(await sb.repl_eval('1 + x'))
    await sb.run_msg_loop()
    print('done')


asyncio.run(main())
