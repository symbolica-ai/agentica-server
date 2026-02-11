#! /usr/bin/env uv run
import asyncio
from os import environ

from sandbox import Sandbox


async def build_wasm():
    async def warp_send_bytes(_: bytes) -> None:
        pass

    async def warp_recv_bytes() -> bytes:
        while True:
            await asyncio.sleep(1)

    environ["AGENTICA_LOG_TAGS"] = "ALL"
    sandbox = Sandbox(
        sdk_send_bytes=warp_send_bytes,
        sdk_recv_bytes=warp_recv_bytes,
        runner_logging=True,
        mode='wasm',
    )

    print("Bringing up sandbox")
    sandbox.start()
    await sandbox.repl_init(globals_data=b"", locals_data=b"")
    print("Triggering WASM compilation")
    _ = await sandbox.repl_dir_vars()
    print("WASM is now compiled")

    await sandbox.aclose()


if __name__ == "__main__":
    asyncio.run(build_wasm())
