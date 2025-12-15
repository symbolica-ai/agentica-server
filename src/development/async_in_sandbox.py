import asyncio

from sandbox import Sandbox


async def warp_recv_bytes() -> bytes:
    print("Resource in")
    return b"Hello, world!"


async def warp_send_bytes(payload: bytes) -> None:
    print(f"Resource out: {payload}")


async def main():
    print("Starting sandbox")
    sandbox = Sandbox(warp_recv_bytes=warp_recv_bytes, warp_send_bytes=warp_send_bytes)
    async_code = """
async def main():
    return 'Hello, world!'
result = asyncio.run(main())
print(result)
"""
    await sandbox.enable_debug(True)

    print("Running async code")
    out, _stdout, _stderr = await sandbox.repl_exec(async_code, "exec")
    print(out)
    print("=" * 80)
    print(_stdout)
    print("=" * 80)
    print(_stderr)


if __name__ == "__main__":
    asyncio.run(main())
