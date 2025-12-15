from __future__ import annotations

import asyncio

import uvicorn

from .app import create_app


def main() -> None:
    app = create_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=23456, log_level="debug", access_log=True)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
