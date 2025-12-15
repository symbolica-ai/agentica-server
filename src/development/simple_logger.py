import asyncio
from logging import getLogger

import uvicorn
from litestar import Litestar, post
from litestar.connection import Request
from litestar.logging import LoggingConfig

logger = getLogger(__name__)

logging_config = LoggingConfig(
    root={"level": "CRITICAL", "handlers": ["console"]},
    loggers={
        __name__: {"level": "INFO", "handlers": ["console"], "propagate": False},
        "uvicorn": {"level": "CRITICAL"},
        "uvicorn.error": {"level": "CRITICAL"},
        "uvicorn.access": {"level": "CRITICAL"},
        "litestar": {"level": "CRITICAL"},
    },
    formatters={"standard": {"format": "%(asctime)s - %(levelname)s - %(name)s: %(message)s"}},
)


@post("/logs")
async def log_endpoint(request: Request) -> None:
    data = await request.json()
    logger.info(data)


app = Litestar(route_handlers=[log_endpoint], logging_config=logging_config)
config = uvicorn.Config(app, host="0.0.0.0", port=23456, log_config=None, log_level="critical")
server = uvicorn.Server(config)

if __name__ == "__main__":
    asyncio.run(server.serve())
