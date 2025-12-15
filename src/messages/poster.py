import logging
from dataclasses import dataclass

import httpx
from agentica_internal.session_manager_messages import AllServerMessage, server_message_to_dict

logger = logging.getLogger(__name__)

LOG_POSTER_ERRORS = False


@dataclass
class Poster:
    url: str

    async def post(self, msg: AllServerMessage) -> None:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(self.url, json=server_message_to_dict(msg))
        except Exception as e:
            if LOG_POSTER_ERRORS:
                logger.error(f"Error posting to {self.url}: {e}")
