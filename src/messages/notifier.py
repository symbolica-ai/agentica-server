import os
from collections.abc import Awaitable
from logging import getLogger
from typing import Callable

from agentica_internal.multiplex_protocol import (
    MultiplexErrorMessage,
    MultiplexInvocationEventMessage,
    MultiplexServerMessage,
)
from agentica_internal.session_manager_messages import (
    AllServerMessage,
    CreateAgentRequest,
    SMCreateAgentMessage,
    SMInferenceErrorMessageMessage,
    SMInferenceRequestMessage,
    SMInferenceResponseMessage,
    SMInvocationEnterMessage,
    SMInvocationErrorMessage,
    SMInvocationExitMessage,
)
from agentica_internal.session_manager_messages.session_manager_messages import (
    InteractionEvent,
    SMDestroyAgentMessage,
    SMInvocationInteractionMessage,
)

from .holder import Holder
from .poster import Poster

logger = getLogger(__name__)

BROADCAST_LOGS = os.getenv('BROADCAST_LOGS') == '1'


class Notifier:
    uid: str
    send_mx_message: Callable[[MultiplexServerMessage], Awaitable[None]]
    log_poster: Poster
    logs: Holder[str, AllServerMessage]

    def __init__(
        self,
        uid: str,
        send_mx_message: Callable[[MultiplexServerMessage], Awaitable[None]],
        log_poster: Poster,
        logs: Holder[str, AllServerMessage],
    ):
        self.uid = uid
        self.send_mx_message = send_mx_message
        self.log_poster = log_poster
        self.logs = logs

    def set_send_mx_message(
        self, send_mx_message: Callable[[MultiplexServerMessage], Awaitable[None]]
    ) -> None:
        self.send_mx_message = send_mx_message

    async def append_to_log(self, msg: AllServerMessage) -> None:
        logger.debug("Log: %s", msg)
        if BROADCAST_LOGS:
            await self.log_poster.post(msg)
        self.logs.add(self.uid, msg)

    async def on_inference_request(
        self,
        inference_id: str,
        iid: str,
        request_str: str,
        timeout: int | None = None,
    ) -> None:
        await self.append_to_log(
            SMInferenceRequestMessage(
                uid=self.uid,
                iid=iid,
                inference_id=inference_id,
                request=request_str,
                timeout=timeout,
            )
        )

    async def on_inference_response(
        self,
        inference_id: str,
        iid: str,
        response_str: str,
    ) -> None:
        await self.append_to_log(
            SMInferenceResponseMessage(
                uid=self.uid,
                iid=iid,
                inference_id=inference_id,
                response=response_str,
            )
        )

    async def on_inference_error(
        self,
        inference_id: str,
        iid: str,
        err: BaseException,
        message: str,
    ) -> None:
        await self.append_to_log(
            SMInferenceErrorMessageMessage(
                uid=self.uid,
                iid=iid,
                inference_id=inference_id,
                error_type=str(type(err)),
                error_message=message,
            )
        )

    async def on_enter(self, iid: str) -> None:
        await self.append_to_log(
            SMInvocationEnterMessage(
                uid=self.uid,
                iid=iid,
            )
        )
        await self.send_mx_message(
            MultiplexInvocationEventMessage(
                uid=self.uid,
                iid=iid,
                event='ENTER',
            )
        )

    async def on_exception(self, iid: str, err: str) -> None:
        await self.append_to_log(
            SMInvocationErrorMessage(
                uid=self.uid,
                iid=iid,
                error_type=str(type(err)),
                error_message=str(err),
            )
        )
        # Send error message so SDK receives details instead of generic "unexpected state" error
        await self.send_mx_message(
            MultiplexErrorMessage(
                iid=iid,
                uid=self.uid,
                error_name="InternalServerError",
                error_message=err,
            )
        )
        # Keep event message for backward compatibility
        await self.send_mx_message(
            MultiplexInvocationEventMessage(
                uid=self.uid,
                iid=iid,
                event='ERROR',
            )
        )

    async def on_exit(self, iid: str) -> None:
        await self.append_to_log(
            SMInvocationExitMessage(
                uid=self.uid,
                iid=iid,
            )
        )
        await self.send_mx_message(
            MultiplexInvocationEventMessage(
                uid=self.uid,
                iid=iid,
                event='EXIT',
            )
        )

    async def log_interaction(
        self,
        iid: str,
        event: InteractionEvent,
    ) -> None:
        await self.append_to_log(
            SMInvocationInteractionMessage(
                uid=self.uid,
                iid=iid,
                event=event,
            )
        )

    async def on_create_agent(self, body: CreateAgentRequest) -> None:
        await self.append_to_log(
            SMCreateAgentMessage(
                uid=self.uid,
                doc=body.doc,
                system=body.system,
                model=body.model,
                streaming=body.streaming,
            )
        )

    async def on_destroy_agent(self) -> None:
        await self.append_to_log(SMDestroyAgentMessage(uid=self.uid))


def server_notifier(log_poster: Poster, logs: Holder[str, AllServerMessage]) -> Notifier:
    async def null(_: MultiplexServerMessage) -> None:
        pass

    return Notifier(uid="server", send_mx_message=null, log_poster=log_poster, logs=logs)
