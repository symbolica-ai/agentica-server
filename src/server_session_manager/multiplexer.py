import asyncio
import os
import traceback
from asyncio import Queue, Task, create_task
from collections.abc import Awaitable
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, TypedDict, Unpack

from agentica_internal.internal_errors import RequestTooLargeError
from agentica_internal.multiplex_protocol import (
    MultiplexCancelMessage,
    MultiplexClientInstanceMessage,
    MultiplexClientMessage,
    MultiplexDataMessage,
    MultiplexErrorMessage,
    MultiplexErrorName,
    MultiplexInvokeMessage,
    MultiplexNewIIDResponse,
    multiplex_from_json,
)
from agentica_internal.session_manager_messages import AllServerMessage, PromptTemplate

from agentic.agent import Agent
from messages import Holder, HybridNotifier, Notifier, OTelNotifier, Poster

logger = getLogger(__name__)

from litestar.connection.websocket import WebSocketDisconnect
from litestar.status_codes import WS_1000_NORMAL_CLOSURE

crash_on_exception = os.getenv('SM_CRASH_ON_EXCEPTION') == '1'


@dataclass
class AgentContext:
    """Per-agent state managed by the Multiplexer."""

    uid: str
    agent: Agent
    notifier: HybridNotifier
    otel_notifier: OTelNotifier | None = None


@dataclass
class ServerSessionContext:
    """Context of the server session manager, required by the Multiplexer for managing agents."""

    # Agent registry
    agents: dict[str, Agent]

    # Global notifier of the server session manager
    notifier: Notifier

    # Notifier dependencies
    log_poster: Poster
    logs: Holder[str, AllServerMessage]

    # OTel dependencies
    tracer: Any  # OTracer
    parent_context: Any
    register_invocation_span: Callable
    get_invocation_span: Callable

    # User info
    user_id: str | None
    uid_to_cid: dict[str, str]


class Multiplexer:
    """
    Handles multiplexed invocations over a single WebSocket connection.
    Supports multiple agents (uids) per socket with lazy initialization.
    """

    fresh_id: Callable[[], str]
    send_bytes: Callable[[bytes], Awaitable[None]]
    recv_bytes: Callable[[], Awaitable[bytes]]

    # Dependencies for creating agent contexts
    _server_session_ctx: ServerSessionContext
    _transport_enqueue: Callable[[Any], Awaitable[None]]

    # Per-uid state (lazily initialized)
    _uid_context: dict[str, AgentContext]

    # Per-iid state
    _iid_recv_queue: dict[str, Queue[MultiplexClientInstanceMessage]]
    _invocation_tasks: dict[str, Task[None]]

    # Concurrency control
    _create_invocation: Callable[[], Awaitable[bool]]
    _destroy_invocation: Callable[[], Awaitable[None]]

    def __init__(
        self,
        fresh_id: Callable[[], str],
        send_bytes: Callable[[bytes], Awaitable[None]],
        recv_bytes: Callable[[], Awaitable[bytes]],
        transport_enqueue: Callable[[Any], Awaitable[None]],
        server_session_ctx: ServerSessionContext,
        create_invocation: Callable[[], Awaitable[bool]],
        destroy_invocation: Callable[[], Awaitable[None]],
    ):
        self.fresh_id = fresh_id
        self.send_bytes = send_bytes
        self.recv_bytes = recv_bytes
        self._transport_enqueue = transport_enqueue
        self._server_session_ctx = server_session_ctx
        self._uid_context = dict()
        self._iid_recv_queue = dict()
        self._invocation_tasks = dict()
        self._create_invocation = create_invocation
        self._destroy_invocation = destroy_invocation

    def _create_agent_context(self, uid: str) -> AgentContext | None:
        """
        Create AgentContext for a uid.
        """
        # Check if agent exists
        if uid not in self._server_session_ctx.agents:
            logger.error(f"Agent {uid} not found in agents, was it created?")
            return None

        agent = self._server_session_ctx.agents[uid]

        # Prepare model info for the session span
        model_info = {
            "gen_ai.provider.name": agent.model.provider,
            "gen_ai.request.model": agent.model.identifier,
            "agent.user.id": self._server_session_ctx.user_id or "",
            "agent.session.id": self._server_session_ctx.uid_to_cid.get(uid, "unknown"),
        }

        # Create legacy notifier
        legacy_notifier = Notifier(
            uid=uid,
            send_mx_message=self._transport_enqueue,
            log_poster=self._server_session_ctx.log_poster,
            logs=self._server_session_ctx.logs,
        )

        # Create OTel notifier - session span will be created on first invocation
        otel_notifier = (
            OTelNotifier(
                uid=uid,
                tracer=self._server_session_ctx.tracer,
                parent_context=self._server_session_ctx.parent_context,
                model_info=model_info,
                register_invocation_span=self._server_session_ctx.register_invocation_span,
                get_invocation_span=self._server_session_ctx.get_invocation_span,
            )
            if self._server_session_ctx.tracer
            else None
        )

        # Combine both notifiers
        notifier = HybridNotifier(
            uid=uid,
            send_mx_message=self._transport_enqueue,
            legacy_notifier=legacy_notifier,
            otel_notifier=otel_notifier,
        )

        return AgentContext(uid=uid, agent=agent, notifier=notifier, otel_notifier=otel_notifier)

    def _get_or_init_context(self, uid: str) -> AgentContext | None:
        """
        Get existing context or lazily initialize for a new uid.
        """
        if uid in self._uid_context:
            return self._uid_context[uid]

        # Lazy initialization
        ctx = self._create_agent_context(uid)
        if ctx is None:
            return None
        self._uid_context[uid] = ctx
        return ctx

    async def _run_invocation(self, ctx: AgentContext, **kwargs: Unpack["InvocationArgs"]) -> None:
        """Run a single invocation for the given agent context."""
        iid: str = kwargs["iid"]
        warp_locals_payload: bytes = kwargs["warp_locals_payload"]
        prompt: str | PromptTemplate = kwargs["prompt"]
        streaming: bool = kwargs["streaming"]
        parent_uid: str | None = kwargs["parent_uid"]
        parent_iid: str | None = kwargs["parent_iid"]

        await ctx.notifier.on_enter(iid, parent_uid=parent_uid, parent_iid=parent_iid)
        invocation_notifier = ctx.notifier.bind_invocation(iid)

        try:
            await ctx.agent.run(
                iid=iid,
                warp_locals_payload=warp_locals_payload,
                prompt=prompt,
                send_message=ctx.notifier.send_mx_message,
                receive_message=self._iid_recv_queue[iid].get,
                streaming=streaming,
                invocation_notifier=invocation_notifier,
            )
        except RequestTooLargeError:
            pass
        except BaseException as exc:
            stack_trace = str(exc) + "\n" + traceback.format_exc()
            await ctx.notifier.on_exception(iid, stack_trace)
            if type(exc) is not RequestTooLargeError:
                logger.error("Exception during run_invocation", exc_info=exc)
            ctx.agent.cancel(iid)
            if crash_on_exception:
                raise exc
        finally:
            await ctx.notifier.on_exit(iid)
            self._iid_recv_queue.pop(iid, None)
            await self._destroy_invocation()

    async def _send_error(
        self, uid: str, iid: str, error_name: str, error_message: str | None = None
    ) -> None:
        """Send an error message using an existing agent context's notifier."""
        ctx = self._uid_context.get(uid)
        erroneous_uid: str | None = None
        session_id: str | None = None
        session_manager_id: str | None = None
        notifier: HybridNotifier | Notifier | None = None
        if ctx is None:
            # No context for this uid - agent was never properly initialized or the packet is malformed
            notifier = self._server_session_ctx.notifier
        else:
            notifier = ctx.notifier
            erroneous_uid = uid
            session_id = ctx.agent.session_id
            session_manager_id = ctx.agent.session_manager_id

        await notifier.send_mx_message(
            MultiplexErrorMessage(
                iid=iid,
                error_name=error_name,
                error_message=error_message,
                uid=erroneous_uid,
                session_id=session_id,
                session_manager_id=session_manager_id,
            )
        )

    async def _handle_client_message(self, multiplex_message: MultiplexClientMessage) -> None:
        """
        Handle a single client message.
        """
        logger.debug("Received: %s", multiplex_message)

        match multiplex_message:
            case MultiplexInvokeMessage(
                match_id=match_id,
                uid=m_uid,
                warp_locals_payload=m_warp_locals_payload,
                prompt=m_prompt,
                streaming=m_streaming,
                parent_uid=m_parent_uid,
                parent_iid=m_parent_iid,
                timestamp=_,
            ):
                # Get or lazily initialize context for this uid
                ctx = self._get_or_init_context(m_uid)
                if ctx is None:
                    await self._send_error(
                        m_uid, match_id, MultiplexErrorName.MalformedInvokeMessageError
                    )
                    return

                iid = self.fresh_id()
                self._iid_recv_queue[iid] = Queue()

                # Check concurrency limits
                if not await self._create_invocation():
                    await self._send_error(
                        m_uid, match_id, MultiplexErrorName.TooManyInvocationsError
                    )
                    return

                await ctx.notifier.send_mx_message(
                    MultiplexNewIIDResponse(uid=m_uid, iid=iid, match_id=match_id)
                )

                self._invocation_tasks[iid] = create_task(
                    self._run_invocation(
                        ctx,
                        iid=iid,
                        warp_locals_payload=m_warp_locals_payload,
                        prompt=m_prompt or "",
                        streaming=m_streaming,
                        parent_uid=m_parent_uid,
                        parent_iid=m_parent_iid,
                    ),
                    name=f"ServerInvocation[{m_uid}:{iid}]",
                )

            case MultiplexCancelMessage(uid=m_uid, iid=m_iid, timestamp=_):
                if m_iid not in self._iid_recv_queue:
                    await self._send_error(m_uid, m_iid, MultiplexErrorName.NotRunningError)
                    return

                logger.info(f"Cancelling invocation {m_iid} for uid {m_uid}")

                ctx = self._uid_context.get(m_uid)
                if ctx:
                    ctx.agent.cancel(m_iid)

                _ = self._iid_recv_queue.pop(m_iid, None)
                _ = self._invocation_tasks.pop(m_iid, None)

            case MultiplexDataMessage(uid=m_uid, iid=m_iid):
                q = self._iid_recv_queue.get(m_iid)
                if q is None:
                    await self._send_error(m_uid, m_iid, MultiplexErrorName.NotRunningError)
                    return
                await q.put(multiplex_message)

            case _:
                raise RuntimeError(f"Unreachable {multiplex_message}")  # type checker agrees!

    async def _background_task_reader(self) -> None:
        """
        Read and dispatch messages from the WebSocket.
        """
        while True:
            try:
                msg_bytes = await self.recv_bytes()
            except WebSocketDisconnect as e:
                # disconnects are not fatal for the server side.
                if e.code != WS_1000_NORMAL_CLOSURE:
                    logger.warning(f"Non-normal websocket disconnect (code={e.code})")
                break

            multiplex_message = multiplex_from_json(msg_bytes)

            # Should never happen?
            if not isinstance(multiplex_message, MultiplexClientMessage):
                raise Exception("Received non-client message")

            await self._handle_client_message(multiplex_message)

    def _cleanup_agent_context(self, uid: str) -> None:
        """Clean up resources for a single agent context."""
        ctx = self._uid_context.get(uid)
        if ctx and ctx.otel_notifier:
            try:
                ctx.otel_notifier.end_session_span()
            except Exception as e:
                logger.debug(f"Error ending session span for {uid}: {e}")

    async def run(self) -> None:
        """
        Run the multiplexer event loop.

        This is the single public method that will be called by the server session manager.
        It processes messages until the WebSocket is closed, then cleans up all resources.
        """
        reader_task = None
        try:
            reader_task = create_task(self._background_task_reader())
            await reader_task
        except Exception as e:
            logger.error(f"reader_task ended abnormally with exception: {e}")
        finally:
            # Cancel all agents (stops inference)
            for uid, ctx in self._uid_context.items():
                try:
                    ctx.agent.cancel()
                except Exception as e:
                    logger.debug(f"Error cancelling agent {uid}: {e}")

            # Cancel and wait for all invocation tasks to actually finish
            tasks_to_cancel = list(self._invocation_tasks.values())
            for task in tasks_to_cancel:
                if not task.done():
                    _ = task.cancel()

            # Wait for all tasks to actually be cancelled/complete
            if tasks_to_cancel:
                _ = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            self._invocation_tasks.clear()
            if self._invocation_tasks:
                logger.error("invocation tasks not empty when multiplexer ended")

            # Cleanup all agent contexts
            for uid in list(self._uid_context.keys()):
                self._cleanup_agent_context(uid)

            self._uid_context.clear()
            # Clear reference to agents to allow garbage collection
            # Might not be needed since this deletes along with the client's websocket?
            del self._uid_context


class InvocationArgs(TypedDict):
    iid: str
    warp_locals_payload: bytes
    prompt: str | PromptTemplate
    streaming: bool
    parent_uid: str | None
    parent_iid: str | None
