import asyncio
import gc
import logging
import re
import uuid
import weakref
from asyncio import Lock
from collections.abc import AsyncGenerator, Generator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import current_thread
from typing import TYPE_CHECKING, Any, Callable, Literal

from agentica_internal.core.gc import inspect_referrers, log_refcount
from agentica_internal.core.strs import timestamp_str
from agentica_internal.session_manager_messages import AllServerMessage, server_message_to_dict
from agentica_internal.session_manager_messages.session_manager_messages import CreateAgentRequest
from agentica_internal.telemetry.otel import *
from litestar import WebSocket
from litestar.exceptions import WebSocketDisconnect
from litestar.status_codes import WS_1011_INTERNAL_ERROR

from agentic import Agent
from agentic.models import ProviderModel
from messages import (
    FilterFn,
    Holder,
    Notifier,
    Poster,
    server_notifier,
)

from .multiplexer import (
    Multiplexer,
    ServerSessionContext,
)
from .transport import WebSocketSender

if TYPE_CHECKING:
    from .provider import InferenceProvider

logger = logging.getLogger(__name__)


type UID = str
type IID = str
type CID = str  # Client Session ID

type SandboxMode = Literal['no_sandbox', 'wasm', 'from_env']


class API(Enum):
    CHAT_COMPLETIONS = "chat/completions"
    RESPONSES = "responses"
    MESSAGES = "messages"


def infer_api_from_endpoint(url: str) -> tuple[str, API]:
    """Infer the API type from the inference endpoint URL.

    Args:
        url: The inference endpoint URL (e.g., "https://api.openai.com/v1/responses")

    Returns:
        API.OPENAI_RESPONSES if URL ends with '/responses'
        API.OPENAI_CHAT_COMPLETIONS if URL ends with '/chat/completions'
        API.MESSAGES if URL ends with '/messages'

    Raises:
        ValueError: If the URL doesn't end with a recognized API path

    Notes:
        For Messages API (Anthropic), the base_url is stripped down to exclude /v1/
        since the Anthropic SDK adds the full path /v1/messages itself.
        For OpenAI APIs, the base_url includes /v1/ as expected.
    """
    normalized = url.rstrip('/')
    for path in API:
        if normalized.endswith(path.value):
            base_url = normalized[: -len(path.value)]
            # Anthropic SDK expects base_url without /v1/ - it adds /v1/messages itself
            if path == API.MESSAGES:
                base_url_stripped = base_url.rstrip('/')
                if base_url_stripped.endswith('/v1'):
                    base_url = base_url_stripped[:-3]
            return (base_url, path)
    raise ValueError(f"Unknown API type: {url}")


@dataclass
class Session:
    """tracks client session against its associated agents."""

    cid: CID
    uids: set[UID] = field(default_factory=set)
    start_timestamp: str = field(default_factory=timestamp_str)

    def add_agent(self, uid: UID) -> None:
        self.uids.add(uid)

    def remove_agent(self, uid: UID) -> None:
        self.uids.discard(uid)

    def drain(self) -> Generator[UID, None, None]:
        while self.uids:
            yield self.uids.pop()

    def is_empty(self) -> bool:
        return len(self.uids) == 0


class ServerSessionManager:
    log_poster: Poster
    id_issuer: Callable[[], str]
    providers: list["InferenceProvider"]
    tracer: OTracer
    user_id: str | None  # inference user-id
    max_concurrent_invocations: int | None
    sandbox_mode: SandboxMode
    sandbox_log_path: str | None
    sandbox_log_tags: str | None

    _agents: dict[UID, Agent]
    _logs: Holder[UID, AllServerMessage]
    _notifier: Notifier
    _concurrent_invocations: int
    _count_lock: Lock

    # Session management
    _sessions: dict[CID, Session]  # Track all sessions and their agents
    _uid_to_cid: dict[UID, CID]  # Reverse mapping for quick lookup (uid -> cid)

    # Span tracking for OpenTelemetry
    _iid_spans: dict[
        tuple[UID, IID], OSpan
    ]  # Store invocation spans for (uid, iid) pair -> span for nesting child agents

    def __init__(
        self,
        log_poster: Poster,
        providers: list["InferenceProvider"],
        user_id: str | None = None,
        tracer: OTracer = None,
        id_issuer: Callable[[], str] = lambda: str(uuid.uuid4()),
        max_concurrent_invocations: int | None = None,
        sandbox_mode: SandboxMode = 'from_env',
        sandbox_log_path: str | None = None,
        sandbox_log_tags: str | None = None,
        silent_for_testing: bool = False,
    ):
        current_thread().name = 'ServerSessionManager'

        self.id_issuer = id_issuer
        self.log_poster = log_poster
        self.providers = providers
        self.user_id = user_id or 'sm-' + uuid.uuid4().hex
        self.max_concurrent_invocations = max_concurrent_invocations
        self.sandbox_mode = sandbox_mode
        self.tracer = tracer
        self.sandbox_log_path = sandbox_log_path
        self.sandbox_log_tags = sandbox_log_tags

        self._agents = dict()
        self._logs = Holder(server_message_to_dict)
        self._notifier = server_notifier(
            log_poster=log_poster,
            logs=self._logs,
        )
        self._concurrent_invocations = 0
        self._count_lock = Lock()

        # Initialize session management
        self._sessions = dict()
        self._uid_to_cid = dict()
        self._silent_for_testing = silent_for_testing

        # Initialize invocation span tracking for nesting child agents
        self._iid_spans = dict()

    async def _create_invocation(self) -> bool:
        async with self._count_lock:
            logger.debug(
                f"Invocation check: {self._concurrent_invocations}/{self.max_concurrent_invocations}"
            )
            if (
                self.max_concurrent_invocations is not None
                and self._concurrent_invocations >= self.max_concurrent_invocations
            ):
                logger.debug(f"LIMIT REACHED: cannot create invocation")
                return False
            self._concurrent_invocations += 1
            logger.debug(
                f"Invocation created: now {self._concurrent_invocations}/{self.max_concurrent_invocations}"
            )
            return True

    async def _destroy_invocation(self) -> None:
        async with self._count_lock:
            self._concurrent_invocations -= 1
            logger.debug(
                f"Invocation destroyed: now {self._concurrent_invocations}/{self.max_concurrent_invocations}"
            )
            if self._concurrent_invocations < 0:
                logger.error(
                    f"Concurrent invocations is less than 0: {self._concurrent_invocations}"
                )

    async def create_agent(
        self, body: CreateAgentRequest, cid: CID, session_manager_id: str | None = None
    ) -> str:
        uid = self.id_issuer()

        model = ProviderModel.parse(body.model)
        streaming = body.streaming
        warp_globals_payload = body.warp_globals_payload
        premise = body.doc
        system = body.system
        max_tokens_per_invocation = body.max_tokens_per_invocation
        max_tokens_per_round = body.max_tokens_per_round
        max_rounds = body.max_rounds
        reasoning_effort = body.reasoning_effort
        cache_ttl = body.cache_ttl
        protocol = body.protocol

        # Create the appropriate InferenceSystem based on matching provider
        from .provider import NoMatchingProviderError

        inferencer = None
        for provider in self.providers:
            inferencer = provider.create_inference_system(
                model_id=body.model,
                fresh_id=self.id_issuer,
                notifier=self._notifier,
            )
            if inferencer is not None:
                break

        if inferencer is None:
            raise NoMatchingProviderError(body.model)

        # TODO: re-implement authentication check
        # TODO: re-implement OpenRouter model validation
        await model.validate_openrouter_model(inferencer)

        # Associate agent with client session
        session = self._sessions.get(cid)
        if session is None:
            # this is somewhat defensive.
            # if we are somehow creating an agent before registering the client session,
            # then register the client session now.
            self._sessions[cid] = session = Session(cid)

        session.add_agent(uid)
        self._uid_to_cid[uid] = cid
        logger.info(f"Created agent {uid} for session {cid}")

        sandbox_log_path = None
        if log_path := self.sandbox_log_path:
            sid_ts = session.start_timestamp
            sid_ymd = sid_ts.split('-', 1)[0]
            uid_ts = timestamp_str()
            uid_ymd = uid_ts.split('-', 1)[0]
            sandbox_log_path = log_path.format(
                sid_ymd=sid_ymd,
                sid_ts=sid_ts,
                uid_ymd=uid_ymd,
                uid_ts=uid_ts,
                sid=cid,
                uid=uid,
            )
            sandbox_log_path = Path(sandbox_log_path)
            sandbox_log_path.unlink(missing_ok=True)
            try:
                sandbox_log_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Logging agent {uid!r} warp messages to {sandbox_log_path}")
            except Exception as exc:
                logger.error(f"Failed to create agent log path: {sandbox_log_path}", exc_info=exc)
                sandbox_log_path = None

        ma = Agent(
            uid=uid,
            fresh_id=self.id_issuer,
            inference_system=inferencer,
            model=model,
            always_streaming=streaming,
            premise=premise,
            system=system,
            max_tokens_per_invocation=max_tokens_per_invocation,
            max_tokens_per_round=max_tokens_per_round,
            max_rounds=max_rounds,
            warp_globals_payload=warp_globals_payload,
            reasoning_effort=reasoning_effort,
            cache_ttl=cache_ttl,
            protocol=protocol,
            sandbox_mode=self.sandbox_mode,
            session_id=cid,
            session_manager_id=session_manager_id,
            sandbox_log_path=sandbox_log_path,
            sandbox_log_tags=self.sandbox_log_tags,
            silent_for_testing=self._silent_for_testing,
        )
        self._agents[uid] = ma

        await self._notifier.on_create_agent(body)

        # Log agent creation with structured data for Loki
        logger.info(
            f"Agent created: uid={uid}, model={body.model}",
            extra={
                "agent_uid": uid,
                "agent_model": body.model,
                "agent_streaming": streaming,
                "has_premise": bool(premise),
                "has_system": bool(system),
                "protocol": protocol,
            },
        )
        return uid

    def sandbox_log_paths(
        self, max_files: int = 0, find: str = '', uid: str = ''
    ) -> tuple[Path, list[Path]] | None:
        ymd = timestamp_str().split('-', 1)[0]
        if log_path := self.sandbox_log_path:
            # substitute in the ymd so that we only retrieve logs for the current day
            logger.info(f"Getting agent log paths for template {log_path!r}")
            log_path = log_path.replace('{sid_ymd}', ymd)
            log_path = Path(log_path).absolute()
            parts = log_path.parts
            idx = next((i for i, part in enumerate(parts) if '{' in part), len(parts))
            # obtain the longest prefix of the template that is not parameterized
            root = Path(*parts[:idx])
            suffix = log_path.suffix
            logger.info(f"Getting agent log paths within {root}")
            # obtain all files within this prefix with the right suffix
            files = [f for f in root.rglob(f"*{suffix}") if f.is_file()]
            logger.info(f"Obtained {len(files)} files")
            if uid:
                files = [f for f in files if uid in f.as_posix()]
                logger.info(f"`uid` narrowed to {len(files)} files")
            if find:
                find_re = re.compile(find)
                files = [f for f in files if find_re.search(f.read_text())]
                logger.info(f"`find` narrowed to {len(files)} files")
            files.sort()
            if max_files and len(files) > max_files:
                logger.info(f"Truncated to {max_files} files")
                files = files[-max_files:]
            return root, files
        return None

    def sandbox_logs_merged(
        self, max_files: int = 0, max_chunks: int = 0, find: str = '', uid: str = ''
    ) -> str:
        import re

        chunks_by_key = {}
        result = self.sandbox_log_paths(max_files=max_files, find=find, uid=uid)
        if result is None:
            return ''
        _, files = result
        find_re = re.compile(find) if find else None
        for file in files:
            text = file.read_text()
            # Split on lines that start with the timestamp pattern
            chunks = re.split(r'^(?=\d{6}-\d{6}\s)', text, flags=re.MULTILINE)

            for chunk in chunks:
                if not chunk:
                    continue
                if find_re and not find_re.search(chunk):
                    continue
                if match := re.match(r'^(\d{6}-\d{6})\s', chunk):
                    key = match.group(1)
                    chunks_by_key.setdefault(key, []).append(chunk)

        logger.info(f"Obtained {len(chunks_by_key)} chunks")
        sorted_keys = sorted(chunks_by_key)
        if max_chunks and len(sorted_keys) > max_chunks:
            logger.info(f"Truncating to {max_chunks} chunks")
            sorted_keys = sorted_keys[-max_chunks:]
        merged = ''.join(''.join(chunks_by_key[key]) for key in sorted_keys)
        return merged

    def get_json_logs_by_uid(
        self, uid: str, filter_fn: FilterFn[AllServerMessage] = None
    ) -> Iterator[dict[str, Any]]:
        return self._logs.get_json_by_key(uid, filter_fn)

    def get_logs(
        self, uid: str, iid: str, filter_fn: FilterFn[AllServerMessage] = None
    ) -> Iterator[AllServerMessage]:
        for log in self._logs.get_by_key(uid, filter_fn):
            if hasattr(log, 'iid') and log.iid == iid:
                yield log

    def get_json_logs(
        self, uid: str, iid: str, filter_fn: FilterFn[AllServerMessage] = None
    ) -> Iterator[dict[str, Any]]:
        for log in self.get_logs(uid, iid, filter_fn):
            yield self._logs.to_json(log)

    def get_json_all_logs(
        self, filter_fn: FilterFn[AllServerMessage] = None
    ) -> Iterator[dict[str, Any]]:
        return self._logs.get_json_all(filter_fn)

    async def listen(self, uid: str, iid: str) -> AsyncGenerator[AllServerMessage, None]:
        """Listen for logs under uid/idd provided"""
        queue: asyncio.Queue[AllServerMessage] = asyncio.Queue()
        self._logs.add_listener(uid, queue.put_nowait)
        while True:
            log = await queue.get()
            if hasattr(log, 'iid') and log.iid == iid:
                yield log

    async def listen_json(self, uid: str, iid: str) -> AsyncGenerator[dict[str, Any], None]:
        async for log in self.listen(uid, iid):
            yield self._logs.to_json(log)

    async def listen_all_json(self, uid: str) -> AsyncGenerator[dict[str, Any], None]:
        async for log in self.listen_all(uid):
            yield self._logs.to_json(log)

    async def listen_all(self, uid: str) -> AsyncGenerator[AllServerMessage, None]:
        """Listen for all logs under uid provided"""
        queue: asyncio.Queue[AllServerMessage] = asyncio.Queue()
        self._logs.add_listener(uid, queue.put_nowait)
        while True:
            log = await queue.get()
            yield log

    async def listen_global(self) -> AsyncGenerator[AllServerMessage, None]:
        queue: asyncio.Queue[AllServerMessage] = asyncio.Queue()
        self._logs.add_global_listener(queue.put_nowait)
        while True:
            log = await queue.get()
            yield log

    async def listen_global_json(self) -> AsyncGenerator[dict[str, Any], None]:
        async for log in self.listen_global():
            yield self._logs.to_json(log)

    def register_session(self, cid: CID) -> None:
        """Register a new client session before any agents are created."""
        sessions = self._sessions
        if cid not in sessions:
            sessions[cid] = Session(cid)
            logger.info(f"Registered new session {cid}")
        else:
            logger.debug(f"Session {cid} already registered")

    def has_session(self, cid: CID) -> bool:
        """Check if a session exists."""
        return cid in self._sessions

    def has_agent(self, uid: UID) -> bool:
        """Check if an agent exists."""
        return uid in self._agents

    async def deregister_session(self, cid: CID) -> None:
        """Deregister a client session."""
        session = self._sessions.pop(cid, None)
        if session is None:
            return
        logger.info(f"[*] Deregistering session {cid}")
        for uid in session.drain():
            logger.info(f"Cleaning up agent {uid} for session {cid}")
            await self._cleanup_agent(uid)

        logger.info(f"[*] Deregistered session {cid}")

    def register_invocation_span(self, uid: UID, iid: IID, span: OSpan) -> None:
        """Register an invocation span for potential use as parent context for child agents."""
        if span:
            self._iid_spans[(uid, iid)] = span

    def get_invocation_span(self, uid: UID, iid: IID) -> OSpan:
        """Get an invocation span by (uid, iid) for nesting child agent sessions."""
        return self._iid_spans.get((uid, iid))

    async def destroy_agent(self, uid: str) -> None:
        """Destroy an agent and its resources."""
        if uid not in self._agents:
            raise ValueError(f"Agent {uid} not found")
        await self._cleanup_agent(uid)

    async def _cleanup_agent(self, uid: str) -> None:
        """Clean up a single agent and its resources."""
        if uid not in self._agents:
            # assume cleanup of all uid-related dicts and loggers has been completed
            return

        logger.info(f"[*] Cleaning up agent {uid}")

        self._logs.remove_key(uid)

        # Clean up invocation spans for this agent
        iid_span_keys = [key for key in self._iid_spans.keys() if key[0] == uid]
        for key in iid_span_keys:
            self._iid_spans.pop(key, None)

        # delete primary reference to agent and allow
        # discussing agent without incrementing its ref-count
        agent_ref = weakref.ref(self._agents.pop(uid))
        try:
            logger.info(f"Cleaning up agent {uid}")

            # logging: check reference count before cleanup
            if agent := agent_ref():
                _ = log_refcount(agent, "before cleanup", uid=uid)
                # cancel any running invocations before closing
                if hasattr(agent, 'iid') and agent.iid:
                    agent.cancel(agent.iid)
                await agent.aclose()
                del agent

            await asyncio.sleep(0)
        except Exception as e:
            logger.warning(f"Error closing agent {uid}: {e}")

        # remove from session tracking
        sessions = self._sessions
        if cid := self._uid_to_cid.pop(uid, None):
            if session := sessions.get(cid):
                session.remove_agent(uid)
                if session.is_empty():
                    del sessions[cid]

        # === aggressive GC triggering ===

        # this part is just trying to trigger the agent resource to be
        # genuinely collected and freed by the GC.
        # logs the success of this effort.

        # check who's still holding references
        if agent := agent_ref():
            _ = log_refcount(agent, "after cleanup", uid=uid)
            inspect_referrers(agent, comparison_dicts={"self._agents": self._agents}, uid=uid)
            del agent

        # encourage garbage collection to clean up the agent
        collected = gc.collect()
        logger.debug(f"Garbage collection collected {collected} objects")

        # check after 1st GC collection
        if agent := agent_ref():
            _ = log_refcount(agent, "after garbage collection", uid=uid)
            inspect_referrers(agent, comparison_dicts={"self._agents": self._agents}, uid=uid)
            del agent

        # give async tasks chance to complete, then do final GC pass
        await asyncio.sleep(0.1)
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"Second garbage collection pass collected {collected} more objects")

        # final check to see if agent was collected
        if agent_ref() is None:
            logger.debug(f"Agent {uid} successfully garbage collected!")
        else:
            logger.info(f"Agent {uid} still alive after both cleanup attempts")

    async def loop_on_socket(self, socket: WebSocket, cid: CID, parent_context: Any = None) -> None:
        """
        Handles multiplexed agent invocations over a single WebSocket connection.

        This method should be called exactly once per WebSocket connection. It accepts
        the socket, creates a Multiplexer, and runs it until the connection closes.

        The Multiplexer supports multiple agents (uids) over the same socket. Agent
        contexts are lazily initialized when the first MultiplexInvokeMessage for each
        uid is received.

        Session cleanup:
        - When the socket closes, only the session identified by `cid` is cleaned up
        - Other sessions (from other sockets) are not affected

        Error handling:
        - Any exceptions: closes with WS_1011_INTERNAL_ERROR
        - We expect the client to close the websocket normally; we only close it
          ourselves if there is an error.

        Args:
            socket: WebSocket connection (must not be accepted yet)
            cid: Client session ID - identifies which session to clean up on close
            parent_context: Optional trace context from client for distributed tracing
        """
        if cid not in self._sessions:
            raise ValueError(
                f"Session {cid} not found when looping on socket, create the session first"
            )

        await socket.accept()

        transport: WebSocketSender | None = None
        multiplexer: Multiplexer | None = None
        close_reason: str = "normal_completion"
        close_exception: Exception | None = None

        try:
            transport = WebSocketSender(send_bytes=socket.send_bytes)
            await transport.start()

            self._notifier.set_send_mx_message(transport.enqueue)

            server_session_ctx = ServerSessionContext(
                agents=self._agents,
                notifier=self._notifier,
                log_poster=self.log_poster,
                logs=self._logs,
                tracer=self.tracer,
                parent_context=parent_context,
                register_invocation_span=self.register_invocation_span,
                get_invocation_span=self.get_invocation_span,
                user_id=self.user_id,
                uid_to_cid=self._uid_to_cid,
            )

            multiplexer = Multiplexer(
                fresh_id=self.id_issuer,
                send_bytes=socket.send_bytes,
                recv_bytes=socket.receive_bytes,
                transport_enqueue=transport.enqueue,
                server_session_ctx=server_session_ctx,
                create_invocation=self._create_invocation,
                destroy_invocation=self._destroy_invocation,
            )

            await multiplexer.run()
        except WebSocketDisconnect as e:
            close_reason = "client_disconnected"
            close_exception = e
            logger.info(f"WebSocket disconnected by client: {e}")
        except Exception as e:
            close_reason = "exception"
            close_exception = e
            logger.error(f"multiplexer.run() raised an exception: {e}")
            try:
                await socket.close(code=WS_1011_INTERNAL_ERROR)
            except:
                pass
        finally:
            # Clean up gracefully - if socket isn't closed, try to close it
            # Catch WebSocketDisconnect which happens if socket is already closed
            logger.info(
                f"Closing websocket connection",
                extra={
                    "user_id": self.user_id,
                    "close_reason": close_reason,
                    "close_exception_type": type(close_exception).__name__
                    if close_exception
                    else None,
                    "close_exception_message": str(close_exception) if close_exception else None,
                    "socket_state": getattr(socket, "state", None),
                    "socket_client": getattr(socket, "client", None),
                    "socket_path": socket.scope.get("path") if hasattr(socket, "scope") else None,
                    "socket_query_string": socket.scope.get("query_string", b"").decode()
                    if hasattr(socket, "scope")
                    else None,
                    "active_sessions": list(self._sessions.keys()),
                    "active_session_count": len(self._sessions),
                    "active_agents": list(self._agents.keys()),
                    "active_agent_count": len(self._agents),
                    "concurrent_invocations": self._concurrent_invocations,
                    "transport_active": transport is not None,
                    "multiplexer_active": multiplexer is not None,
                    "has_parent_context": parent_context is not None,
                },
            )
            try:
                await socket.close(code=WS_1011_INTERNAL_ERROR)
            except (WebSocketDisconnect, RuntimeError):
                # Socket already closed or disconnected - this is fine
                pass
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")

            if transport is not None:
                await transport.stop()

            # Explicitly delete multiplexer to break reference cycles
            if multiplexer is not None:
                del multiplexer

            # Clean up only the session associated with this socket
            await self.deregister_session(cid)

    def close(self) -> None:
        """Clean up all agentic functions and their resources."""
        for uid, agent in list(self._agents.items()):
            try:
                logger.info(f"Closing agentic function/agent {uid}")
                agent.close()
            except Exception as e:
                logger.warning(f"Error closing agentic function/agent {uid}: {e}")
        self._agents.clear()

        # Clear span tracking
        self._iid_spans.clear()

        # Clear session tracking
        self._sessions.clear()
        self._uid_to_cid.clear()
