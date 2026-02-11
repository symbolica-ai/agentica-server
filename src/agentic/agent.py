import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from agentica_internal.core.log import should_log_cls
from agentica_internal.core.print import *
from agentica_internal.core.result import Result
from agentica_internal.internal_errors import GenerationError
from agentica_internal.multiplex_protocol.multiplex_protocol import (
    MultiplexClientInstanceMessage,
    MultiplexDataMessage,
    MultiplexErrorMessage,
    MultiplexServerInstanceMessage,
)
from agentica_internal.session_manager_messages import CacheTTL, PromptTemplate, ReasoningEffort

from com.context import Context, InferenceConfig
from inference.endpoint import InferenceSystem
from messages import InvocationNotifier
from sandbox import Sandbox, SandboxMode

from .models import ProviderModel
from .monads import AgentMonads

logger = logging.getLogger(__name__)


class Agent:
    """
    A agent is an LLM session consisting of:
    - a system message: an initial introduction to the world, its capabilities, and its purpose;
    - user messages: specifying tasks, information, tools, and follow ups on the previous interaction;
    - agent messages: reasoning-repl loops that interact with the user's tools and objects;
      the repl is central to tool use and to specify final results/answers to tasks.

    There are three corresponding *interaction modes* specified by separate context monads.
    These are composed in (executed using the same execution context/state) as
    the appropriate requests come through.
    """

    uid: str
    iid: str
    session_id: str | None
    api_key: str | None

    fresh_id: Callable[[], str]
    inference_system: InferenceSystem
    sandbox: Sandbox
    inference_context: Context
    model: ProviderModel
    always_streaming: bool
    premise_prompt: str | None
    system_prompt: str | PromptTemplate | None
    warp_globals_payload: bytes
    logging: bool

    _silent_for_testing: bool
    _was_closed: bool
    _timestamp: datetime
    _sandbox_lock: asyncio.Lock
    # monads
    _interactions: AgentMonads

    # callbacks
    _send_gen_err: Callable[[GenerationError], Awaitable[None]] | None
    _send_message: Callable[[MultiplexDataMessage], Awaitable[None]] | None
    _recv_message: Callable[[], Awaitable[MultiplexClientInstanceMessage]] | None

    # task
    _tasks: list[asyncio.Task[None]]
    _pending: asyncio.Queue[bytes]

    # logging
    _sandbox_log_path: Path | None
    _sandbox_log_tags: str | None

    def __init__(
        self,
        *,
        uid: str,
        protocol: str,
        fresh_id: Callable[[], str],
        inference_system: InferenceSystem,
        model: ProviderModel,
        always_streaming: bool,
        premise: str | None,
        system: str | PromptTemplate | None,
        max_tokens_per_invocation: int | None,
        max_tokens_per_round: int | None,
        max_rounds: int | None,
        warp_globals_payload: bytes,
        reasoning_effort: ReasoningEffort | None = None,
        cache_ttl: CacheTTL | None = None,
        sandbox_mode: SandboxMode = 'from_env',
        session_id: str | None = None,
        session_manager_id: str | None = None,
        sandbox_log_path: Path | None = None,
        sandbox_log_tags: str | None = None,
        silent_for_testing: bool = False,
    ):
        self._was_closed = True  # prevent cleanup if we fail to init
        self.uid = uid
        self.iid = 'NOTRUNYET'
        self.session_id = session_id
        self.session_manager_id = session_manager_id

        self.logging = should_log_cls(False, Agent)
        self.log(".init(", colorize(uid), ")")

        self.fresh_id = fresh_id
        self.inference_system = inference_system
        self.model = model
        self.always_streaming = always_streaming
        self.premise_prompt = premise
        self.system_prompt = system
        self.warp_globals_payload = warp_globals_payload

        self._send_gen_err = None
        self._send_message = None
        self._recv_message = None
        self._sandbox_lock = asyncio.Lock()
        self._tasks = []
        self._pending = asyncio.Queue()
        self._sandbox_log_path = sandbox_log_path
        self._sandbox_log_tags = sandbox_log_tags
        self._silent_for_testing = silent_for_testing

        self.sandbox = sandbox = Sandbox(
            sdk_send_bytes=self.warp_send_bytes,
            sdk_recv_bytes=self.warp_recv_bytes,
            protocol=protocol,
            id_name=uid[:5],
            mode=sandbox_mode,
            log_path=sandbox_log_path,
            log_tags=sandbox_log_tags,
            log_inherit=sandbox_log_path is None,
        )
        sandbox.set_exception_handler(self.handle_exception)

        # Set tool support flag for prompts.
        # Only OpenAI models support custom tools (even via OpenRouter's Responses API).
        sandbox.session_info.supports_custom_tools = inference_system.uses_tool_calls

        max_tokens = max_tokens_per_invocation
        max_inference_tokens = max_tokens_per_round

        inference_config = InferenceConfig(
            iid=uid,  # TODO: what is this supposed to be?
            model=model.endpoint_identifier,
            max_rounds=max_rounds,
            max_invocation_tokens=max_tokens,
            max_inference_tokens=max_inference_tokens,
            max_completion_tokens=max_tokens,
            streaming=always_streaming,
            reasoning_effort=reasoning_effort,
            cache_ttl=cache_ttl,
            send_gen_err=self.send_gen_error,
        )
        self.inference_context = Context(
            inference_config=inference_config,
            sandbox=sandbox,
            protocol=protocol,
            system=inference_system,
        )
        self._tasks = []
        self._interactions = AgentMonads.from_model(model)

        self._timestamp = datetime.now()
        self._was_closed = False

    async def handle_exception(self, _e: BaseException) -> None:
        """Genuine internal exception from the sandbox"""
        self.cancel(self.iid)
        self.log_error('handle_exception', self.iid, _e)
        await self.send_gen_error(_e)

    def __repr__(self):
        return f'ServerAgent[{id(self):x}]'

    __short_str__ = __repr__

    def log(self, *args):
        if self.logging:
            tprint(colorize(repr(self)), *args)

    def log_error(self, *args):
        if not self._silent_for_testing:
            tprint(colorize(repr(self)), ERROR('ERROR:'), *args)

    async def fill_inbox(self) -> None:
        while True:
            self.log(f"fill_inbox awaiting message")
            recv_msg = self._recv_message
            if recv_msg is None:
                self.log_error('recv_msg gone, ending task')
                return
            msg = await recv_msg()
            self.log(f"fill_inbox got message:", msg)
            if type(msg) is not MultiplexDataMessage:
                continue
            if msg.iid != self.iid:
                self.log(f"skipping stale message {msg.iid=!r} != {self.iid}")
                continue
            await self._pending.put(msg.data)
            self.log(f"fill_inbox placed on pending queue")

    async def _ensure_system_message(self) -> None:
        """Ensure the system message is set in the inference context."""
        if self.inference_context.system.has_messages:
            return
        await self.inference_context.repl_update(
            globals_data=self.warp_globals_payload,
        )
        self.inference_context.mark_system_messages(True)
        await self.inference_context.run(
            self._interactions.init_monad(self.premise_prompt, self.system_prompt)
        )
        self.inference_context.mark_system_messages(False)

    async def warp_recv_bytes(self) -> bytes:
        self.log("warp_recv_bytes()")
        pending = self._pending
        if self._recv_message is None:
            if pending.empty():
                self.log("Agent was closed, and no pending messages, raising exception for WASM")
                raise RuntimeError('Cannot receive: Agent was closed')  # should be caught by WASM
            else:
                self.log("Agent was closed, but still have pending messages")
        data = await pending.get()
        self.log("warp_recv_bytes() ->", data)
        return data

    async def warp_send_bytes(self, payload: bytes) -> None:
        send_msg = self._send_message
        if send_msg is None:
            self.log("Agent was closed, raising exception for WASM")
            raise RuntimeError('Cannot send: Agent was closed')  # should be caught by WASM
        self.log("warp_send_bytes:", payload)
        msg = MultiplexDataMessage(
            uid=self.uid,
            iid=self.iid,
            data=payload,
        )
        await send_msg(msg)

    async def send_gen_error(self, err: GenerationError) -> None:
        self.log(f"send_gen_error ({self.iid}):", err)
        if self._send_gen_err is None:
            self.log_error("send_gen_err callback not set, cannot send error:", err)
            return
        await self._send_gen_err(err)

    async def invocation_return(self, iid: str, result: Result) -> None:
        """
        Returns a given value (or error) for an invocation, which cannot contain any weird types
        otherwise warp will not encode it. Use `Result.good()` and `Result.bad()` to make these.
        """
        assert type(result) is Result
        await self.sandbox.invocation_return(iid, result)

    async def call(self, task: str | PromptTemplate) -> None:
        """
        Give the agent a task to perform, have it return a result of the given type,
        and allow it to use any of the provided tools.
        """
        self.log(f"call()")
        try:
            user_monad = self._interactions.user_monad(task, self.system_prompt)
            m = user_monad >> self._interactions.interaction_monad
            await self.inference_context.run(m)
            self.log("call() done normally")
        finally:
            # Guard against inference_context being deleted by close() during cancellation
            if hasattr(self, 'inference_context'):
                self.inference_context.inference_config.finish_invocation()
            self.log("call() exited")

    def cancel(self, iid: str | None = None) -> None:
        self.log(f"cancel({iid!r})")
        try:
            for task in self._tasks:
                try:
                    if not task.done():
                        task.cancel()
                except:
                    pass
        finally:
            self._tasks.clear()

    def done(self) -> bool:
        return len(self._tasks) == 1

    def _cleanup_references(self) -> None:
        """Clean up circular references and resources. Shared by close() and aclose()."""
        # Break circular reference: agent -> sandbox -> bound methods -> agent
        del self.sandbox
        try:
            self._pending.put_nowait(b'')  # Sentinel to wake up waiters
        except Exception as e:
            logger.debug(f"Failed to send sentinel to pending queue during cleanup: {e}")
        del self._pending
        self._tasks.clear()
        del self.inference_context
        self._send_gen_err = None
        self._send_message = None
        self._recv_message = None
        del self._interactions

    def close(self) -> None:
        """Synchronous close - best effort, does not wait for guest shutdown.

        Use aclose() in async contexts for proper coordination.
        """
        self._was_closed = True
        self.log("close()")
        try:
            # Clear exception handler to break circular reference
            self.sandbox.set_exception_handler(None)
            self.sandbox.close()
            # Reset guest state after each invocation to prevent state leakage
            if hasattr(self.sandbox, 'reset_guest_state'):
                self.sandbox.reset_guest_state()
        except Exception as e:
            logger.warning(f"Error closing agent {self.uid}: {e}")
        finally:
            self._cleanup_references()
        self.log("closed()")

    async def aclose(self) -> None:
        """Async close - waits for guest to process shutdown before cleanup."""
        self._was_closed = True
        self.log("aclose()")
        try:
            # Clear exception handler to break circular reference
            self.sandbox.set_exception_handler(None)
            await self.sandbox.aclose()
            # Reset guest state after each invocation to prevent state leakage
            if hasattr(self.sandbox, 'reset_guest_state'):
                self.sandbox.reset_guest_state()
        except Exception as e:
            logger.warning(f"Error closing agent {self.uid}: {e}")
        finally:
            self._cleanup_references()
        self.log("aclosed()")

    def __del__(self) -> None:
        if not self._was_closed:
            logger.warning(
                f"__del__(): agent {self.uid} not closed before being garbage collected, closing now"
            )
            try:
                self.close()
            except Exception as e:
                logger.error(f"__del__(): error while closing agent {self.uid}: {e}")

    async def run(
        self,
        iid: str,
        warp_locals_payload: bytes,
        prompt: str | PromptTemplate,
        send_message: Callable[[MultiplexServerInstanceMessage], Awaitable[None]],
        receive_message: Callable[[], Awaitable[MultiplexClientInstanceMessage]],
        streaming: bool,
        invocation_notifier: InvocationNotifier,
    ) -> None:
        self.log('run', colorize(iid)) if self.logging else None
        self.log('run: awaiting sandbox lock')
        async with self._sandbox_lock:
            self.iid = iid
            self.log('run: acquired sandbox lock')
            self._send_message = send_message
            self._recv_message = receive_message
            inbox_coro = self.fill_inbox()

            async def coro():
                # setup callbacks
                await self._setup_callbacks(
                    iid=iid,
                    streaming=streaming,
                    send_message=send_message,
                    invocation_notifier=invocation_notifier,
                )
                # insert the system message
                await self._ensure_system_message()
                # run the inference monad
                return await self._run(warp_locals_payload=warp_locals_payload, prompt=prompt)

            self.log('run: creating tasks')
            inbox_task = asyncio.create_task(inbox_coro, name=f'Agent.run[{iid!r}]')
            run_task = asyncio.create_task(coro(), name=f'Agent.fill_inbox[{iid!r}]')
            self._tasks = [inbox_task, run_task]
            try:
                await run_task
            except asyncio.CancelledError:
                self.log('run: cancelled')
                pass
            except BaseException as e:
                await self.handle_exception(e)
            finally:
                if not inbox_task.done():
                    inbox_task.cancel()
                if not run_task.done():
                    run_task.cancel()
                self._tasks.clear()
                self.log('run: complete')

    async def _setup_callbacks(
        self,
        iid: str,
        streaming: bool,
        send_message: Callable[[MultiplexServerInstanceMessage], Awaitable[None]],
        invocation_notifier: InvocationNotifier,
    ) -> None:
        async def send_gen_error(err: GenerationError) -> None:
            await send_message(
                MultiplexErrorMessage.from_error(
                    iid,
                    err,
                    uid=self.uid,
                    session_id=self.session_id,
                    session_manager_id=self.session_manager_id,
                )
            )
            await invocation_notifier.log_invocation_error(err.__class__.__name__, str(err))

        async def monad_log(body: str) -> None:
            await invocation_notifier.log_monad(body)

        self.inference_context.monad_log = monad_log
        self.inference_context.invocation = invocation_notifier
        self.inference_context.inference_config.iid = iid
        self.inference_context.inference_config.streaming = self.always_streaming or streaming
        invocation_notifier.with_agent_metadata(
            model=self.model.identifier,
            provider=self.model.provider,
        )
        self._send_gen_err = send_gen_error

    async def _run(
        self,
        warp_locals_payload: bytes,
        prompt: str | PromptTemplate,
    ) -> None:
        self.log('inference_context.exec.update')
        _ = await self.inference_context.repl_update(locals_data=warp_locals_payload)
        self.log('awaiting call()')
        await self.call(prompt)
        self.log('call() completed')
        await asyncio.sleep(0.1)  # please document why
