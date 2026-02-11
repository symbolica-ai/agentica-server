import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Protocol, TextIO

import msgspec.json
from agentica_internal.core.ansi.code import strip_ansi
from agentica_internal.repl.info import ReplEvaluationInfo, ReplSessionInfo, ReplVarInfo
from msgspec import Raw

if TYPE_CHECKING:
    from agentic.protocol import MagicProtocol

from agentica_internal.core.log import (
    LogBase,
    LogContext,
    get_log_tags,
    should_log_cls,
    write_out_no_log_fn,
)
from agentica_internal.core.print import print_unclosed_error, tprint
from agentica_internal.core.result import *
from agentica_internal.repl import Scope
from agentica_internal.warpc.alias import *
from agentica_internal.warpc.messages import *
from agentica_internal.warpc.predicates import *
from agentica_internal.warpc.pure import PURE_CODEC
from agentica_internal.warpc.request.request_repl import *
from agentica_internal.warpc.worlds.interface import *
from agentica_internal.warpc_transcode.transcoder import InterceptorProto, NoopInterceptor

os.environ["WASMTIME_BACKTRACE_DETAILS"] = "1"

# Select runner implementation based on environment.
# If AGENTICA_NO_SANDBOX=1, use the pure-Python threaded runner.
# Otherwise, use the default native extension runner.


__all__ = [
    "Sandbox",
    "SandboxMode",
    "SandboxError",
    "ExceptionHandlerFn",
]

pkg_root = Path(__file__).resolve().parent
default_wasm_path = pkg_root.parent / "env.wasm"
default_compiled_cache = pkg_root / "env.wasm.compiled"

SANDBOX_ID = 0

type WasmRunnerMode = Literal['no_sandbox', 'wasm']

type ExceptionHandlerFn = Callable[[BaseException], Awaitable[None]]


class WasmRunnerP(Protocol):
    def __init__(
        self,
        *,
        id_name: str,
        recv_bytes: AsyncRecvBytes,
        send_bytes: AsyncSendBytes,
        recv_ready: SyncRecvReady,
        write_log: SyncWriteLog,
        log_tags: str | None = None,
        wasm_path: str | None = None,
        wasm_compiled_cache: str | None = None,
        runner_logging: bool = False,
        wasm_inherit_io: bool = True,
    ) -> None: ...

    async def run_msg_loop(self) -> None: ...

    def close(self) -> None: ...


type VarScope = Scope | Literal['user', 'locals', 'globals']
type SandboxMode = Literal['no_sandbox', 'wasm', 'from_env']


class Sandbox(LogBase):
    """Wraps the host WasmRunner, see env.wit.

    _pending: dict of MessageID to Future for messages issued by the controller, which should NOT be returned
    to the SDK and instead intercepted by the Sandbox itself.
    """

    _NEXT_ID: ClassVar[int] = 0

    _runner: WasmRunnerP | None
    _runner_cls: type[WasmRunnerP]
    _future: asyncio.Future[None] | None
    _run_task: asyncio.Task[None] | None
    _runner_task: asyncio.Task[None] | None  # tracks runner specifically for shutdown

    _lazy_init: dict[str, Any]
    _inbox: asyncio.Queue[bytes]
    _outbox: asyncio.Queue[bytes]
    _loop: asyncio.AbstractEventLoop | None
    _pending: dict[int, asyncio.Future[bytes]]
    _closed: bool
    _mode: Literal['no_sandbox', 'wasm']
    _name: str
    _repl_next_mid: int
    _interceptor: InterceptorProto
    _protocol: 'MagicProtocol'
    _sdk_send_bytes: AsyncSendBytes
    _sdk_recv_bytes: AsyncRecvBytes
    _exception_handler: ExceptionHandlerFn | None
    _log_path: Path | None
    _log_stream: TextIO | None

    session_info: ReplSessionInfo
    eval_info: ReplEvaluationInfo

    def __init__(
        self,
        *,
        sdk_send_bytes: AsyncSendBytes,
        sdk_recv_bytes: AsyncRecvBytes,
        id_name: str | None = None,
        log_tags: str | None = None,
        runner_logging: bool = False,
        mode: SandboxMode = 'from_env',
        protocol: str | None = None,
        logging: bool = False,
        # these control where sandbox logs are sent
        log_path: Path | None = None,
        log_inherit: bool = True,
        # Additional parameters to the Rust host:
        wasm_path: str | None = None,
        wasm_compiled_cache: str | None = None,
    ):
        self._closed = True  # in case init throws exception
        self.log_strs = []
        if id_name is None:
            id_name = str(Sandbox._NEXT_ID)
            Sandbox._NEXT_ID += 1

        super().__init__(logging=logging, id_name=id_name)

        self._mode = mode = choose_mode(mode)

        if mode == 'no_sandbox':
            from .py_runner import PyWasmRunner

            runner_cls = PyWasmRunner
        else:
            from host import WasmRunner as RustWasmRunner  # type: ignore

            runner_cls = RustWasmRunner

            # ensure wasm gets our current log tags
            if log_tags is None:
                log_tags = get_log_tags()

        self._runner_cls = runner_cls
        self._repl_next_mid = -256
        self._inbox = inbox = asyncio.Queue()
        self._outbox = outbox = asyncio.Queue()
        self._pending = {}
        self._exception_handler = None

        log = self.log

        log('init')

        def recv_ready() -> bool:
            return inbox.qsize() > 0

        async def send_bytes(b: bytes) -> None:
            try:
                log('warp_send_bytes: waiting for outbox.put', b)
                await outbox.put(b)
                log('warp_send_bytes: done')
            except BaseException:
                log('warp_send_bytes: failed; dropping')

        async def recv_bytes() -> bytes:
            try:
                log('warp_recv_bytes: waiting for inbox.get')
                data = await inbox.get()
                log('warp_recv_bytes: received', data)
                return data
            except BaseException:
                log('warp_recv_bytes: failed; returning QUIT')
                return QUIT

        def write_log(text: str) -> None:
            if log_stream := self._log_stream:
                ts = datetime.now(timezone.utc).strftime("%H%M%S-%f")
                n = len(text)
                text = text.lstrip('\n')
                nl = '\n' * (n - len(text))
                log_stream.write(f'{nl}{ts} {text}')
                log_stream.flush()
            if log_inherit:
                write_out_no_log_fn(text)

        self._log_path = log_path
        self._log_stream = None

        wasm_path = str(wasm_path or default_wasm_path)
        wasm_compiled_cache = str(wasm_compiled_cache or default_compiled_cache)
        runner_logging = should_log_cls(runner_logging, runner_cls)

        args: dict[str, Any] = {
            "id_name": id_name,
            "send_bytes": send_bytes,
            "recv_bytes": recv_bytes,
            "recv_ready": recv_ready,
            "write_log": write_log,
            "log_tags": log_tags,
            "runner_logging": runner_logging,
            "wasm_path": wasm_path,
            "wasm_compiled_cache": wasm_compiled_cache,
            "wasm_inherit_io": runner_logging,
        }
        self._lazy_init = args
        self._runner = None
        self._future = None
        self._loop = None
        self._run_task = None
        self._runner_task = None
        self._closed = False
        self._runner_cls = runner_cls

        try:
            self._log_stream = open(log_path, 'a') if log_path else None
        except Exception as exc:
            self.log_forced('cannot open log_path =', log_path, exc)

        from agentic.protocol import MagicProtocol

        self._protocol = MagicProtocol.parse(protocol)

        if self._protocol.sdk == 'python':
            interceptor = NoopInterceptor()
        else:
            from agentica_internal.warpc_transcode.transcoder import TranscodingInterceptor

            interceptor = TranscodingInterceptor()

        log('interceptor:', interceptor)
        self._interceptor = interceptor
        intercepted_recv, intercepted_send = self._interceptor.intercept_sdk(
            sdk_recv_bytes, sdk_send_bytes
        )
        self._sdk_send_bytes = intercepted_send
        self._sdk_recv_bytes = intercepted_recv

        self.session_info = ReplSessionInfo.empty()
        self.eval_info = ReplEvaluationInfo.empty()

    ###############################################################################

    def read_log_file(self, *, ansi: bool = True) -> str | None:
        if path := self._log_path:
            if path.is_file():
                text = path.read_text()
                return text if ansi else strip_ansi(text)
        return None

    ###############################################################################

    def _raise_sandbox_error(self, problem: str) -> NoReturn:
        raise SandboxError(problem)

    ###############################################################################

    def _send_quit(self, ctx: 'LogContext') -> None:
        """Send QUIT to guest."""
        if self._runner is not None:
            try:
                ctx.info('sending QUIT to ExecEnv')
                self._inbox.put_nowait(QUIT)
            except BaseException as exc:
                ctx.print_exception(exc)

    def _cleanup_runner(self, ctx: 'LogContext') -> None:
        """Clean up runner resources. Called after QUIT is sent/processed."""
        future = self._future
        runner = self._runner
        run_task = self._run_task

        # call underlying runner close
        if runner is not None:
            ctx.log('closing WASMRunner')
            runner.close()  # type: ignore[misc]
            self._runner = None

        # cancel run task if still running
        if run_task is not None and not run_task.done():
            ctx.log('cancelling run task')
            run_task.cancel()  # type: ignore[misc]
            self._run_task = None

        if future is not None and not future.done():
            ctx.info('cancelling Rust future')
            future.cancel()
            self._future = None

        if log_stream := self._log_stream:
            ctx.info('closing log stream')
            log_stream.close()
            self._log_stream = None

    def close(self) -> None:
        """Synchronous close - best effort, does not wait for guest shutdown.

        Use aclose() in async contexts for proper coordination.
        """
        if self._closed:
            return
        self._closed = True

        with self.log_as("close") as ctx:
            # Clear circular references to agent methods
            # Use None instead of del since these may be accessed in exception handlers
            self._sdk_send_bytes = None  # type: ignore
            self._sdk_recv_bytes = None  # type: ignore
            self._exception_handler = None

            self._send_quit(ctx)
            self._cleanup_runner(ctx)

    async def aclose(self, timeout: float = 0.1) -> None:
        """Async close - waits for runner to finish before cleanup.

        Args:
            timeout: Maximum time to wait for runner shutdown (default 0.1s)
        """
        if self._closed:
            return
        self._closed = True

        with self.log_as("aclose") as ctx:
            self._send_quit(ctx)

            # Wait for the runner task to actually complete (not just QUIT delivery)
            runner_task = self._runner_task
            if runner_task is not None and not runner_task.done():
                try:
                    await asyncio.wait_for(runner_task, timeout=timeout)
                    ctx.info('runner task completed; shutdown confirmed')
                except asyncio.TimeoutError:
                    ctx.info('shutdown timeout; proceeding with forced close')
                except asyncio.CancelledError:
                    ctx.info('runner task was cancelled')

            # Now safe to clear references and cleanup
            self._sdk_send_bytes = None  # type: ignore
            self._sdk_recv_bytes = None  # type: ignore
            self._exception_handler = None

            self._cleanup_runner(ctx)

    ###############################################################################

    def __del__(self):
        if self.logging:
            self.log('garbage collected')
        if not self._closed:
            print_unclosed_error(self)

    ###############################################################################

    @property
    def is_no_sandbox(self) -> bool:
        return self._mode == 'no_sandbox'

    @property
    def is_wasm(self) -> bool:
        return self._mode == 'wasm'

    ###############################################################################

    @property
    def wasm_runner(self) -> WasmRunnerP:
        if self._runner is not None:
            return self._runner
        with self.log_as(f"initializing {self._mode} wasm_runner") as ctx:
            runner = self._runner_cls(**self._lazy_init)
            self._runner = runner
            return runner

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        if loop := self._loop:
            return loop
        try:
            self._loop = loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # this should never happen? most famous words in programming
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.log("created new loop for sandbox:", loop)

    ###############################################################################

    def set_exception_handler(self, handler: ExceptionHandlerFn | None) -> None:
        self._exception_handler = handler

    @property
    def running(self) -> bool:
        return bool(self._run_task) and not self._run_task.done()

    ###############################################################################

    def start(self) -> None:
        if self.running:
            return
        with self.log_as("start") as ctx:
            loop = self.event_loop
            ctx.info('creating run task')
            self._run_task = loop.create_task(self.run(), name=f'{self.log_name}.run')

    async def run(self) -> None:
        """
        Executes three sub-tasks:
        * one for the wasm_runner itself
        * one to drain the outbox
        * one to fill the inbox

        The runner task is tracked separately so aclose() can wait for it to complete.
        When the runner finishes, we cancel the helper tasks.
        """
        inbox_task: asyncio.Task | None = None
        outbox_task: asyncio.Task | None = None

        try:
            with self.log_as("run") as ctx:
                async with asyncio.TaskGroup() as tg:
                    name = self.log_name
                    ctx.info('creating fill_inbox task')
                    inbox_task = tg.create_task(self.fill_inbox(), name=f'{name}.fill_inbox')
                    ctx.info('creating drain_outbox task')
                    outbox_task = tg.create_task(self.drain_outbox(), name=f'{name}.drain_outbox')
                    # coro if python, future if rust
                    future_or_coro = self.wasm_runner.run_msg_loop()
                    if isinstance(future_or_coro, asyncio.Future):
                        ctx.info("Rust backend")
                        self._future = future_or_coro
                        # Wrap the future in a task so we can track it
                        self._runner_task = tg.create_task(
                            self._await_runner(future_or_coro, inbox_task, outbox_task),
                            name=f'{name}.runner_wrapper',
                        )
                    else:
                        ctx.info("Python backend")
                        ctx.info('creating warp_run_msg_loop task')
                        self._runner_task = tg.create_task(
                            self._await_runner(future_or_coro, inbox_task, outbox_task),
                            name=f'{name}.runner_wrapper',
                        )
        except* asyncio.CancelledError:
            pass
        except* BaseException as e:
            self.log_forced("exception during Sandbox.run:", e)
            if self._exception_handler is not None:
                await self._exception_handler(e)
        finally:
            if (
                self._run_task is not None
                and self._run_task.done()
                and (exc := self._run_task.exception()) is not None
                and self._exception_handler is not None
            ):
                await self._exception_handler(exc)
            await self.aclose()
            self._future = None
            self._run_task = None
            self._runner_task = None

    async def _await_runner(
        self,
        runner_coro,
        inbox_task: asyncio.Task | None,
        outbox_task: asyncio.Task | None,
    ) -> None:
        """Wrapper that awaits the runner and signals completion."""
        try:
            await runner_coro
        finally:
            # Runner has finished - cancel helper tasks
            if inbox_task and not inbox_task.done():
                inbox_task.cancel()
            if outbox_task and not outbox_task.done():
                outbox_task.cancel()

    async def fill_inbox(self):
        with self.log_as("fill_inbox") as ctx:
            put_inbox = self._inbox.put
            recv_bytes = self._sdk_recv_bytes
            while True:
                ctx.info('waiting for inbox to fill')
                data = await recv_bytes()
                ctx.info('received', data)
                if type(data) is not bytes:
                    ctx.info('invalid, aborting inbox')
                    break
                await put_inbox(data)
                ctx.info('added to inbox')

    async def drain_outbox(self):
        with self.log_as("drain_outbox") as ctx:
            get_outbox = self._outbox.get
            msg_done = self._outbox.task_done
            send_bytes = self._sdk_send_bytes
            while True:
                ctx.info('waiting for outbox to fill')
                data = await get_outbox()
                ctx.info('received', data)

                # decode replies destined for Sandbox
                msg = RPCMsg.from_msgpack(data)
                ctx.info('decoded to', msg)

                # only intercept replies and only if we have a waiter for this mid
                if isinstance(msg, FramedResponseMsg):
                    mid = msg.mid
                    ctx.info('message id =', mid)
                    fut = self._pending.pop(mid, None)
                    if fut is not None and not fut.done():
                        fut.set_result(data)
                        ctx.info("reply intercepted by sandbox to fulfill pending REPL request")
                        msg_done()
                        continue  # do not forward

                ctx.info('forwarding to sdk...')
                await send_bytes(data)
                msg_done()

                ctx.info('forwarded')

    # WARP functionality
    # ------------------

    async def execute_repl_request(
        self, req: ReplRequest, test: PredicateFn | None, fmt: EncodeFmt, ctx: LogContext
    ) -> Any:
        req_msg = req.encode(PURE_CODEC)
        result = await self.execute_repl_request_msg(req_msg, (), fmt, ctx)
        if result.is_err:
            inner_error = result.error
            error = SandboxError(f'sandbox-local error executing {req}: {repr(inner_error)}')
            raise error
        value = result.value
        if fmt == 'raw':
            test = lambda m: type(m) is Raw
        if test and not test(value):
            raise SandboxError(f'executing {req} gave invalid type {type(value).__name__!r}')
        return value

    async def execute_repl_request_msg(
        self, repl_msg: ReplRequestMsg, defs: tuple, fmt: EncodeFmt, ctx: LogContext
    ) -> Result:
        self.start()
        ctx.info("sending repl request to wasm:", repl_msg)
        mid = self._repl_next_mid
        self._repl_next_mid -= 1

        request_msg = FramedRequestMsg(mid=mid, fid=0, request=repl_msg, fmt=fmt, defs=defs)
        request_data = request_msg.to_msgpack()

        fut: asyncio.Future[bytes] = self.event_loop.create_future()
        # Store before sending to avoid a race with a fast reply
        self._pending[mid] = fut
        ctx.info("adding to inbox")
        await self._inbox.put(request_data)
        ctx.info("awaiting reply")
        response_data = await fut
        response_msg = FramedResponseMsg.from_msgpack(response_data)
        result = response_msg.result.decode(PURE_CODEC)
        ctx.info("got (decoded) reply:", result)
        return result

    async def invocation_return(self, iid: str | int, result: Result):
        with self.log_as("invocation_return_value", iid):
            result_msg = ResultMsg.encode(result, PURE_CODEC)
            future_msg = FutureResultMsg(iid, result_msg)
            data = future_msg.to_msgpack()
            await self._outbox.put(data)
            await self._outbox.join()

    async def flush(self):
        await self._outbox.join()

    ############################################################################

    # REPL functionality
    # ------------------

    async def repl_init(self, *, globals_data: bytes, locals_data: bytes):
        with self.log_as("repl_init") as ctx:
            globals_data = self._interceptor.decode_defs(globals_data)
            locals_data = self._interceptor.decode_defs(locals_data)
            ctx.log("init with interceptor", self._interceptor)
            repl_init_msg, def_msgs = bytes_to_repl_init_data(globals_data, locals_data)
            # print(f"@@@ def_msgs: {repl_init_msg}")
            result = await self.execute_repl_request_msg(repl_init_msg, def_msgs, 'json', ctx)
            info_updates = result.realize()
            if isinstance(info_updates, dict) and info_updates:
                update_keys = info_updates.keys()
                ctx.log("got session updates:", ', '.join(update_keys))
                self.session_info.update(info_updates)
            else:
                ctx.log("no session updates:", type(info_updates))

    # --------------------------------------------------------------------------

    async def repl_run_code(self, code: str, **options) -> ReplEvaluationInfo:
        with self.log_as('repl_run_code') as ctx:
            request = ReplRunCode(code, **options)
            dct = await self.execute_repl_request(request, is_rec, 'json', ctx)
            try:
                self.log('info dict keys: ', list(dct.keys()))
                info = ReplEvaluationInfo(**dct)
                self.eval_info.__dict__.update(dct)
            except:
                raise SandboxError(f'invalid ReplEvaluationInfo dict {dct}')
            # the logic here is that we treat the key 'iid' in the options specially, and use it
            # to reply with a future_msg
            if info.has_result:
                iid = options.get('iid')
                if isinstance(iid, str):
                    await self.__send_eval_summary_future_msg(info, iid)
            return info

    async def __send_eval_summary_future_msg(self, summary: ReplEvaluationInfo, iid: str):
        if summary.has_raised_error:
            raw = await self.repl_call_method('get_last_exception', fmt='raw')
            result_msg = ResultMsg(True, None, raw)
        elif summary.has_return_value:
            raw = await self.repl_call_method('get_last_return_value', fmt='raw')
            result_msg = ResultMsg(True, raw, None)
        else:
            raise SandboxError(f'last evaluation did not raise an exception or return a value')
        response_msg = FutureResultMsg(fid=iid, result=result_msg)
        payload = response_msg.to_msgpack()
        await self._outbox.put(payload)
        await self._outbox.join()

    # --------------------------------------------------------------------------

    async def repl_call_method(
        self, name: str, *args, test: PredicateFn | None = None, fmt: EncodeFmt = 'json', **kwargs
    ):
        with self.log_as('repl_call_method', name) as ctx:
            request = ReplCallMethod(name, args, kwargs)
            return await self.execute_repl_request(request, test, fmt, ctx)

    # --------------------------------------------------------------------------

    async def repl_dir_vars(self, scope: VarScope = Scope.USER) -> list[str]:
        return await self.repl_call_method('dir_vars', str(scope), test=is_strlist)

    async def repl_has_var(self, key: str, scope: VarScope = Scope.USER) -> list[str]:
        return await self.repl_call_method('has_var', str(scope), key, test=is_bool)

    async def repl_var_info(self, key: str, scope: VarScope = Scope.USER) -> ReplVarInfo | None:
        result = await self.repl_call_method('var_info', str(scope), key)
        return ReplVarInfo(**result) if isinstance(result, dict) else None

    # --------------------------------------------------------------------------

    async def repl_mfi_payload(self, fid: FutureID, key: str, is_exc: bool) -> bytes:
        raw_bytes = await self.repl_call_method('get_var', Scope.USER, key, fmt='raw')
        result_cls = ErrorMsg if is_exc else ValueMsg
        result_msg = result_cls(Raw(raw_bytes))
        response_msg = FutureResultMsg(fid=fid, result=result_msg)
        payload = response_msg.to_msgpack()
        return payload

    # --------------------------------------------------------------------------

    async def repl_exec_mode(self, code: str, mode: str) -> tuple[str, str, str]:
        summary = await self.repl_run_code(code)
        out_str = summary.out_str
        output = summary.output
        if mode == 'eval' and out_str:
            output = output.removesuffix(out_str + '\n')
        std_err = summary.traceback_str or ''
        std_out = output.replace(std_err, '') if std_err else output
        if mode == 'eval':
            return out_str, std_out, std_err
        if mode == 'command':
            return output.removesuffix('\n'), std_out.removesuffix('\n'), std_err
        return output, std_out, std_err

    # --------------------------------------------------------------------------

    async def repl_return_var(self, iid: str, name: str) -> None:
        data = await self.repl_mfi_payload(iid, name, False)
        await self._outbox.put(data)
        await self._outbox.join()

    # --------------------------------------------------------------------------

    async def repl_raise_var(self, iid: str, name: str) -> None:
        data = await self.repl_mfi_payload(iid, name, True)
        await self._outbox.put(data)
        await self._outbox.join()

    # --------------------------------------------------------------------------

    # we should delete some or all of these

    async def repl_exec(self, code: str) -> tuple[str, str, str]:
        return await self.repl_exec_mode(code, "exec")

    async def repl_eval(self, code: str) -> tuple[str, str, str]:
        return await self.repl_exec_mode(code, "eval")

    async def repl_command(self, line: str) -> tuple[str, str, str]:
        return await self.repl_exec_mode(line, "command")

    async def repl_snippet(self, code: str) -> tuple[str, str, str]:
        """Run a code snippet iPython style (last expression is printed)"""
        return await self.repl_exec_mode(code, "snippet")

    async def repl_run(self, code: str) -> str:
        """Run a command in single-mode. Get result of sys.displayhook."""
        summary = await self.repl_run_code(code)
        return summary.out_str or ''

    # --------------------------------------------------------------------------

    # legacy functionality probably ready to delete

    @staticmethod
    def testing_sandbox(
        *,
        logging: bool = False,
        mode: WasmRunnerMode = 'no_sandbox',
        log_tags: str = None,
        log_path: Path | None = None,
        log_inherit: bool = True,
    ) -> 'Sandbox':
        async def send(data: bytes) -> None:
            tprint('sending:', data)

        async def recv() -> bytes:
            return None

        return Sandbox(
            sdk_send_bytes=send,
            sdk_recv_bytes=recv,
            logging=logging,
            mode=mode,
            log_tags=log_tags,
            log_path=log_path,
            log_inherit=log_inherit,
        )


class SandboxError(BaseException):
    pass


def choose_mode(mode: Literal['no_sandbox', 'wasm', 'from_env']) -> WasmRunnerMode:
    if mode == 'from_env':
        no_sandbox = os.environ.get("AGENTICA_NO_SANDBOX")
        return 'no_sandbox' if (no_sandbox == '1') else 'wasm'
    elif mode == 'no_sandbox':
        return 'no_sandbox'
    elif mode == 'wasm':
        return 'wasm'
    raise ValueError(f"Unknown WASMRunner mode: {mode!r}")


enc_json = msgspec.json.Encoder().encode
dec_json = msgspec.json.Decoder().decode
