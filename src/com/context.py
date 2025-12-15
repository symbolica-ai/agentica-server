import json
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable

from agentica_internal.core.log import should_log_cls
from agentica_internal.session_manager_messages import DEFAULT_PROTOCOL

from com.abstract import HistoryMonad
from com.gen_model import GenModel

if TYPE_CHECKING:
    from messages import InvocationNotifier
    from sandbox import Sandbox


__all__ = ['Context', 'GenModel', 'SandboxRepl']


type MonadLogFn = Callable[[str], Awaitable[None]]
type SandboxRepl = Sandbox


class Context:
    """Keeps track of chat generation that actions act upon."""

    # This is where side effects can take place as actions are executed.

    sandbox: 'SandboxRepl'
    gen: 'GenModel'
    captures: dict[str, Any]
    logging: bool
    _name: str
    monad_log: MonadLogFn
    invocation: 'InvocationNotifier | None'
    protocol: str

    _sending_system_message: bool

    def __init__(
        self,
        *,
        gen: 'GenModel',
        sandbox: 'Sandbox',
        # exec: 'Exec',
        protocol: str = DEFAULT_PROTOCOL,
        captures: dict[str, Any] | None = None,
        logging: bool = False,
        monad_log: MonadLogFn | None = None,
        invocation: 'InvocationNotifier | None' = None,
    ):
        from com.apis import API
        from com.conversion import Convert

        self.protocol = protocol
        self.monad_log = monad_log or null_monad_log
        self.invocation = invocation
        self.sandbox = sandbox
        self.gen = gen
        self.gen._parent = self
        self.captures = captures or dict()
        self.logging = should_log_cls(logging, Context)
        self._sending_system_message = False
        self._name = f'MonadContext[{sandbox.id_name if sandbox else "?"}]'

        match gen.api:
            case API.OPENAI_CHAT_COMPLETIONS:
                self.converter = Convert.OpenAIChatCompletions()
            case _:
                raise ValueError(f"Invalid API: {gen.api}")

    def __short_str__(self) -> str:
        return self._name

    def mark_system_messages(self, is_system: bool) -> None:
        self._sending_system_message = is_system

    async def repl_update(
        self, globals_data: bytes | None = None, locals_data: bytes | None = None
    ) -> None:
        globals_data = globals_data or b''
        locals_data = locals_data or b''
        return await self.sandbox.repl_init(globals_data=globals_data, locals_data=locals_data)

    async def log(self, ty: str, *args: Any) -> None:
        msg = json.dumps(
            {'type': ty, 'args': args, 'system': self._sending_system_message},
            default=str,
        )
        await self.monad_log(msg)
        if not self.logging:
            return
        from agentica_internal.core.print import tprint

        tprint(self, ty, *args)

    async def run[A](self, history: 'HistoryMonad[A]') -> A:
        while isinstance(history, HistoryMonad.Do):
            # await self.log('action', history.action)
            action_result = await history.action.perform(self)
            # await self.log('result', action_result)
            history = history.continuation(action_result)

        # await self.log('end')
        assert isinstance(history, HistoryMonad.Pure), f"Expected a Pure monad, got {type(history)}"
        match history:
            case HistoryMonad.Pure(value):
                # await self.log('return', value)
                return value
        raise ValueError(f"Monad failed to terminate")

    def __str__(self) -> str:
        from textwrap import indent

        gen_str = indent(str(self.gen), '\t')
        return f'Context(\n{gen_str}\n)'


async def null_monad_log(_: str) -> None:
    pass
