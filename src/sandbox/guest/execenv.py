import asyncio
import random
from asyncio import AbstractEventLoop
from types import CoroutineType, ModuleType

import typeguard  # noqa: F401
import wit_world
from agentica_internal.core.asserts import assert_raises
from agentica_internal.core.log import set_log_tags, set_write_log_fn, should_log_tag
from agentica_internal.core.print import local_print, tprint
from agentica_internal.core.utils import is_type_like
from agentica_internal.warpc.worlds.agent_world import AgentWorld
from typeguard import check_type

from . import (
    poll_loop,  # noqa: F401
    the_prelude,  # noqa: F401
)
from .agent_error import AgentError
from .agent_repl import AgentRepl
from .common import _is_instance, _try_repr, _warp_aware_vars
from .stubs import (
    _hidden_locals,
    before_locals_population,
    clean_type_name,
    emit_stubs,
    format_definition,
    hide_variable,
    locals_delta,
    print_annotations,
    print_stubs,
    print_stubs_stateful,
    show_definition,
)


def _remote_print(*args):
    if AGENT_WORLD:
        AGENT_WORLD.remote_print(*args)


EXTRA_BUILTINS = {
    # exceptions
    "AgentError": AgentError,
    # repr
    "repr": _try_repr,
    "vars": _warp_aware_vars,
    # stubs
    "show_definition": show_definition,
    "format_definition": format_definition,
    "_print_annotations": print_annotations,
    "_clean_type_name": clean_type_name,
    "_emit_stubs": emit_stubs,
    "_print_stubs": print_stubs,
    "_print_stubs_stateful": print_stubs_stateful,
    "_before_locals_population": before_locals_population,
    "_locals_delta": locals_delta,
    "_hide_variable": hide_variable,
    "_hidden_locals": _hidden_locals,
    # type checking
    "_check_type": check_type,
    "_is_instance": _is_instance,
    "_is_type_like": is_type_like,
    # debugging
    "remote_print": _remote_print,
    "local_print": local_print,
    "assert_raises": assert_raises,  # replacement for pytest.raises
}

PRELOADED_MODULES = the_prelude.__modules__


def asyncio_patch():
    def run(_coro):
        if isinstance(_coro, CoroutineType):
            _coro.close()
        # patched asyncio.run to raise an exception
        raise NotImplementedError(
            "asyncio.run is not needed. "
            + "The REPL already has an event loop running. "
            + "Use `await` instead."
        )

    def get_event_loop():
        # patched asyncio.get_event_loop to raise an exception
        raise NotImplementedError(
            "<loop>.get_event_loop is not needed. "
            + "The REPL already has an event loop running. "
            + "Use `await` instead."
        )

    def set_event_loop(_loop):
        # patched asyncio.set_event_loop to raise an exception
        raise NotImplementedError(
            "<loop>.set_event_loop is not needed. "
            + "The REPL already has an event loop running. "
            + "Use `await` instead."
        )

    def new_event_loop():
        # patched asyncio.new_event_loop to raise an exception
        raise NotImplementedError(
            "asyncio.new_event_loop is not needed. "
            + "The REPL already has an event loop running. "
            + "Use `await` instead."
        )

    mod = ModuleType('asyncio')
    mod.__dict__.update(asyncio.__dict__)
    mod.run = run
    mod.get_event_loop = get_event_loop
    mod.set_event_loop = set_event_loop
    mod.new_event_loop = new_event_loop
    return mod


if asyncio in PRELOADED_MODULES:
    PRELOADED_MODULES.remove(asyncio)
    PRELOADED_MODULES.append(asyncio_patch())


class WitWorld(wit_world.WitWorld):
    def __init__(self):
        global COUNT
        COUNT += 1
        self.COUNT = COUNT

    def init_exec_env(self, id_name: str, log_tags: str | None) -> None:
        global AGENT_WORLD, AGENT_REPL, EVENT_LOOP, INITIALIZED

        if log_tags is not None:
            set_log_tags(log_tags)

        set_write_log_fn(wit_world.write_log)

        if not INITIALIZED:
            AGENT_WORLD, AGENT_REPL, EVENT_LOOP = create_agent_environment(id_name)
            INITIALIZED = True

    def get_event_loop(self):
        assert INITIALIZED
        return EVENT_LOOP

    def run_msg_loop(self):
        assert INITIALIZED
        AGENT_WORLD.run_msg_loop(
            wit_world.send_bytes,
            wit_world.recv_bytes,
            wit_world.recv_ready,
        )


COUNT: int = 0
INITIALIZED: bool = False
AGENT_WORLD: AgentWorld
AGENT_REPL: AgentRepl
EVENT_LOOP: AbstractEventLoop


def create_agent_environment(id_name: str) -> tuple[AgentWorld, AgentRepl, AbstractEventLoop]:
    log = should_log_tag(False, {'execenv'})

    if poll_loop.IN_WASM:
        loop = asyncio.get_event_loop()
        tprint('create_agent_environment used existing loop:', loop) if log else None
    else:
        # this happens in PyWasmRunner, within a thread
        assert asyncio._get_running_loop() is None, (
            "impossible: existing loop in PyWasmRunner thread"
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tprint('create_agent_environment made new loop:', loop) if log else None

    repl = AgentRepl()
    repl.vars.builtins.update(EXTRA_BUILTINS)
    repl.preload_modules(PRELOADED_MODULES)

    world = AgentWorld(name=id_name, world_id=random.randint(0, 255))
    world.attach_repl(repl)

    world.set_loop(loop)
    repl.set_loop(loop)

    return world, repl, loop
