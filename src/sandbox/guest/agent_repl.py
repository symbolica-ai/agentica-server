from collections.abc import Callable
from types import ModuleType, NoneType
from typing import Any

import typeguard
from agentica_internal.core.anno import anno_str, is_anno
from agentica_internal.repl.all import *

from .agent_error import AgentError
from .safe_repr import safe_repr
from .stubs import emit_stubs

__all__ = [
    'AgentRepl',
]


class AgentRepl(BaseRepl):
    """
    Specialization of `repl.Repl` for agentic use.

    Functionality / behavior this introduces are:
    1. checking of `return XXX` for type correctness
    2. capturing of dunder vars (`__return_type` etc.)
    3. serving as a gateway for obtaining information for agent prompting

    Attributes:
    * `ret_type`:      the actual return type
    * `ret_anno`:  `   anno_str` form of the above
    * `self_fn`:       for agentic functions, the *actual* function the repl is computing
    * `self_info`:     serializable information about `self_fn`
    * `locals_delta`:  running delta of added/removed/deleted locals
    * `globals_delta`: running delta of added/removed/deleted globals
    """

    # stable across invocations; contains SystemInfo etc.
    info: ReplSessionInfo

    # the actual return type
    return_type: Any

    # track what changes between invocations
    locals_delta: VarsDelta
    globals_delta: VarsDelta
    dirty: set[str]

    max_repr_len: int

    def __post_init__(self):
        self.options.run_is_await = True
        self.locals_delta = self.vars.local_deltas.open()
        self.globals_delta = self.vars.global_deltas.open()
        self.info = ReplSessionInfo.empty()
        self.dirty = set()
        self.vars.hide(REPL_VAR_NAMES)
        self.update_system_info()
        self.return_type = Any
        self.max_repr_len = 128

    ############################################################################

    def initialize(
        self,
        *,
        local_vars: Vars | None = None,
        global_vars: Vars | None = None,
        hidden_vars: VarKeys = (),
    ) -> Any:
        if self.logging:
            self.log('initialize()')
            self.log('local_vars =', local_vars) if local_vars else None
            self.log('global_vars =', global_vars) if global_vars else None
            self.log('hidden_vars =', hidden_vars) if hidden_vars else None

        # update the running deltas for locals and globals
        self.locals_delta.for_update(self.vars.locals, local_vars)
        self.globals_delta.for_update(self.vars.globals, global_vars)

        super().initialize(local_vars=local_vars, global_vars=global_vars, hidden_vars=hidden_vars)

        # after initialization, capture the __return_type etc. note that
        # AgenticFunction puts __return_type in globals, but Agent puts it in locals
        # better yet would be to have at third 'internal' scope that is both automatically
        # hidden, the official place for these to go, and immutable to the agent itself
        for name, setter in REPL_VAR_TABLE:
            if local_vars and name in local_vars:
                setter(self, local_vars[name])
            elif global_vars and name in global_vars:
                setter(self, global_vars[name])

        # find modules
        if global_vars:
            if modules := tuple(v for v in global_vars.values() if issubclass(type(v), ModuleType)):
                self.log("exposing global modules", modules)
                self.add_modules(*modules)

        self.update_resources_info(local_vars, Scope.LOCALS)
        self.update_resources_info(global_vars, Scope.GLOBALS)
        self.update_system_info()

        return self.get_updated_info()

    ############################################################################

    def on_reset(self):
        self.info = ReplSessionInfo.empty()
        self.vars.clear()
        self.dirty.clear()
        self.vars.hide(REPL_VAR_NAMES)

    def update_resources_info(self, scope_vars: Vars | None, scope: Scope):
        if scope_vars is None:
            return
        hidden = self.vars.hidden
        new_dict = {k: v for k, v in scope_vars.items() if k not in hidden}
        self.log('new', scope, 'resources:', new_dict) if new_dict else None
        fmt_repr = self.fmt_repr
        new_reprs = {k: fmt_repr(v) for k, v in new_dict.items()}
        new_kinds = {k: var_kind(v) for k, v in new_dict.items()}
        new_stub = self.make_stub_str(new_dict) if new_dict else ''
        new_names = tuple(new_dict.keys())
        if scope == 'globals':
            self.info.globals.update(new_names, new_kinds, new_reprs, new_stub)
            self.dirty.add('globals')
        else:
            self.info.locals.clear()
            self.info.locals.update(new_names, new_kinds, new_reprs, new_stub)
            self.dirty.add('locals')

    def update_system_info(self) -> None:
        info = self.info.system
        info.modules = self.get_loaded_modules()
        self.dirty.add('system')

    ############################################################################

    def get_session_info(self) -> ReplSessionInfo:
        return self.info

    def get_global_resources_info(self) -> ReplResourcesInfo:
        return self.info.globals

    def get_local_resources_info(self) -> ReplResourcesInfo:
        return self.info.locals

    def get_inputs_info(self) -> ReplResourcesInfo:
        return self.info.locals

    def get_updated_info(self) -> dict[str, Any]:
        keys = tuple(self.dirty)
        self.dirty.clear()
        info = self.info
        self.log('updating info for:', ', '.join(keys))
        return {key: getattr(info, key) for key in keys}

    ############################################################################

    # these setters are invoked when `.initialize` sees `__return_type` etc.

    def set_task_description(self, description: str) -> None:
        pass

    def set_return_type(self, return_type: Any, /) -> None:
        if is_anno(return_type):
            self.log('setting return_type to', return_type)
            self.return_type = return_type
            info = self.info.returns
            info.type_str = anno_str(return_type)
            info.is_text = return_type is str
            info.is_none = return_type is None or return_type is NoneType
            self.dirty.add('returns')
        else:
            self.log('invalid return_type', return_type)

    def set_agentic_fn(self, agentic_fn: Callable, /) -> None:
        if callable(agentic_fn):
            self.log('setting agentic_fn to', agentic_fn)
            fn_info = self.info.agentic_fn
            fn_info.set_from_function(agentic_fn)
            # fn_info.doc_str = self.fmt_doc_str(fn_info.doc_str)
            fn_info.fun_stub = self.make_stub_str({fn_info.fun_name: agentic_fn})
            fn_info.args_stub = self.make_anno_only_stub_str(fn_info.arg_annos)
            self.dirty.add('agentic_fn')
        else:
            self.log('invalid agentic_fn', agentic_fn)

    def set_role(self, role: ReplRole, /) -> None:
        if type(role) is str and role in VALID_REPL_ROLES:
            self.log('setting role to', role)
            self.info.role = role
            self.dirty.add('role')
        else:
            self.log('invalid role', role)

    def set_max_repr_len(self, max_len: int, /) -> None:
        if type(max_len) is int:
            self.log('setting max_repr_len to', max_len)
            self.max_repr_len = max_len
        else:
            self.log('invalid max_repr_len', max_len)

    ############################################################################

    def on_raised_exception(self, evaluation: ReplEvaluationData, error: ReplError, /) -> None:
        # if the agent raised an exception we will catch it here.
        # we strip off the AgentError, if necessary
        if isinstance(error.exception, AgentError):
            error.set_exception(error.exception.inner_exception)
        super().on_raised_exception(evaluation, error)

    def return_hook(self, value: object, /) -> None:
        # called at the moment that `result = XXX` or `return XXX` occurs.
        # `check_return_value` will cause a TypeGuard exception to be thrown if
        # the value is bad, which prevents `Agent.return_hook` from obtaining the
        # value
        self.check_return_value(value)
        super().return_hook(value)

    ############################################################################

    def check_return_value(self, value: object, /) -> None:
        """
        Either returns None or raises an exception (from TypeGuard) that
        includes a note saying the return type was wrong.
        """
        ret_type = self.return_type
        if ret_type is Any:
            return
        try:
            typeguard.check_type(value, ret_type)
            return
        except typeguard.TypeCheckError as e:
            type_error = e
        except:
            return

        actual = type(value).__name__
        expected = self.info.returns.type_str
        type_error.add_note(f'cannot return value: expected {expected}, got {actual}')
        raise type_error

    ############################################################################

    # we *do* allow RPC to occur when formatting the outer, single value
    def fmt_display_arg(self, value: Any, /) -> str:
        cls = type(value)
        if cls is NoneType:
            return 'None'
        if cls is type or isinstance(value, type):
            return f'<class {cls.__name__!r}>'
        if cls in (bool, int, float, NoneType):
            return repr(value)
        return safe_repr(value, True, self.max_repr_len * 2)

    # we do not allow RPC to occur for general `repr`
    def fmt_repr(self, value: object) -> str:
        cls = type(value)
        if cls is NoneType:
            return 'None'
        if cls is type or isinstance(value, type):
            return f'<class {cls.__name__!r}>'
        if cls in (bool, int, float, NoneType):
            return repr(value)
        return safe_repr(value, False, self.max_repr_len)

    # def fmt_doc_str(self, doc_str: str | None) -> str | None:
    #     return _format_docstring(doc_str) if type(doc_str) is str else None

    ############################################################################

    def get_loaded_modules(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.vars.globals.items() if isinstance(v, ModuleType))

    ############################################################################

    def make_stub_str(self, ns: Vars, max_lines: int | None = None) -> str:
        try:
            text, _ = emit_stubs(
                ns, max_lines=max_lines, exclude_names=self.vars.hidden, sort_items=False
            )
            assert isinstance(text, str)
            return text
        except Exception as error:
            self.log('error computing stubs:', error)
            return ''

    def make_anno_only_stub_str(self, annos: dict[str, str]) -> str:
        return '\n'.join(f'{k}: {v}' for k, v in annos.items())

    # --------------------------------------------------------------------------

    # if we wished to keep these, we should move these to just be dataclasses like
    # `ReplCallableInfo`, and use the same mechanism as they do. they are *very*
    # similar to `ReplVarInfo` and `ReplCallableInfo`, which would just need a
    # `json_schema` field to subsume this functionality

    # def get_schema_info_object(self, obj_name: str) -> str:
    #     schema_result = repl_get_schema_info_object(self.uuid, self.symbols, obj_name)
    #     return schema_result_to_str(schema_result)
    #
    # def get_schema_info_callable(self, callable_name: str) -> str:
    #     schema_result = repl_get_schema_info_callable(self.uuid, self.symbols, callable_name)
    #     return schema_result_to_str(schema_result)
    #
    # def execute_callable(self, obj_name: str, content: str, assign_to: str) -> str:
    #     schema_result = repl_execute_callable(self.uuid, self.symbols, obj_name, content, assign_to)
    #     return schema_result_to_str(schema_result)


NONE = object()

# these are triggered when the `.initialize` is called and these are present in
# locals or globals
REPL_VAR_TABLE: list[tuple[str, Callable]] = [
    (REPL_VAR.ROLE, AgentRepl.set_role),
    (REPL_VAR.TASK_DESCRIPTION, AgentRepl.set_task_description),
    (REPL_VAR.SELF_FN, AgentRepl.set_agentic_fn),
    (REPL_VAR.RETURN_TYPE, AgentRepl.set_return_type),
    (REPL_VAR.MAX_REPR_LEN, AgentRepl.set_max_repr_len),
]

REPL_VAR_NAMES = [name for name, _ in REPL_VAR_TABLE]

# so that tracebacks avoid frames associated with this directory
register_repl_path(__file__)

SPECIAL_PATCHED_ATTR = '___patched_module___'
